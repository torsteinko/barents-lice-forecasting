from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from .config import (
    LATEST_MASTER_TABLE_PATH,
    LATEST_SITE_MAP_PATH,
    MASTER_TABLE_PATH,
    SITE_MAP_PATH,
)

try:
    import google.auth
except ImportError:
    google = None

try:
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.agent_toolkits.sql.toolkit import (
        SQLDatabaseToolkit as BaseSQLDatabaseToolkit,
    )
    from langchain_community.tools.sql_database.tool import (
        InfoSQLDatabaseTool,
        ListSQLDatabaseTool,
        QuerySQLCheckerTool,
        QuerySQLDatabaseTool as BaseQuerySQLDatabaseTool,
    )
    from langchain_community.utilities import SQLDatabase
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    create_sql_agent = None
    BaseSQLDatabaseToolkit = None
    BaseQuerySQLDatabaseTool = None
    InfoSQLDatabaseTool = None
    ListSQLDatabaseTool = None
    QuerySQLCheckerTool = None
    SQLDatabase = None
    ChatVertexAI = None


DEFAULT_CHAT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_SITE_CARD_METRIC = "near_term_risk"
DEFAULT_SQL_TOP_K = 10
SITE_CARD_METRIC_LABELS = {
    "near_term_risk": "Near-term breach risk",
    "classifier_1w_score": "1-week breach risk",
    "classifier_2w_score": "2-week breach risk",
    "classifier_12w_score": "12-week breach risk",
    "femaleadult_to_limit_ratio": "Current raw-week ratio",
    "currently_over_limit": "Currently over limit",
}
READ_ONLY_ALLOWED_LEADING_KEYWORDS = {"select", "with"}
READ_ONLY_BLOCKED_KEYWORDS = {
    "alter",
    "analyze",
    "attach",
    "call",
    "comment",
    "commit",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "execute",
    "export",
    "grant",
    "insert",
    "install",
    "load",
    "merge",
    "pragma",
    "prepare",
    "replace",
    "revoke",
    "rollback",
    "set",
    "show",
    "truncate",
    "update",
    "use",
    "vacuum",
}
READ_ONLY_BLOCKED_FUNCTIONS = {
    "glob",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_text",
    "sniff_csv",
}


class AgentResultPayload(BaseModel):
    answer: str = Field(default="")
    sitenumbers: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


if BaseQuerySQLDatabaseTool is not None and BaseSQLDatabaseToolkit is not None:

    class ReadOnlyQuerySQLDatabaseTool(BaseQuerySQLDatabaseTool):
        name: str = "sql_db_query"
        description: str = (
            "Execute exactly one read-only SELECT or WITH SQL query against the database. "
            "Only the master_table and site_snapshot relations are allowed. "
            "If an error is returned, rewrite the query, re-check it, and try again."
        )

        def _run(self, query: str, run_manager: Any = None) -> Any:
            violation = _validate_read_only_query(query)
            if violation is not None:
                return f"Error: {violation}"
            return self.db.run_no_throw(query, include_columns=True)

    class ReadOnlySQLDatabaseToolkit(BaseSQLDatabaseToolkit):
        def get_tools(self) -> list[Any]:
            list_tool = ListSQLDatabaseTool(db=self.db)
            info_tool = InfoSQLDatabaseTool(
                db=self.db,
                description=(
                    "Input to this tool is a comma-separated list of tables, output is the "
                    "schema and sample rows for those tables. Be sure that the tables exist "
                    f"by calling {list_tool.name} first. Example input: table1, table2"
                ),
            )
            query_tool = ReadOnlyQuerySQLDatabaseTool(db=self.db)
            query_checker_tool = QuerySQLCheckerTool(
                db=self.db,
                llm=self.llm,
                description=(
                    "Use this tool to double check if your query is correct before executing "
                    f"it. Always use this tool before executing a query with {query_tool.name}."
                ),
            )
            return [query_tool, info_tool, list_tool, query_checker_tool]

else:
    ReadOnlySQLDatabaseToolkit = None


class SiteChatService:
    def __init__(
        self,
        geojson_path: Path | None = None,
        master_table_path: Path | None = None,
    ) -> None:
        self._requested_geojson_path = geojson_path
        self._requested_master_table_path = master_table_path
        self.default_geojson_path = SITE_MAP_PATH
        self.latest_geojson_path = LATEST_SITE_MAP_PATH
        self.default_master_table_path = MASTER_TABLE_PATH
        self.latest_master_table_path = LATEST_MASTER_TABLE_PATH
        self._cached_geojson: dict[str, object] | None = None
        self._cached_frame: pd.DataFrame | None = None
        self._cached_site_index: pd.DataFrame | None = None
        self._cached_mtime: float | None = None
        self._cached_parquet_mtime: float | None = None
        self._cached_parquet_columns: set[str] = set()
        self._parquet_schema_error: str | None = None
        self._cached_llm: Any | None = None
        self._cached_llm_config: tuple[str, str, str] | None = None
        self._cached_extractor: Any | None = None
        self._last_llm_error: str | None = None
        self._case_cutoff_date: pd.Timestamp | None = None
        self._latest_raw_week_label: str | None = None
        self._forecast_anchor_week_label: str | None = None
        self._active_geojson_path: Path | None = None
        self._active_master_table_path: Path | None = None

    def get_geojson(self) -> dict[str, object]:
        self._ensure_loaded()
        return self._cached_geojson or {"type": "FeatureCollection", "features": []}

    def get_site_count(self) -> int:
        self._ensure_loaded()
        if self._cached_frame is None:
            return 0
        return int(len(self._cached_frame))

    def get_llm_status(self) -> dict[str, object]:
        geojson_path, master_table_path = self._resolve_data_paths()
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        model_name = os.getenv("VERTEX_GEMINI_MODEL", DEFAULT_CHAT_MODEL)
        adc_project = None
        adc_error = None

        if google is not None:
            try:
                _, adc_project = google.auth.default()
            except Exception as exc:
                adc_error = _truncate_error(exc)
        else:
            adc_error = "google-auth is not installed"

        configured = bool(
            project
            and location
            and ChatVertexAI is not None
            and create_sql_agent is not None
            and SQLDatabase is not None
            and ReadOnlySQLDatabaseToolkit is not None
        )
        return {
            "provider": (
                "vertex-gemini" if ChatVertexAI is not None else "gemini-unavailable"
            ),
            "configured": configured,
            "project": project,
            "location": location,
            "model": model_name,
            "adc_project": adc_project,
            "adc_available": adc_error is None,
            "adc_error": adc_error,
            "last_error": self._last_llm_error,
            "site_snapshot_path": str(geojson_path),
            "chat_dataset_path": str(master_table_path),
        }

    def answer_question(
        self,
        message: str,
        selected_site_id: str | None = None,
        visible_site_ids: list[str] | None = None,
    ) -> dict[str, object]:
        self._ensure_loaded()
        message_text = str(message or "").strip()

        if not message_text:
            return self._build_failure_response(
                "Chat messages cannot be empty.",
                selected_site_id=selected_site_id,
                visible_site_ids=visible_site_ids,
            )

        if self._cached_frame is None:
            return self._build_failure_response(
                "The site snapshot is not available right now.",
                selected_site_id=selected_site_id,
                visible_site_ids=visible_site_ids,
            )

        master_table_path = (
            self._active_master_table_path or self._resolve_data_paths()[1]
        )
        if not master_table_path.exists():
            self._last_llm_error = f"Missing chat dataset at {master_table_path}"
            return self._build_failure_response(
                "The chat dataset is not available right now.",
                selected_site_id=selected_site_id,
                visible_site_ids=visible_site_ids,
            )

        if self._parquet_schema_error is not None:
            self._last_llm_error = self._parquet_schema_error
            return self._build_failure_response(
                "The chat dataset could not be inspected safely.",
                selected_site_id=selected_site_id,
                visible_site_ids=visible_site_ids,
            )

        self._last_llm_error = None
        metric_key = _infer_site_card_metric(message_text)
        agent_message = _augment_chat_question(message_text)
        engine = None
        try:
            engine = self._create_duckdb_engine()
            database = self._create_sql_database(engine)
            raw_output = self._run_sql_agent(agent_message, database)
            parsed = self._parse_agent_result(agent_message, raw_output)
            if _looks_like_interrupted_answer(parsed.answer):
                retry_message = _build_retry_chat_question(message_text)
                retry_output = self._run_sql_agent(retry_message, database)
                retry_parsed = self._parse_agent_result(retry_message, retry_output)
                if not _looks_like_interrupted_answer(retry_parsed.answer):
                    parsed = retry_parsed
        except Exception as exc:
            self._last_llm_error = _truncate_error(exc)
            return self._build_failure_response(
                "The SQL chat agent could not answer this request.",
                selected_site_id=selected_site_id,
                visible_site_ids=visible_site_ids,
            )
        finally:
            if engine is not None:
                engine.dispose()

        return {
            "answer": parsed.answer,
            "used_llm": True,
            "metric_key": metric_key,
            "metric_label": SITE_CARD_METRIC_LABELS.get(
                metric_key, "Current snapshot context"
            ),
            "filters_applied": [],
            "proxy_note": None,
            "sites": self._hydrate_sites(parsed.sitenumbers, metric_key),
            "llm": self.get_llm_status(),
        }

    def _resolve_data_paths(self) -> tuple[Path, Path]:
        if (
            self._requested_geojson_path is not None
            or self._requested_master_table_path is not None
        ):
            geojson_path = self._requested_geojson_path or self.default_geojson_path
            master_table_path = (
                self._requested_master_table_path or self.default_master_table_path
            )
            return geojson_path, master_table_path

        if self.latest_geojson_path.exists() and self.latest_master_table_path.exists():
            return self.latest_geojson_path, self.latest_master_table_path

        return self.default_geojson_path, self.default_master_table_path

    def _ensure_loaded(self) -> None:
        geojson_path, master_table_path = self._resolve_data_paths()

        if not geojson_path.exists():
            raise FileNotFoundError(f"Missing map dataset at {geojson_path}")

        current_mtime = geojson_path.stat().st_mtime
        parquet_mtime = (
            master_table_path.stat().st_mtime if master_table_path.exists() else None
        )
        if (
            self._cached_geojson is not None
            and self._cached_frame is not None
            and self._cached_mtime == current_mtime
            and self._cached_parquet_mtime == parquet_mtime
            and self._active_geojson_path == geojson_path
            and self._active_master_table_path == master_table_path
        ):
            return

        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        metadata = geojson.get("metadata", {}) if isinstance(geojson, dict) else {}
        case_cutoff_date = _parse_case_cutoff_date(geojson)
        records: list[dict[str, object]] = []
        for feature in geojson.get("features", []):
            properties = dict(feature.get("properties", {}))
            coordinates = feature.get("geometry", {}).get("coordinates", [None, None])
            properties["longitude"] = coordinates[0]
            properties["latitude"] = coordinates[1]
            properties["site_id"] = str(properties.get("sitenumber"))
            records.append(properties)

        frame = pd.DataFrame(records)
        numeric_columns = [
            "latitude",
            "longitude",
            "femaleadult",
            "mobilelice",
            "persistentlice",
            "licelimitweek",
            "femaleadult_to_limit_ratio",
            "seatemperature",
            "neighbor_site_count",
            "neighbor_breach_this_week_lag1",
            "neighbor_femaleadult_to_limit_ratio_lag1",
            "pa_breach_rate_lag1",
            "pa_treatment_rate_lag1",
            "classifier_1w_score",
            "classifier_2w_score",
            "classifier_12w_score",
            "count_1w_prediction",
            "count_2w_prediction",
            "count_12w_prediction",
            "current_limit_excess",
            "weeks_since_any_treatment",
            "weeks_since_last_breach",
            "weeks_since_last_counted",
            "treatment_count",
            "cleanerfish_treatment_count",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        for column in [
            "currently_over_limit",
            "likelynofish",
            "havecountedlice",
            "any_treatment",
            "breach_this_week",
        ]:
            if column in frame.columns:
                frame[column] = frame[column].astype("boolean")

        near_term_columns = [
            column
            for column in ["classifier_1w_score", "classifier_2w_score"]
            if column in frame.columns
        ]
        frame["near_term_risk"] = (
            frame[near_term_columns].max(axis=1, skipna=True)
            if near_term_columns
            else np.nan
        )

        near_term_count_columns = [
            column
            for column in ["count_1w_prediction", "count_2w_prediction"]
            if column in frame.columns
        ]
        frame["near_term_count_prediction"] = (
            frame[near_term_count_columns].max(axis=1, skipna=True)
            if near_term_count_columns
            else np.nan
        )

        frame["site_label"] = frame.get("sitename", pd.Series(dtype=object)).fillna("")
        frame["site_label"] = (
            frame["site_label"].replace("", np.nan).fillna(frame["site_id"])
        )

        parquet_columns: set[str] = set()
        parquet_schema_error = None
        if master_table_path.exists():
            try:
                parquet_columns = set(pq.ParquetFile(master_table_path).schema.names)
            except Exception as exc:
                parquet_schema_error = _truncate_error(exc)

        site_index = (
            frame.drop_duplicates("site_id").set_index("site_id", drop=False)
            if not frame.empty and "site_id" in frame.columns
            else pd.DataFrame()
        )

        self._cached_geojson = geojson
        self._cached_frame = frame
        self._cached_site_index = site_index
        self._cached_mtime = current_mtime
        self._cached_parquet_mtime = parquet_mtime
        self._cached_parquet_columns = parquet_columns
        self._parquet_schema_error = parquet_schema_error
        self._case_cutoff_date = case_cutoff_date
        self._latest_raw_week_label = _safe_metadata_text(
            metadata.get("latest_raw_week_label")
            if isinstance(metadata, dict)
            else None
        )
        self._forecast_anchor_week_label = _safe_metadata_text(
            metadata.get("forecast_anchor_week_label")
            if isinstance(metadata, dict)
            else None
        )
        self._active_geojson_path = geojson_path
        self._active_master_table_path = master_table_path

    def _create_duckdb_engine(self) -> Any:
        return create_engine(
            "duckdb:///:memory:",
            poolclass=StaticPool,
            future=True,
        )

    def _create_sql_database(
        self,
        engine: Any,
    ) -> Any:
        master_table_path = (
            self._active_master_table_path or self._resolve_data_paths()[1]
        )
        parquet_path = master_table_path.resolve().as_posix()
        with engine.begin() as connection:
            connection.exec_driver_sql(self._build_master_view_sql(parquet_path))
            self._build_site_snapshot_sql_frame().to_sql(
                "site_snapshot",
                connection,
                if_exists="replace",
                index=False,
            )

        return SQLDatabase(
            engine=engine,
            include_tables=["master_table", "site_snapshot"],
            view_support=True,
            sample_rows_in_table_info=2,
            max_string_length=500,
        )

    def _build_site_snapshot_sql_frame(self) -> pd.DataFrame:
        if self._cached_frame is None:
            return pd.DataFrame()

        frame = self._cached_frame.copy().astype(object)
        return frame.where(pd.notna(frame), None)

    def _build_master_view_sql(self, parquet_path: str) -> str:
        where_clause = self._build_case_cutoff_where_clause()
        return (
            "CREATE OR REPLACE VIEW master_table AS "
            f"SELECT * FROM read_parquet({_quote_sql_literal(parquet_path)}){where_clause}"
        )

    def _build_case_cutoff_where_clause(self) -> str:
        if (
            self._case_cutoff_date is not None
            and "week_start_date" in self._cached_parquet_columns
        ):
            cutoff = self._case_cutoff_date.date().isoformat()
            return f" WHERE CAST(week_start_date AS DATE) <= DATE {_quote_sql_literal(cutoff)}"
        if "year" in self._cached_parquet_columns:
            return " WHERE CAST(year AS INTEGER) <= 2025"
        return ""

    def _run_sql_agent(
        self,
        message: str,
        database: Any,
    ) -> str:
        llm = self._get_llm()
        toolkit = ReadOnlySQLDatabaseToolkit(db=database, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type="tool-calling",
            prefix=self._build_agent_prefix(),
            suffix=self._build_agent_suffix(),
            top_k=DEFAULT_SQL_TOP_K,
            max_iterations=10,
            verbose=False,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        result = agent_executor.invoke({"input": self._build_agent_input(message)})
        output = result.get("output") if isinstance(result, dict) else None
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("SQL agent returned no answer text")
        return output.strip()

    def _build_agent_prefix(self) -> str:
        cutoff_text = (
            self._case_cutoff_date.date().isoformat()
            if self._case_cutoff_date is not None
            else "the validated pre-2026 case window"
        )
        return (
            "You are an expert aquaculture analyst for Mowi. "
            "You are designed to interact with a SQL database that contains aquaculture observations and forecasts. "
            "Base every claim strictly on the database results you retrieve. If the data does not support a claim, say so rather than guessing.\n\n"
            "Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. "
            "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} rows. "
            "You can order the results by a relevant column to return the most useful examples. "
            "Never query for every column from a table; ask only for the columns needed to answer the question.\n\n"
            "You have access to tools for interacting with the database. Only use those tools. "
            "You MUST use the SQL checker tool before executing a query. If a query returns an error, rewrite it and try again.\n\n"
            "Security and scope rules:\n"
            "- The database is read-only. Never attempt INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, EXPORT, ATTACH, DETACH, INSTALL, LOAD, SET, PRAGMA, or any other write or admin statement.\n"
            "- Use site_snapshot for questions about right now, current status, the latest raw reporting week, current ratios, current counts, or current forecast cards. site_snapshot has one row per site.\n"
            "- Use master_table for historical site-week analysis, trends, or week-by-week reasoning across time.\n"
            f"- master_table is capped at the forecast anchor date {cutoff_text}. Treat any later week-level history as out of scope in master_table.\n"
            "- site_snapshot can be newer than master_table because it reflects the latest raw operational week plus the current forecast snapshot.\n\n"
            "- No visible-only site subset is provided to chat in this build. If the user says visible sites, interpret that as the full current site_snapshot and answer from the available data instead of claiming the search was interrupted.\n"
            "- For questions about sites already above the lice limit right now, use currently_over_limit from site_snapshot. If no rows match, answer that there are currently no sites identified as above the lice limit in the latest site snapshot.\n\n"
            "When you are done, your final answer MUST be a valid JSON object with this exact shape: "
            '{{"answer":"...","sitenumbers":["12345","67890"]}}.\n'
            "- The answer field must contain the user-facing answer text.\n"
            "- The sitenumbers field must contain only relevant sitenumber values from the database, converted to strings. Use an empty list when no sites are relevant.\n"
            "- Do not wrap the final JSON in code fences.\n"
            "- Never say the search was interrupted or ask the user to try again unless the database genuinely has no relevant rows.\n"
            '- If the question is unrelated to the database, return JSON with answer set to "I don\'t know" and an empty sitenumbers list.'
        )

    def _build_agent_suffix(self) -> str:
        return (
            "I should list the tables first, inspect the schema for the relevant tables, "
            "double-check any query before execution, and then return only the final JSON object."
        )

    def _build_agent_input(self, message: str) -> str:
        parts = [f"User question: {message}"]
        if self._latest_raw_week_label:
            parts.append(
                f"Latest raw site snapshot week: {self._latest_raw_week_label}."
            )
        if self._forecast_anchor_week_label:
            parts.append(f"Forecast anchor week: {self._forecast_anchor_week_label}.")
        parts.append(
            "Use site_snapshot for current/latest/right-now questions. Use master_table for historical weekly questions."
        )
        parts.append(
            "Return only the final JSON object when you have enough information."
        )
        return "\n".join(parts)

    def _get_llm(self) -> Any:
        if (
            ChatVertexAI is None
            or create_sql_agent is None
            or SQLDatabase is None
            or ReadOnlySQLDatabaseToolkit is None
        ):
            raise RuntimeError(
                "LangChain SQL agent dependencies are not installed in this environment"
            )

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        model_name = os.getenv("VERTEX_GEMINI_MODEL", DEFAULT_CHAT_MODEL)
        if not project or not location:
            raise RuntimeError(
                "Missing GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION in the process environment"
            )

        config = (project, location, model_name)
        if self._cached_llm is None or self._cached_llm_config != config:
            self._cached_llm = ChatVertexAI(
                model=model_name,
                project=project,
                location=location,
                temperature=0,
                max_retries=3,
            )
            self._cached_llm_config = config
            self._cached_extractor = None
        return self._cached_llm

    def _get_output_extractor(self) -> Any:
        if self._cached_extractor is None:
            self._cached_extractor = self._get_llm().with_structured_output(
                AgentResultPayload
            )
        return self._cached_extractor

    def _parse_agent_result(self, message: str, raw_output: str) -> AgentResultPayload:
        parsed_payload = _load_json_like_payload(raw_output)
        if parsed_payload is not None:
            return _coerce_agent_payload(parsed_payload, raw_output)

        try:
            extracted = self._get_output_extractor().invoke(
                "Extract the final user-facing answer and relevant sitenumbers from the SQL agent output below. "
                "Only include sitenumbers that are explicitly supported by the output.\n\n"
                f"User question:\n{message}\n\n"
                f"SQL agent output:\n{raw_output}"
            )
            return _coerce_agent_payload(extracted, raw_output)
        except Exception:
            answer = raw_output.strip() or "I couldn't produce a usable answer."
            return AgentResultPayload(answer=answer, sitenumbers=[])

    def _hydrate_sites(
        self,
        sitenumbers: Sequence[str],
        metric_key: str,
    ) -> list[dict[str, object]]:
        if self._cached_site_index is None or self._cached_site_index.empty:
            return []

        site_cards: list[dict[str, object]] = []
        seen: set[str] = set()
        for site_id in _normalize_site_ids(sitenumbers):
            if site_id in seen or site_id not in self._cached_site_index.index:
                continue
            row = self._cached_site_index.loc[site_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            site_cards.append(self._serialize_site(row, metric_key))
            seen.add(site_id)
        return site_cards

    def _build_scope_notes(
        self,
        selected_site_id: str | None,
        visible_site_ids: Sequence[str] | None,
    ) -> list[str]:
        notes: list[str] = []
        if visible_site_ids is None:
            notes.append("map scope: full dataset")
        else:
            visible_count = len(_normalize_site_ids(visible_site_ids))
            if visible_count:
                notes.append(f"map scope: {visible_count} visible site(s)")
            else:
                notes.append("map scope: no visible sites")
        if selected_site_id:
            notes.append(f"selected site: {selected_site_id}")
        return notes

    def _build_scope_note(self, visible_site_ids: Sequence[str] | None) -> str | None:
        if visible_site_ids is None:
            return None
        visible_count = len(_normalize_site_ids(visible_site_ids))
        if visible_count:
            return (
                "SQL chat defaults to visible_master_table for the current map scope. "
                "Ask explicitly for all sites to broaden the query."
            )
        return "The current map scope has no visible sites, so visible_master_table is empty unless the question explicitly asks for all sites."

    def _describe_visible_scope(self, visible_site_ids: Sequence[str] | None) -> str:
        if visible_site_ids is None:
            return "no explicit visible-site subset was provided, so visible_master_table mirrors master_table"
        visible_count = len(_normalize_site_ids(visible_site_ids))
        if visible_count:
            return f"visible_master_table contains {visible_count} visible site(s) from the current map state"
        return "visible_master_table is empty because the current map state has no visible sites"

    def _build_failure_response(
        self,
        answer: str,
        *,
        selected_site_id: str | None,
        visible_site_ids: Sequence[str] | None,
    ) -> dict[str, object]:
        return {
            "answer": answer,
            "used_llm": False,
            "metric_key": DEFAULT_SITE_CARD_METRIC,
            "metric_label": "current snapshot context",
            "filters_applied": [],
            "proxy_note": None,
            "sites": [],
            "llm": self.get_llm_status(),
        }

    def _serialize_site(self, row: pd.Series, metric_key: str) -> dict[str, object]:
        metric_value = row.get(metric_key)
        return {
            "site_id": str(row.get("site_id")),
            "sitename": _jsonify(row.get("site_label")),
            "productionarea": _jsonify(row.get("productionarea")),
            "county": _jsonify(row.get("county")),
            "municipality": _jsonify(row.get("municipality")),
            "latitude": _jsonify(row.get("latitude")),
            "longitude": _jsonify(row.get("longitude")),
            "coordinates_text": _format_coordinates(
                row.get("latitude"), row.get("longitude")
            ),
            "metric_key": metric_key,
            "metric_value": _jsonify(metric_value),
            "metric_display": _format_metric_value(metric_key, metric_value),
            "metric_label": SITE_CARD_METRIC_LABELS.get(
                metric_key, "Current snapshot context"
            ),
            "femaleadult": _jsonify(row.get("femaleadult")),
            "femaleadult_to_limit_ratio": _jsonify(
                row.get("femaleadult_to_limit_ratio")
            ),
            "currently_over_limit": _jsonify(row.get("currently_over_limit")),
            "near_term_risk": _jsonify(row.get("near_term_risk")),
            "classifier_1w_score": _jsonify(row.get("classifier_1w_score")),
            "classifier_2w_score": _jsonify(row.get("classifier_2w_score")),
            "classifier_12w_score": _jsonify(row.get("classifier_12w_score")),
            "count_1w_prediction": _jsonify(row.get("count_1w_prediction")),
            "count_2w_prediction": _jsonify(row.get("count_2w_prediction")),
            "count_12w_prediction": _jsonify(row.get("count_12w_prediction")),
            "year": _jsonify(row.get("year")),
            "week": _jsonify(row.get("week")),
            "latest_reporting_week_label": _jsonify(
                row.get("latest_reporting_week_label")
            ),
            "last_counted_week_label": _jsonify(row.get("last_counted_week_label")),
            "last_treatment_week_label": _jsonify(row.get("last_treatment_week_label")),
            "last_treatment_action": _jsonify(row.get("last_treatment_action")),
            "last_treatment_activeingredient": _jsonify(
                row.get("last_treatment_activeingredient")
            ),
        }


def _validate_read_only_query(query: str) -> str | None:
    stripped = _strip_sql_comments(query).strip()
    if not stripped:
        return "Query was empty."

    compact = stripped.rstrip().rstrip(";").strip()
    if ";" in compact:
        return "Only a single SQL statement is allowed."

    normalized = re.sub(r"\s+", " ", compact.lower())
    leading_match = re.match(r"^([a-z_]+)", normalized)
    if leading_match is None:
        return "Only a read-only SELECT or WITH query is allowed."

    leading_keyword = leading_match.group(1)
    if leading_keyword not in READ_ONLY_ALLOWED_LEADING_KEYWORDS:
        return "Only a read-only SELECT or WITH query is allowed."

    blocked_keyword_pattern = (
        r"\b(" + "|".join(sorted(READ_ONLY_BLOCKED_KEYWORDS)) + r")\b"
    )
    blocked_keyword = re.search(blocked_keyword_pattern, normalized)
    if blocked_keyword is not None:
        return f"The query contains a blocked keyword: {blocked_keyword.group(1)}."

    blocked_function_pattern = (
        r"\b(" + "|".join(sorted(READ_ONLY_BLOCKED_FUNCTIONS)) + r")\s*\("
    )
    blocked_function = re.search(blocked_function_pattern, normalized)
    if blocked_function is not None:
        return f"The query contains a blocked function: {blocked_function.group(1)}."

    allowed_relations = {"master_table", "site_snapshot"}
    cte_normalized = normalized.replace("with recursive ", "with ")
    cte_names = {
        match.group(1)
        for match in re.finditer(
            r"(?:with|,)\s*([a-z_][\w]*)\s+as\s*\(", cte_normalized
        )
    }
    allowed_relations.update(cte_names)

    for token in re.findall(r"\b(?:from|join)\s+([^\s,]+)", normalized):
        candidate = token.strip().rstrip(")").rstrip(";")
        if not candidate or candidate.startswith("("):
            continue
        relation_name = candidate.split(".")[-1].strip('"')
        if relation_name not in allowed_relations:
            return "Queries may only read from the master_table or site_snapshot relations."

    return None


def _strip_sql_comments(query: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", " ", query, flags=re.S)
    return re.sub(r"--.*?$", " ", without_block_comments, flags=re.M)


def _load_json_like_payload(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    candidates.extend(
        match.strip()
        for match in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
        if match.strip()
    )

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_agent_payload(payload: Any, raw_output: str) -> AgentResultPayload:
    if isinstance(payload, AgentResultPayload):
        return AgentResultPayload(
            answer=payload.answer.strip(),
            sitenumbers=_normalize_site_ids(payload.sitenumbers),
        )

    if isinstance(payload, BaseModel):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = payload
    else:
        data = {}

    answer = str(data.get("answer") or raw_output or "").strip()
    if not answer:
        answer = "I couldn't produce a usable answer."

    sitenumber_values = data.get("sitenumbers")
    if sitenumber_values is None:
        sitenumber_values = data.get("sites")

    coerced_sitenumbers: list[str] = []
    if isinstance(sitenumber_values, list):
        for value in sitenumber_values:
            if isinstance(value, dict):
                site_value = (
                    value.get("sitenumber") or value.get("site_id") or value.get("site")
                )
                if site_value is not None:
                    coerced_sitenumbers.append(str(site_value))
            elif value is not None:
                coerced_sitenumbers.append(str(value))
    elif sitenumber_values is not None:
        coerced_sitenumbers.append(str(sitenumber_values))

    return AgentResultPayload(
        answer=answer,
        sitenumbers=_normalize_site_ids(coerced_sitenumbers),
    )


def _normalize_site_ids(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        site_id = str(value or "").strip()
        if not site_id or site_id in seen:
            continue
        normalized.append(site_id)
        seen.add(site_id)
    return normalized


def _quote_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _safe_metadata_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_case_cutoff_date(geojson: dict[str, object]) -> pd.Timestamp | None:
    metadata = geojson.get("metadata", {}) if isinstance(geojson, dict) else {}
    raw_value = metadata.get("case_cutoff_date") if isinstance(metadata, dict) else None
    if not raw_value:
        return None
    parsed = pd.to_datetime(raw_value, errors="coerce")
    return parsed if not pd.isna(parsed) else None


def _format_coordinates(latitude: object, longitude: object) -> str:
    if latitude is None or longitude is None:
        return "--"
    try:
        return f"{float(latitude):.4f}, {float(longitude):.4f}"
    except (TypeError, ValueError):
        return "--"


def _format_metric_value(metric_key: str, value: object) -> str:
    numeric = _jsonify(value)
    if numeric is None:
        return "--"
    if isinstance(numeric, bool):
        return "Yes" if numeric else "No"
    if metric_key in {
        "near_term_risk",
        "classifier_1w_score",
        "classifier_2w_score",
        "classifier_12w_score",
    }:
        return f"{numeric * 100:.0f}%"
    if metric_key.endswith("ratio"):
        return f"{numeric:.2f}x"
    return f"{numeric:.2f}"


def _augment_chat_question(message: str) -> str:
    normalized = str(message or "").strip()
    if not normalized:
        return normalized

    notes: list[str] = []
    lowered = normalized.lower()
    if "visible" in lowered:
        notes.append(
            "Clarification: chat does not receive a visible-only map subset in this build, so interpret visible sites as the full current site_snapshot."
        )
    if (
        "above the lice limit" in lowered
        or "over the lice limit" in lowered
        or "already above" in lowered
    ):
        notes.append(
            "Clarification: for right-now lice-limit questions, use currently_over_limit from site_snapshot. If no rows match, answer that there are currently no sites identified as above the lice limit in the latest site snapshot."
        )
    if not notes:
        return normalized
    return normalized + "\n\n" + "\n".join(notes)


def _build_retry_chat_question(message: str) -> str:
    base = _augment_chat_question(message)
    retry_note = (
        "Retry instruction: answer directly from the available tables. Do not claim the search was interrupted. "
        "If the user asks about visible sites, answer from site_snapshot across all sites in the current snapshot. "
        "If no rows match currently_over_limit, say there are currently no sites identified as above the lice limit in the latest site snapshot."
    )
    return base + "\n\n" + retry_note


def _looks_like_interrupted_answer(answer: str) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return True
    interruption_markers = [
        "search process was interrupted",
        "please try your request again",
        "could not retrieve the information",
        "i am sorry, but i could not retrieve",
    ]
    return any(marker in text for marker in interruption_markers)


def _infer_site_card_metric(message: str) -> str:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return DEFAULT_SITE_CARD_METRIC
    if "ratio" in normalized or "to limit" in normalized:
        return "femaleadult_to_limit_ratio"
    if (
        "currently over limit" in normalized
        or "above the limit" in normalized
        or "above the lice limit" in normalized
        or "over the lice limit" in normalized
    ):
        return "currently_over_limit"
    if "12w" in normalized or "12-week" in normalized or "12 week" in normalized:
        return "classifier_12w_score"
    if "2w" in normalized or "2-week" in normalized or "2 week" in normalized:
        return "classifier_2w_score"
    if "1w" in normalized or "1-week" in normalized or "1 week" in normalized:
        return "classifier_1w_score"
    if "near-term" in normalized or "near term" in normalized:
        return "near_term_risk"
    return DEFAULT_SITE_CARD_METRIC


def _truncate_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message if len(message) <= 400 else f"{message[:397]}..."


def _jsonify(value: object) -> object | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
