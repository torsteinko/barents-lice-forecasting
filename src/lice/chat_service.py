from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RESULTS_DIR

try:
    import google.auth
except ImportError:
    google = None

try:
    from google import genai
except ImportError:
    genai = None


DEFAULT_CHAT_MODEL = "gemini-2.5-flash"


@dataclass
class QueryPlan:
    frame: pd.DataFrame
    metric_key: str
    metric_label: str
    descending: bool
    filters_applied: list[str]
    proxy_note: str | None
    top_n: int


@dataclass
class AnswerContext:
    ranked: pd.DataFrame
    question_type: str
    candidate_mode: str
    summary: str
    rationale: list[str]
    exact_match_count: int | None = None
    supporting_title: str | None = None
    supporting_rows: list[dict[str, object]] | None = None
    window_note: str | None = None


class SiteChatService:
    def __init__(
        self,
        geojson_path: Path | None = None,
        master_table_path: Path | None = None,
    ) -> None:
        self.geojson_path = geojson_path or RESULTS_DIR / "site_map.geojson"
        self.master_table_path = master_table_path or PROCESSED_DIR / "master_table.parquet"
        self._cached_geojson: dict[str, object] | None = None
        self._cached_frame: pd.DataFrame | None = None
        self._cached_history_frame: pd.DataFrame | None = None
        self._cached_mtime: float | None = None
        self._cached_history_mtime: float | None = None
        self._cached_client = None
        self._last_llm_error: str | None = None
        self._case_cutoff_date: pd.Timestamp | None = None

    def get_geojson(self) -> dict[str, object]:
        self._ensure_loaded()
        return self._cached_geojson or {"type": "FeatureCollection", "features": []}

    def get_site_count(self) -> int:
        self._ensure_loaded()
        if self._cached_frame is None:
            return 0
        return int(len(self._cached_frame))

    def get_llm_status(self) -> dict[str, object]:
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

        configured = bool(project and location)
        return {
            "provider": "vertex-gemini" if genai is not None else "fallback-only",
            "configured": configured,
            "project": project,
            "location": location,
            "model": model_name,
            "adc_project": adc_project,
            "adc_available": adc_error is None,
            "adc_error": adc_error,
            "last_error": self._last_llm_error,
        }

    def answer_question(
        self,
        message: str,
        selected_site_id: str | None = None,
        visible_site_ids: list[str] | None = None,
    ) -> dict[str, object]:
        self._ensure_loaded()
        assert self._cached_frame is not None

        working = self._cached_frame.copy()
        if visible_site_ids:
            visible_ids = {str(site_id) for site_id in visible_site_ids}
            working = working[working["site_id"].isin(visible_ids)].copy()

        query_plan = self._build_query_plan(message, working)
        answer_context = self._build_answer_context(message, query_plan)
        ranked = answer_context.ranked

        if ranked.empty and not answer_context.supporting_rows:
            return {
                "answer": "No sites matched the current question and the active map filters. Try widening the area filter or removing extra constraints.",
                "used_llm": False,
                "metric_key": query_plan.metric_key,
                "metric_label": query_plan.metric_label,
                "filters_applied": query_plan.filters_applied,
                "proxy_note": query_plan.proxy_note,
                "sites": [],
                "llm": self.get_llm_status(),
            }

        selected_site = None
        if selected_site_id and not ranked.empty:
            selected_rows = ranked[ranked["site_id"] == str(selected_site_id)]
            if not selected_rows.empty:
                selected_site = self._serialize_site(
                    selected_rows.iloc[0], query_plan.metric_key
                )

        site_cards = (
            [
                self._serialize_site(row, query_plan.metric_key)
                for _, row in ranked.head(query_plan.top_n).iterrows()
            ]
            if not ranked.empty
            else []
        )
        answer, used_llm = self._generate_answer(
            message,
            query_plan,
            answer_context,
            site_cards,
            selected_site,
        )

        return {
            "answer": answer,
            "used_llm": used_llm,
            "metric_key": query_plan.metric_key,
            "metric_label": query_plan.metric_label,
            "filters_applied": query_plan.filters_applied,
            "proxy_note": query_plan.proxy_note,
            "sites": site_cards,
            "llm": self.get_llm_status(),
        }

    def _ensure_loaded(self) -> None:
        if not self.geojson_path.exists():
            raise FileNotFoundError(f"Missing map dataset at {self.geojson_path}")

        current_mtime = self.geojson_path.stat().st_mtime
        history_mtime = (
            self.master_table_path.stat().st_mtime
            if self.master_table_path.exists()
            else None
        )
        if (
            self._cached_geojson is not None
            and self._cached_frame is not None
            and self._cached_mtime == current_mtime
            and self._cached_history_frame is not None
            and self._cached_history_mtime == history_mtime
        ):
            return

        geojson = json.loads(self.geojson_path.read_text(encoding="utf-8"))
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

        history_frame = self._load_history_frame(case_cutoff_date)

        self._cached_geojson = geojson
        self._cached_frame = frame
        self._cached_history_frame = history_frame
        self._cached_mtime = current_mtime
        self._cached_history_mtime = history_mtime
        self._case_cutoff_date = case_cutoff_date

    def _load_history_frame(
        self, case_cutoff_date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        if not self.master_table_path.exists():
            return pd.DataFrame()

        history_columns = [
            "sitenumber",
            "sitename",
            "productionarea",
            "county",
            "municipality",
            "week_start_date",
            "femaleadult",
            "mobilelice",
            "persistentlice",
            "seatemperature",
            "femaleadult_to_limit_ratio",
            "breach_this_week",
            "havecountedlice",
            "any_treatment",
            "treatment_count",
            "any_treatment_roll4_sum",
            "neighbor_femaleadult_to_limit_ratio_lag1",
            "pa_breach_rate_lag1",
            "pa_treatment_rate_lag1",
        ]
        try:
            history = pd.read_parquet(self.master_table_path, columns=history_columns)
        except Exception:
            history = pd.read_parquet(self.master_table_path)
            history = history[[column for column in history_columns if column in history.columns]]

        if history.empty:
            return history

        if "week_start_date" in history.columns:
            history["week_start_date"] = pd.to_datetime(
                history["week_start_date"], errors="coerce"
            )

        for column in [
            "femaleadult",
            "mobilelice",
            "persistentlice",
            "seatemperature",
            "femaleadult_to_limit_ratio",
            "treatment_count",
            "any_treatment_roll4_sum",
            "neighbor_femaleadult_to_limit_ratio_lag1",
            "pa_breach_rate_lag1",
            "pa_treatment_rate_lag1",
        ]:
            if column in history.columns:
                history[column] = pd.to_numeric(history[column], errors="coerce")

        for column in ["breach_this_week", "havecountedlice", "any_treatment"]:
            if column in history.columns:
                history[column] = history[column].astype("boolean")

        history["site_id"] = history.get("sitenumber", pd.Series(dtype=object)).astype(
            str
        )
        history["site_label"] = history.get("sitename", pd.Series(dtype=object)).fillna("")
        history["site_label"] = (
            history["site_label"].replace("", np.nan).fillna(history["site_id"])
        )
        if case_cutoff_date is not None and "week_start_date" in history.columns:
            history = history[history["week_start_date"] <= case_cutoff_date].copy()
        return history

    def _build_query_plan(self, message: str, frame: pd.DataFrame) -> QueryPlan:
        query = _normalize_text(message)
        working = frame.copy()
        filters_applied: list[str] = []
        is_current_limit_query = _is_current_limit_query(query)

        for column, label in [
            ("productionarea", "production area"),
            ("county", "county"),
            ("municipality", "municipality"),
            ("sitename", "site"),
        ]:
            matches = self._match_named_values(query, working.get(column))
            if matches:
                working = working[working[column].isin(matches)].copy()
                filters_applied.append(f"{label}: {', '.join(matches)}")

        if (
            "currently_over_limit" in working.columns
            and not is_current_limit_query
            and any(
            token in query for token in ["over limit", "above limit", "breach now"]
            )
        ):
            working = working[working["currently_over_limit"].fillna(False)].copy()
            filters_applied.append("currently over limit")

        if "weeks_since_any_treatment" in working.columns and any(
            token in query
            for token in ["recent treatment", "recently treated", "treated recently"]
        ):
            working = working[
                working["weeks_since_any_treatment"].fillna(np.inf) <= 4
            ].copy()
            filters_applied.append("treated within last 4 weeks")

        if "havecountedlice" in working.columns and any(
            token in query for token in ["counted", "reported lice", "reported count"]
        ):
            working = working[working["havecountedlice"].fillna(False)].copy()
            filters_applied.append("reported lice count available")

        metric_key = "classifier_12w_score"
        metric_label = "12-week breach risk"
        proxy_note = None

        if is_current_limit_query:
            metric_key = "femaleadult_to_limit_ratio"
            metric_label = "current female-adult-to-limit ratio"
        elif _contains_any(
            query,
            [
                "4 weeks",
                "4 week",
                "next month",
                "four weeks",
                "near term",
                "short term",
                "coming weeks",
                "next weeks",
            ],
        ):
            metric_key = "near_term_risk"
            metric_label = "near-term breach risk"
            proxy_note = "Near-term risk uses the maximum of the 1-week and 2-week breach probabilities because the trained classifier horizons are 1w, 2w, and 12w."
        elif _contains_any(query, ["2 weeks", "2 week", "two weeks", "fortnight"]):
            metric_key = "classifier_2w_score"
            metric_label = "2-week breach risk"
        elif _contains_any(query, ["1 week", "1-week", "one week", "next week"]):
            metric_key = "classifier_1w_score"
            metric_label = "1-week breach risk"
        elif _contains_any(
            query,
            ["12 weeks", "12 week", "three months", "12-week", "long term", "season"],
        ):
            metric_key = "classifier_12w_score"
            metric_label = "12-week breach risk"
        elif _contains_any(
            query, ["current pressure", "limit ratio", "adult lice ratio"]
        ):
            metric_key = "femaleadult_to_limit_ratio"
            metric_label = "current female-adult-to-limit ratio"
        elif _contains_any(
            query, ["predicted lice count", "expected lice count", "count forecast"]
        ):
            metric_key = "count_12w_prediction"
            metric_label = "12-week predicted lice count"

        descending = not _contains_any(query, ["lowest", "safest", "least", "smallest"])
        top_n = 5
        match = re.search(r"top\s+(\d+)", query)
        if match:
            top_n = max(1, min(15, int(match.group(1))))
        elif _contains_any(query, ["list", "show me", "which sites", "rank"]):
            top_n = 7

        return QueryPlan(
            frame=working,
            metric_key=metric_key,
            metric_label=metric_label,
            descending=descending,
            filters_applied=filters_applied,
            proxy_note=proxy_note,
            top_n=top_n,
        )

    def _build_answer_context(
        self,
        message: str,
        query_plan: QueryPlan,
    ) -> AnswerContext:
        working = query_plan.frame.copy()
        normalized_query = _normalize_text(message)
        question_type = _detect_question_type(normalized_query)

        if question_type == "current_limit_status":
            if "currently_over_limit" in working.columns:
                exact_matches = working[working["currently_over_limit"].fillna(False)].copy()
            else:
                exact_matches = working.head(0).copy()

            if not exact_matches.empty:
                ranked = self._sort_limit_candidates(exact_matches)
                return AnswerContext(
                    ranked=ranked,
                    question_type="current_limit_status",
                    candidate_mode="exact-matches",
                    summary=f"{len(ranked)} visible site(s) are already above the lice limit.",
                    rationale=[
                        "Exact matches are defined by currently_over_limit = true.",
                        "Matching sites are ranked by the current female-adult-to-limit ratio.",
                    ],
                    exact_match_count=int(len(ranked)),
                )

            ranked = self._sort_limit_candidates(working)
            return AnswerContext(
                ranked=ranked,
                question_type="current_limit_status",
                candidate_mode="near-misses",
                summary="No visible sites are currently above the lice limit.",
                rationale=[
                    "No visible site has currently_over_limit = true in the active filter set.",
                    "Nearest alternatives are ranked by the current female-adult-to-limit ratio.",
                ],
                exact_match_count=0,
            )

        if question_type == "area_pressure":
            return self._build_area_pressure_context(query_plan)

        if question_type == "repeated_breaches":
            return self._build_repeated_breach_context(query_plan)

        if question_type == "treatment_intensity_area":
            return self._build_treatment_intensity_context(query_plan)

        if question_type == "pre_breach_patterns":
            return self._build_pre_breach_pattern_context(query_plan)

        ranked = self._rank_sites(query_plan)
        rationale: list[str] = [f"Candidates are ranked by {query_plan.metric_label}."]
        if query_plan.filters_applied:
            rationale.append(
                f"Active filters: {', '.join(query_plan.filters_applied)}."
            )
        if query_plan.proxy_note:
            rationale.append(query_plan.proxy_note)
        return AnswerContext(
            ranked=ranked,
            question_type="generic",
            candidate_mode="ranked",
            summary=f"Candidates ranked by {query_plan.metric_label}.",
            rationale=rationale,
        )

    def _build_area_pressure_context(self, query_plan: QueryPlan) -> AnswerContext:
        history = self._get_history_subset(query_plan.frame)
        if history.empty or "productionarea" not in history.columns:
            return self._build_generic_context(query_plan)

        history = history.dropna(subset=["week_start_date", "productionarea"]).copy()
        if history.empty:
            return self._build_generic_context(query_plan)

        latest = history["week_start_date"].max()
        current_cutoff = latest - pd.Timedelta(weeks=4)
        previous_cutoff = latest - pd.Timedelta(weeks=8)

        current = history[history["week_start_date"] > current_cutoff].copy()
        previous = history[
            (history["week_start_date"] <= current_cutoff)
            & (history["week_start_date"] > previous_cutoff)
        ].copy()
        if current.empty:
            return self._build_generic_context(query_plan)

        current_group = current.groupby("productionarea").agg(
            current_ratio=("femaleadult_to_limit_ratio", "mean"),
            current_breach_rate=("breach_this_week", "mean"),
            active_sites=("site_id", pd.Series.nunique),
        )
        previous_group = previous.groupby("productionarea").agg(
            previous_ratio=("femaleadult_to_limit_ratio", "mean"),
            previous_breach_rate=("breach_this_week", "mean"),
        )
        summary = current_group.join(previous_group, how="left").fillna(0.0)
        summary["ratio_change"] = summary["current_ratio"] - summary["previous_ratio"]
        summary["breach_change"] = (
            summary["current_breach_rate"] - summary["previous_breach_rate"]
        )
        summary = summary.sort_values(
            ["ratio_change", "breach_change", "current_ratio"],
            ascending=[False, False, False],
        )

        supporting_rows = [
            {
                "label": area,
                "detail": (
                    f"Current ratio {_format_ratio(row.current_ratio)} versus {_format_ratio(row.previous_ratio)} "
                    f"({_format_signed_ratio(row.ratio_change)}); breach rate {_format_percent(row.current_breach_rate)} "
                    f"versus {_format_percent(row.previous_breach_rate)}; {int(row.active_sites)} active site(s)."
                ),
            }
            for area, row in summary.head(5).iterrows()
        ]
        top_areas = list(summary.head(max(3, query_plan.top_n)).index)
        ranked = self._rank_snapshot_subset(
            query_plan.frame[query_plan.frame["productionarea"].isin(top_areas)].copy(),
            metric_key="near_term_risk",
            top_n=query_plan.top_n,
        )
        answer_areas = ", ".join(top_areas[:3]) if top_areas else "the visible production areas"
        return AnswerContext(
            ranked=ranked,
            question_type="area_pressure",
            candidate_mode="area-ranking",
            summary=f"The clearest current increase in lice pressure is in {answer_areas}.",
            rationale=[
                "This compares the last 4 weeks with the previous 4 weeks.",
                "Ranking uses the change in mean female-adult-to-limit ratio, with breach rate change as a tie-breaker.",
            ],
            supporting_title="Production Areas",
            supporting_rows=supporting_rows,
            window_note="Last 4 weeks versus the previous 4 weeks.",
        )

    def _build_repeated_breach_context(self, query_plan: QueryPlan) -> AnswerContext:
        history = self._get_history_subset(query_plan.frame)
        if history.empty:
            return self._build_generic_context(query_plan)

        history = history.dropna(subset=["week_start_date", "site_id"]).copy()
        if history.empty:
            return self._build_generic_context(query_plan)

        latest = history["week_start_date"].max()
        window = history[history["week_start_date"] > latest - pd.Timedelta(weeks=52)].copy()
        if window.empty:
            return self._build_generic_context(query_plan)

        window = window.sort_values(["site_id", "week_start_date"]).copy()
        window["breach_flag"] = window["breach_this_week"].fillna(False).astype(int)
        window["previous_breach_flag"] = (
            window.groupby("site_id")["breach_flag"].shift(1).fillna(0).astype(int)
        )
        window["breach_episode_start"] = (
            (window["breach_flag"] == 1) & (window["previous_breach_flag"] == 0)
        ).astype(int)

        summary = (
            window.groupby(["site_id", "site_label", "productionarea"]).agg(
                breach_weeks=("breach_flag", "sum"),
                breach_episodes=("breach_episode_start", "sum"),
                peak_ratio=("femaleadult_to_limit_ratio", "max"),
            )
        ).reset_index()
        summary = summary[summary["breach_weeks"] > 0].sort_values(
            ["breach_weeks", "breach_episodes", "peak_ratio"],
            ascending=[False, False, False],
        )
        if summary.empty:
            return AnswerContext(
                ranked=query_plan.frame.head(0),
                question_type="repeated_breaches",
                candidate_mode="no-repeats",
                summary="No visible sites show repeated breaches in the last 52 weeks.",
                rationale=[
                    "The historical check looked for breach weeks in the last 52 weeks of the visible population.",
                    "Consecutive breach weeks were collapsed into a single episode when they belonged to the same run.",
                ],
                supporting_title="Repeated-Breach Sites",
                supporting_rows=[],
                window_note="Last 52 weeks.",
            )

        supporting_rows = [
            {
                "label": f"{row.site_label} ({row.productionarea})",
                "detail": (
                    f"{int(row.breach_weeks)} breach week(s) across {int(row.breach_episodes)} separate episode(s) "
                    f"in the last 52 weeks; peak ratio {_format_ratio(row.peak_ratio)}."
                ),
            }
            for _, row in summary.head(5).iterrows()
        ]
        top_site_ids = summary["site_id"].head(query_plan.top_n).tolist()
        ranked = self._ordered_snapshot_subset(query_plan.frame, top_site_ids)
        lead = summary.iloc[0]
        return AnswerContext(
            ranked=ranked,
            question_type="repeated_breaches",
            candidate_mode="site-history",
            summary=(
                f"{lead.site_label} in {lead.productionarea} shows the strongest repeated-breach pattern in the visible population."
            ),
            rationale=[
                "The ranking uses breach weeks in the last 52 weeks.",
                "Consecutive breach weeks count as one episode when they are part of the same run.",
            ],
            supporting_title="Repeated-Breach Sites",
            supporting_rows=supporting_rows,
            window_note="Last 52 weeks.",
        )

    def _build_treatment_intensity_context(self, query_plan: QueryPlan) -> AnswerContext:
        history = self._get_history_subset(query_plan.frame)
        if history.empty or "productionarea" not in history.columns:
            return self._build_generic_context(query_plan)

        history = history.dropna(subset=["week_start_date", "productionarea"]).copy()
        if history.empty:
            return self._build_generic_context(query_plan)

        latest = history["week_start_date"].max()
        window = history[history["week_start_date"] > latest - pd.Timedelta(weeks=12)].copy()
        if window.empty:
            return self._build_generic_context(query_plan)

        summary = window.groupby("productionarea").agg(
            treatment_count=("treatment_count", "sum"),
            active_site_weeks=("site_id", "count"),
            treated_weeks=("any_treatment", "sum"),
        )
        summary["treatment_intensity"] = (
            summary["treatment_count"] / summary["active_site_weeks"].replace(0, np.nan)
        )
        summary["treated_share"] = (
            summary["treated_weeks"] / summary["active_site_weeks"].replace(0, np.nan)
        )
        summary = summary.fillna(0.0).sort_values(
            ["treatment_intensity", "treated_share", "treatment_count"],
            ascending=[False, False, False],
        )

        supporting_rows = [
            {
                "label": area,
                "detail": (
                    f"{_format_decimal(row.treatment_intensity, 3)} treatment event(s) per active site-week in the last 12 weeks; "
                    f"treated-share {_format_percent(row.treated_share)}; {int(round(row.treatment_count))} total treatments."
                ),
            }
            for area, row in summary.head(5).iterrows()
        ]
        top_areas = list(summary.head(max(3, query_plan.top_n)).index)
        snapshot = query_plan.frame[query_plan.frame["productionarea"].isin(top_areas)].copy()
        if "weeks_since_any_treatment" in snapshot.columns:
            recent_snapshot = snapshot[
                snapshot["weeks_since_any_treatment"].fillna(np.inf) <= 4
            ].copy()
            if not recent_snapshot.empty:
                snapshot = recent_snapshot
        ranked = self._rank_snapshot_subset(
            snapshot,
            metric_key="near_term_risk",
            top_n=query_plan.top_n,
        )
        answer_areas = ", ".join(top_areas[:3]) if top_areas else "the visible production areas"
        return AnswerContext(
            ranked=ranked,
            question_type="treatment_intensity_area",
            candidate_mode="area-ranking",
            summary=f"The highest recent treatment intensity is in {answer_areas}.",
            rationale=[
                "Treatment intensity is total treatment events per active site-week.",
                "Treated-share is the share of active site-weeks with any treatment in the same 12-week window.",
            ],
            supporting_title="Production Areas",
            supporting_rows=supporting_rows,
            window_note="Last 12 weeks.",
        )

    def _build_pre_breach_pattern_context(self, query_plan: QueryPlan) -> AnswerContext:
        history = self._get_history_subset(query_plan.frame)
        if history.empty:
            return self._build_generic_context(query_plan)

        history = history.dropna(subset=["week_start_date", "site_id"]).copy()
        history = history.sort_values(["site_id", "week_start_date"]).copy()
        if history.empty:
            return self._build_generic_context(query_plan)

        history["next_breach"] = history.groupby("site_id")["breach_this_week"].shift(-1)
        baseline = history[history["havecountedlice"].fillna(False)].copy()
        pre_breach = baseline[baseline["next_breach"].fillna(False)].copy()
        if pre_breach.empty or baseline.empty:
            return self._build_generic_context(query_plan)

        supporting_rows: list[dict[str, object]] = []
        for label, column, formatter in [
            ("Female-adult-to-limit ratio", "femaleadult_to_limit_ratio", "ratio"),
            (
                "Neighbor limit ratio lag 1",
                "neighbor_femaleadult_to_limit_ratio_lag1",
                "ratio",
            ),
            ("Production-area breach rate lag 1", "pa_breach_rate_lag1", "percent"),
            ("Treatments in prior 4 weeks", "any_treatment_roll4_sum", "count"),
        ]:
            if column not in pre_breach.columns or column not in baseline.columns:
                continue
            pre_mean = pre_breach[column].mean()
            baseline_mean = baseline[column].mean()
            if pd.isna(pre_mean) or pd.isna(baseline_mean):
                continue
            supporting_rows.append(
                {
                    "label": label,
                    "detail": _format_pattern_detail(
                        pre_mean,
                        baseline_mean,
                        formatter,
                    ),
                }
            )

        summary = (
            "Before breaches, sites usually already show a much higher lice-to-limit ratio and stronger surrounding pressure."
            if supporting_rows
            else "The visible history does not show a stable pre-breach pattern summary."
        )
        return AnswerContext(
            ranked=query_plan.frame.head(0),
            question_type="pre_breach_patterns",
            candidate_mode="pattern-summary",
            summary=summary,
            rationale=[
                "Pre-breach rows are site-weeks whose next observed week is a breach.",
                "The baseline uses counted site-weeks in the same visible population so the comparison stays local.",
            ],
            supporting_title="Observed Patterns",
            supporting_rows=supporting_rows,
            window_note="Across the available historical record in the current visible population.",
        )

    def _build_generic_context(self, query_plan: QueryPlan) -> AnswerContext:
        ranked = self._rank_sites(query_plan)
        rationale: list[str] = [f"Candidates are ranked by {query_plan.metric_label}."]
        if query_plan.filters_applied:
            rationale.append(
                f"Active filters: {', '.join(query_plan.filters_applied)}."
            )
        if query_plan.proxy_note:
            rationale.append(query_plan.proxy_note)
        return AnswerContext(
            ranked=ranked,
            question_type="generic",
            candidate_mode="ranked",
            summary=f"Candidates ranked by {query_plan.metric_label}.",
            rationale=rationale,
        )

    def _get_history_subset(self, snapshot_frame: pd.DataFrame) -> pd.DataFrame:
        history = self._cached_history_frame
        if history is None or history.empty or snapshot_frame.empty:
            return pd.DataFrame()

        if "site_id" not in snapshot_frame.columns:
            return pd.DataFrame()

        site_ids = snapshot_frame["site_id"].dropna().astype(str).unique().tolist()
        if not site_ids:
            return pd.DataFrame()
        return history[history["site_id"].isin(site_ids)].copy()

    def _rank_snapshot_subset(
        self,
        frame: pd.DataFrame,
        metric_key: str,
        top_n: int,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame

        working = frame.copy()
        sort_columns: list[str] = []
        ascending: list[bool] = []
        if metric_key in working.columns and working[metric_key].notna().any():
            working = working[working[metric_key].notna()].copy()
            sort_columns.append(metric_key)
            ascending.append(False)

        for column in ["femaleadult_to_limit_ratio", "classifier_12w_score", "near_term_risk"]:
            if column in working.columns and column not in sort_columns:
                sort_columns.append(column)
                ascending.append(False)

        sort_columns.append("site_label")
        ascending.append(True)
        if working.empty:
            return working
        return working.sort_values(sort_columns, ascending=ascending).head(top_n).reset_index(
            drop=True
        )

    def _ordered_snapshot_subset(
        self,
        frame: pd.DataFrame,
        site_ids: list[str],
    ) -> pd.DataFrame:
        if frame.empty or not site_ids:
            return frame.head(0)

        order = {str(site_id): index for index, site_id in enumerate(site_ids)}
        working = frame[frame["site_id"].isin(order)].copy()
        if working.empty:
            return working
        working["__order"] = working["site_id"].map(order)
        return (
            working.sort_values("__order")
            .drop(columns="__order")
            .reset_index(drop=True)
        )

    def _sort_limit_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        sort_columns: list[str] = []
        ascending: list[bool] = []
        for column in ["femaleadult_to_limit_ratio", "femaleadult"]:
            if column in working.columns:
                working = working[working[column].notna()].copy()
                sort_columns.append(column)
                ascending.append(False)
        sort_columns.append("site_label")
        ascending.append(True)
        if working.empty:
            return working
        return working.sort_values(sort_columns, ascending=ascending).reset_index(
            drop=True
        )

    def _rank_sites(self, query_plan: QueryPlan) -> pd.DataFrame:
        working = query_plan.frame.copy()
        metric_key = query_plan.metric_key
        if metric_key not in working.columns:
            return working.head(0)

        working = working[working[metric_key].notna()].copy()
        if working.empty:
            return working

        tie_breaker_columns = [
            column
            for column in [
                "femaleadult_to_limit_ratio",
                "classifier_12w_score",
                "count_12w_prediction",
            ]
            if column in working.columns
        ]
        sort_columns = [metric_key, *tie_breaker_columns, "site_label"]
        ascending = [not query_plan.descending] * (len(sort_columns) - 1) + [True]
        return working.sort_values(sort_columns, ascending=ascending).reset_index(
            drop=True
        )

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
            "latest_observation_date": _jsonify(row.get("latest_observation_date")),
            "last_treatment_date": _jsonify(row.get("last_treatment_date")),
            "last_treatment_action": _jsonify(row.get("last_treatment_action")),
            "last_treatment_activeingredient": _jsonify(
                row.get("last_treatment_activeingredient")
            ),
        }

    def _generate_answer(
        self,
        message: str,
        query_plan: QueryPlan,
        answer_context: AnswerContext,
        site_cards: list[dict[str, object]],
        selected_site: dict[str, object] | None,
    ) -> tuple[str, bool]:
        prompt_payload = {
            "question": message,
            "metric_label": query_plan.metric_label,
            "filters_applied": query_plan.filters_applied,
            "proxy_note": query_plan.proxy_note,
            "answer_context": {
                "question_type": answer_context.question_type,
                "candidate_mode": answer_context.candidate_mode,
                "summary": answer_context.summary,
                "rationale": answer_context.rationale,
                "exact_match_count": answer_context.exact_match_count,
                "supporting_title": answer_context.supporting_title,
                "supporting_rows": answer_context.supporting_rows,
                "window_note": answer_context.window_note,
            },
            "selected_site": selected_site,
            "candidate_sites": site_cards,
        }
        llm_answer = self._call_gemini(prompt_payload)
        if llm_answer:
            return llm_answer, True
        return self._fallback_answer(query_plan, answer_context, site_cards), False

    def _call_gemini(self, prompt_payload: dict[str, object]) -> str | None:
        self._last_llm_error = None
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        model_name = os.getenv("VERTEX_GEMINI_MODEL", DEFAULT_CHAT_MODEL)
        if genai is None:
            self._last_llm_error = "google-genai is not installed"
            return None
        if not project or not location:
            self._last_llm_error = "Missing GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION in the process environment"
            return None

        if self._cached_client is None:
            self._cached_client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )

        prompt = (
            "You are an operational forecasting assistant for a salmon farming company. "
            "Answer only from the provided site data. Think through the exact question before you answer. "
            "Use Markdown. Start with **Answer** and give the direct answer first. Then add **Why** with short bullets. "
            "Prefer the structured answer_context over guessing from candidate_sites. "
            "If answer_context.question_type refers to production areas, answer in terms of production areas rather than sites. "
            "If answer_context.question_type is pre_breach_patterns, describe patterns only and do not invent causal claims. "
            "If answer_context.candidate_mode is near-misses, explicitly say there are no exact matches and then add **Closest sites** explaining that they are nearest alternatives, not exact matches. "
            "If answer_context.candidate_mode is exact-matches, only describe the exact matching sites. "
            "When ranking sites, mention coordinates and the metric used. If the request uses a proxy horizon, explain that briefly. If supporting_rows are present, summarize them directly and keep the wording faithful to the provided numbers.\n\n"
            f"Context:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            response = self._cached_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        except Exception as exc:
            self._last_llm_error = _truncate_error(exc)
            return None

        text = getattr(response, "text", None)
        if not text:
            self._last_llm_error = "Vertex response did not include response.text"
            return None
        return text.strip()

    def _fallback_answer(
        self,
        query_plan: QueryPlan,
        answer_context: AnswerContext,
        site_cards: list[dict[str, object]],
    ) -> str:
        if not site_cards and not answer_context.supporting_rows:
            return "**Answer**\n\nNo candidate sites were available after applying the current filters."

        lines: list[str] = ["**Answer**"]

        if answer_context.question_type == "current_limit_status":
            if answer_context.candidate_mode == "exact-matches":
                lines.append(
                    f"{answer_context.exact_match_count} visible site(s) are already above the lice limit."
                )
                section_title = "**Matching sites**"
            else:
                lines.append("No visible sites are currently above the lice limit.")
                section_title = "**Closest sites**"
        elif answer_context.question_type in {
            "area_pressure",
            "repeated_breaches",
            "treatment_intensity_area",
            "pre_breach_patterns",
        }:
            lines.append(answer_context.summary)
            section_title = (
                f"**{answer_context.supporting_title}**"
                if answer_context.supporting_title
                else "**Supporting detail**"
            )
        else:
            lead = site_cards[0]
            lines.append(
                f"Top match by {query_plan.metric_label}: {lead['sitename']} in {lead['productionarea']} at {lead['metric_display']}."
            )
            section_title = "**Top sites**"

        lines.extend(["", "**Why**"])
        for reason in answer_context.rationale:
            lines.append(f"- {reason}")
        if answer_context.window_note:
            lines.append(f"- Window: {answer_context.window_note}")
        if query_plan.filters_applied and answer_context.question_type != "generic":
            lines.append(f"- Active filters: {', '.join(query_plan.filters_applied)}.")
        if query_plan.proxy_note and query_plan.proxy_note not in answer_context.rationale:
            lines.append(f"- {query_plan.proxy_note}")

        lines.extend(["", section_title])
        if answer_context.supporting_rows:
            for index, row in enumerate(
                answer_context.supporting_rows[: min(5, len(answer_context.supporting_rows))],
                start=1,
            ):
                lines.append(f"{index}. **{row['label']}** - {row['detail']}")
        else:
            for index, site in enumerate(site_cards[: min(5, len(site_cards))], start=1):
                lines.append(
                    f"{index}. **{site['sitename']}** ({site['productionarea']}) - {site['metric_display']} at {site['coordinates_text']}"
                )
        return "\n".join(lines)

    def _match_named_values(self, query: str, series: pd.Series | None) -> list[str]:
        if series is None:
            return []
        values = [
            value
            for value in series.dropna().astype(str).unique().tolist()
            if value.strip()
        ]
        matches: list[str] = []
        for value in values:
            normalized = _normalize_text(value)
            if normalized and normalized in query:
                matches.append(value)
        return matches


def _contains_any(query: str, candidates: list[str]) -> bool:
    return any(candidate in query for candidate in candidates)


def _is_current_limit_query(query: str) -> bool:
    return _contains_any(
        query,
        [
            "over limit",
            "above limit",
            "lice limit",
            "weekly threshold",
            "current limit",
        ],
    )


def _detect_question_type(query: str) -> str:
    if _is_current_limit_query(query):
        return "current_limit_status"
    if _contains_any(
        query,
        [
            "increasing lice pressure",
            "increasing pressure",
            "areas currently show increasing",
            "show increasing lice pressure",
        ],
    ):
        return "area_pressure"
    if _contains_any(
        query,
        [
            "repeated breaches",
            "repeated breach",
            "breached repeatedly",
            "repeat breaches",
        ],
    ):
        return "repeated_breaches"
    if _contains_any(
        query,
        [
            "treatment intensity",
            "highest treatment intensity",
            "most treatment intensity",
            "most treated production areas",
        ],
    ):
        return "treatment_intensity_area"
    if _contains_any(
        query,
        [
            "patterns before breaches",
            "before breaches occur",
            "before breaches",
            "pre breach",
            "pre-breach",
        ],
    ):
        return "pre_breach_patterns"
    return "generic"


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


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


def _format_ratio(value: object) -> str:
    numeric = _jsonify(value)
    if numeric is None:
        return "--"
    return f"{float(numeric):.2f}x"


def _format_signed_ratio(value: object) -> str:
    numeric = _jsonify(value)
    if numeric is None:
        return "--"
    return f"{float(numeric):+.2f}x"


def _format_percent(value: object) -> str:
    numeric = _jsonify(value)
    if numeric is None:
        return "--"
    return f"{float(numeric) * 100:.1f}%"


def _format_decimal(value: object, digits: int = 2) -> str:
    numeric = _jsonify(value)
    if numeric is None:
        return "--"
    return f"{float(numeric):.{digits}f}"


def _format_pattern_detail(
    pre_breach_value: object,
    baseline_value: object,
    formatter: str,
) -> str:
    pre_numeric = _jsonify(pre_breach_value)
    baseline_numeric = _jsonify(baseline_value)
    if pre_numeric is None or baseline_numeric is None:
        return "No stable comparison was available."

    if formatter == "ratio":
        difference = float(pre_numeric) - float(baseline_numeric)
        return (
            f"{_format_ratio(pre_numeric)} before a breach versus {_format_ratio(baseline_numeric)} baseline "
            f"({_format_signed_ratio(difference)})."
        )

    if formatter == "percent":
        difference = float(pre_numeric) - float(baseline_numeric)
        sign = "+" if difference >= 0 else "-"
        return (
            f"{_format_percent(pre_numeric)} before a breach versus {_format_percent(baseline_numeric)} baseline "
            f"({sign}{abs(difference) * 100:.1f} pp)."
        )

    difference = float(pre_numeric) - float(baseline_numeric)
    sign = "+" if difference >= 0 else "-"
    return (
        f"{_format_decimal(pre_numeric, 2)} before a breach versus {_format_decimal(baseline_numeric, 2)} baseline "
        f"({sign}{abs(difference):.2f})."
    )


def _truncate_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message if len(message) <= 400 else f"{message[:397]}..."


def _jsonify(value: object):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
