from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable

DEFAULT_TEMPERATURES = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
TEMPERATURE_COLORS = {
    4.0: "#1D3557",
    6.0: "#2A6F97",
    8.0: "#2D6A4F",
    10.0: "#40916C",
    12.0: "#74C69D",
    14.0: "#F4A261",
    16.0: "#E76F51",
    18.0: "#B56576",
    20.0: "#7F5539",
}


@dataclass(frozen=True)
class GrowthModelConfig:
    initial_weight_g: float = 50.0
    beta: float = -0.30
    optimum_temperature_c: float = 13.5
    curvature: float = 0.018
    target_sgr_percent: float = 1.53
    reference_weight_g: float = 150.0
    calibration_temperature_c: float = 14.0
    days: int = 400
    temperatures_c: tuple[float, ...] = DEFAULT_TEMPERATURES

    @property
    def alpha(self) -> float:
        return self.target_sgr_percent / (self.reference_weight_g**self.beta)

    def sgr_percent(self, temperature_c: float, weight_g: float) -> float:
        if weight_g <= 0:
            raise ValueError("weight_g must be positive")

        thermal_response = math.exp(
            -self.curvature * ((temperature_c - self.optimum_temperature_c) ** 2)
        )
        return self.alpha * (weight_g**self.beta) * thermal_response


@dataclass(frozen=True)
class DailyObservation:
    day: int
    temperature_c: float
    weight_g: float
    sgr_percent: float
    daily_gain_g: float


def format_temperature(temperature_c: float) -> str:
    if float(temperature_c).is_integer():
        return f"{int(temperature_c)}C"
    return f"{temperature_c:.1f}C"


def parse_temperatures(raw_temperatures: str) -> tuple[float, ...]:
    temperatures = tuple(
        float(token.strip()) for token in raw_temperatures.split(",") if token.strip()
    )
    if not temperatures:
        raise ValueError("At least one temperature must be provided")
    return temperatures


def simulate_curve(
    temperature_c: float, config: GrowthModelConfig
) -> list[DailyObservation]:
    observations: list[DailyObservation] = [
        DailyObservation(
            day=0,
            temperature_c=temperature_c,
            weight_g=config.initial_weight_g,
            sgr_percent=config.sgr_percent(temperature_c, config.initial_weight_g),
            daily_gain_g=0.0,
        )
    ]
    previous_weight = config.initial_weight_g

    for day in range(1, config.days + 1):
        sgr_percent = config.sgr_percent(temperature_c, previous_weight)
        current_weight = previous_weight * math.exp(sgr_percent / 100.0)
        observations.append(
            DailyObservation(
                day=day,
                temperature_c=temperature_c,
                weight_g=current_weight,
                sgr_percent=sgr_percent,
                daily_gain_g=current_weight - previous_weight,
            )
        )
        previous_weight = current_weight

    return observations


def run_simulation(config: GrowthModelConfig) -> dict[float, list[DailyObservation]]:
    return {
        temperature_c: simulate_curve(temperature_c, config)
        for temperature_c in config.temperatures_c
    }


def find_day_to_weight(
    curve: Iterable[DailyObservation], target_weight_g: float
) -> int | None:
    for observation in curve:
        if observation.weight_g >= target_weight_g:
            return observation.day
    return None


def build_curve_summary(
    temperature_c: float,
    curve: list[DailyObservation],
) -> dict[str, float | int | None | str]:
    return {
        "temperature_c": temperature_c,
        "temperature_label": format_temperature(temperature_c),
        "final_weight_g": round(curve[-1].weight_g, 3),
        "mean_sgr_percent": round(
            fmean(observation.sgr_percent for observation in curve[1:]), 4
        ),
        "max_daily_gain_g": round(
            max(observation.daily_gain_g for observation in curve), 4
        ),
        "day_to_500g": find_day_to_weight(curve, 500.0),
        "day_to_1000g": find_day_to_weight(curve, 1000.0),
    }


def build_validation(
    config: GrowthModelConfig, scenarios: dict[float, list[DailyObservation]]
) -> list[dict[str, object]]:
    calibration_sgr = config.sgr_percent(
        config.calibration_temperature_c, config.reference_weight_g
    )
    target_without_thermal = config.alpha * (config.reference_weight_g**config.beta)
    validation = [
        {
            "name": "alpha_matches_stata_formula",
            "passed": math.isclose(
                target_without_thermal,
                config.target_sgr_percent,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "expected": config.target_sgr_percent,
            "observed": target_without_thermal,
        },
        {
            "name": "effective_14c_sgr_is_close_to_comment_target",
            "passed": abs(calibration_sgr - config.target_sgr_percent) < 0.01,
            "expected": config.target_sgr_percent,
            "observed": calibration_sgr,
            "delta": calibration_sgr - config.target_sgr_percent,
        },
        {
            "name": "all_weights_positive",
            "passed": all(
                observation.weight_g > 0
                for curve in scenarios.values()
                for observation in curve
            ),
        },
    ]

    if 10.0 in scenarios and 14.0 in scenarios and 18.0 in scenarios:
        final_14 = scenarios[14.0][-1].weight_g
        validation.append(
            {
                "name": "14c_outperforms_10c_and_18c",
                "passed": final_14 > scenarios[10.0][-1].weight_g
                and final_14 > scenarios[18.0][-1].weight_g,
                "observed": {
                    "10C": round(scenarios[10.0][-1].weight_g, 3),
                    "14C": round(final_14, 3),
                    "18C": round(scenarios[18.0][-1].weight_g, 3),
                },
            }
        )

    return validation


def build_summary(
    config: GrowthModelConfig, scenarios: dict[float, list[DailyObservation]]
) -> dict[str, object]:
    curve_summaries = [
        build_curve_summary(temperature_c, curve)
        for temperature_c, curve in scenarios.items()
    ]
    ranked = sorted(
        curve_summaries, key=lambda item: item["final_weight_g"], reverse=True
    )

    return {
        "config": asdict(config),
        "calibrated_alpha": round(config.alpha, 8),
        "curve_summary": curve_summaries,
        "ranked_by_final_weight": ranked,
        "best_temperature_by_final_weight": ranked[0],
        "validation": build_validation(config, scenarios),
    }


def write_wide_csv(
    output_dir: Path, scenarios: dict[float, list[DailyObservation]]
) -> Path:
    output_path = output_dir / "weights_wide.csv"
    ordered_temperatures = tuple(scenarios.keys())
    max_day = len(next(iter(scenarios.values())))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "day",
                *[
                    f"w_temp_{format_temperature(temperature_c)}"
                    for temperature_c in ordered_temperatures
                ],
            ]
        )

        for index in range(max_day):
            row = [index]
            for temperature_c in ordered_temperatures:
                row.append(round(scenarios[temperature_c][index].weight_g, 6))
            writer.writerow(row)

    return output_path


def write_long_csv(
    output_dir: Path, scenarios: dict[float, list[DailyObservation]]
) -> Path:
    output_path = output_dir / "weights_long.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "day",
                "temperature_c",
                "temperature_label",
                "weight_g",
                "sgr_percent",
                "daily_gain_g",
            ],
        )
        writer.writeheader()

        for temperature_c, curve in scenarios.items():
            for observation in curve:
                writer.writerow(
                    {
                        "day": observation.day,
                        "temperature_c": temperature_c,
                        "temperature_label": format_temperature(temperature_c),
                        "weight_g": round(observation.weight_g, 6),
                        "sgr_percent": round(observation.sgr_percent, 6),
                        "daily_gain_g": round(observation.daily_gain_g, 6),
                    }
                )

    return output_path


def write_summary_json(output_dir: Path, summary: dict[str, object]) -> Path:
    output_path = output_dir / "growth_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return output_path


def render_svg(
    config: GrowthModelConfig, scenarios: dict[float, list[DailyObservation]]
) -> str:
    width = 1200
    height = 720
    margin_left = 90
    margin_right = 40
    margin_top = 100
    margin_bottom = 80

    max_day = max(
        observation.day for curve in scenarios.values() for observation in curve
    )
    max_weight = max(
        observation.weight_g for curve in scenarios.values() for observation in curve
    )
    padded_max_weight = max_weight * 1.08

    def x_position(day: int) -> float:
        usable_width = width - margin_left - margin_right
        return margin_left + (day / max_day) * usable_width

    def y_position(weight_g: float) -> float:
        usable_height = height - margin_top - margin_bottom
        return height - margin_bottom - (weight_g / padded_max_weight) * usable_height

    grid_lines: list[str] = []
    for index in range(6):
        day = round((max_day / 5) * index)
        x_value = x_position(day)
        grid_lines.append(
            f'<line x1="{x_value:.2f}" y1="{margin_top}" x2="{x_value:.2f}" y2="{height - margin_bottom}" '
            'stroke="#D9E2EC" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{x_value:.2f}" y="{height - margin_bottom + 28}" text-anchor="middle" '
            'font-size="14" fill="#334E68">'
            f"{day}</text>"
        )

    for index in range(6):
        weight = (padded_max_weight / 5) * index
        y_value = y_position(weight)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y_value:.2f}" x2="{width - margin_right}" y2="{y_value:.2f}" '
            'stroke="#D9E2EC" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 16}" y="{y_value + 4:.2f}" text-anchor="end" '
            'font-size="14" fill="#334E68">'
            f"{weight:.0f}</text>"
        )

    polylines: list[str] = []
    legend_entries: list[str] = []
    for index, (temperature_c, curve) in enumerate(scenarios.items()):
        points = " ".join(
            f"{x_position(obs.day):.2f},{y_position(obs.weight_g):.2f}" for obs in curve
        )
        stroke = TEMPERATURE_COLORS.get(temperature_c, "#102A43")
        stroke_width = (
            4.5
            if math.isclose(temperature_c, config.calibration_temperature_c)
            else 2.5
        )
        polylines.append(
            f'<polyline fill="none" stroke="{stroke}" stroke-width="{stroke_width}" points="{points}" />'
        )

        legend_x = margin_left + 18 + (index % 3) * 160
        legend_y = 52 + (index // 3) * 24
        legend_entries.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" />'
        )
        legend_entries.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-size="14" fill="#102A43">'
            f"{format_temperature(temperature_c)}</text>"
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#F8FBFF" />',
            '<text x="90" y="34" font-size="28" font-weight="700" fill="#102A43">Atlantic salmon growth by temperature</text>',
            '<text x="90" y="78" font-size="16" fill="#486581">',
            "Python port of the Stata model using a calibrated bell-shaped SGR response and daily Euler integration.</text>",
            *legend_entries,
            *grid_lines,
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" '
            'stroke="#243B53" stroke-width="2" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" '
            'stroke="#243B53" stroke-width="2" />',
            *polylines,
            f'<text x="{(margin_left + width - margin_right) / 2:.2f}" y="{height - 22}" text-anchor="middle" '
            'font-size="16" fill="#102A43">Day post sea-transfer</text>',
            f'<text x="26" y="{(margin_top + height - margin_bottom) / 2:.2f}" text-anchor="middle" '
            'font-size="16" fill="#102A43" transform="rotate(-90 26 '
            f'{(margin_top + height - margin_bottom) / 2:.2f})">Weight (g)</text>',
            "</svg>",
        ]
    )


def build_plotly_figure(
    config: GrowthModelConfig, scenarios: dict[float, list[DailyObservation]]
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "plotly is required for interactive HTML output"
        ) from exc

    figure = go.Figure()
    for temperature_c, curve in scenarios.items():
        stroke = TEMPERATURE_COLORS.get(temperature_c, "#102A43")
        line_width = 4.5 if math.isclose(temperature_c, config.calibration_temperature_c) else 2.5
        figure.add_trace(
            go.Scatter(
                x=[observation.day for observation in curve],
                y=[observation.weight_g for observation in curve],
                mode="lines",
                name=format_temperature(temperature_c),
                line={"color": stroke, "width": line_width},
                customdata=[
                    [observation.sgr_percent, observation.daily_gain_g]
                    for observation in curve
                ],
                hovertemplate=(
                    "Temperature=%{fullData.name}<br>"
                    "Day=%{x}<br>"
                    "Weight=%{y:.2f} g<br>"
                    "SGR=%{customdata[0]:.3f}%/day<br>"
                    "Daily gain=%{customdata[1]:.2f} g<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title="Atlantic salmon growth by temperature",
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Day post sea-transfer",
        yaxis_title="Weight (g)",
        legend_title="Temperature",
        title_x=0.02,
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
    )
    figure.add_annotation(
        x=0.02,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            "Bell-shaped SGR response with daily Euler integration. "
            "The 14C line is emphasized because it is the calibration scenario."
        ),
    )
    return figure


def write_svg(
    output_dir: Path,
    config: GrowthModelConfig,
    scenarios: dict[float, list[DailyObservation]],
) -> Path:
    output_path = output_dir / "growth_curves.svg"
    output_path.write_text(render_svg(config, scenarios), encoding="utf-8")
    return output_path


def write_interactive_html(
    output_dir: Path,
    config: GrowthModelConfig,
    scenarios: dict[float, list[DailyObservation]],
) -> Path:
    output_path = output_dir / "growth_curves.html"
    figure = build_plotly_figure(config, scenarios)
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def run_and_export(
    config: GrowthModelConfig,
    output_dir: Path,
    write_svg_chart: bool = True,
    write_html_chart: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = run_simulation(config)
    summary = build_summary(config, scenarios)

    artifacts = {
        "wide_csv": write_wide_csv(output_dir, scenarios),
        "long_csv": write_long_csv(output_dir, scenarios),
        "summary_json": write_summary_json(output_dir, summary),
    }
    if write_html_chart:
        artifacts["html_chart"] = write_interactive_html(output_dir, config, scenarios)
    if write_svg_chart:
        artifacts["svg_chart"] = write_svg(output_dir, config, scenarios)

    return {
        "config": config,
        "summary": summary,
        "artifacts": artifacts,
        "scenarios": scenarios,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Port the Stata Atlantic salmon growth model to Python, with reusable outputs for CSV, JSON, and SVG."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("stata") / "output")
    parser.add_argument(
        "--temperatures-c",
        default=",".join(str(int(value)) for value in DEFAULT_TEMPERATURES),
    )
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--initial-weight-g", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=-0.30)
    parser.add_argument("--optimum-temperature-c", type=float, default=13.5)
    parser.add_argument("--curvature", type=float, default=0.018)
    parser.add_argument("--target-sgr-percent", type=float, default=1.53)
    parser.add_argument("--reference-weight-g", type=float, default=150.0)
    parser.add_argument("--calibration-temperature-c", type=float, default=14.0)
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--skip-svg", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GrowthModelConfig(
        initial_weight_g=args.initial_weight_g,
        beta=args.beta,
        optimum_temperature_c=args.optimum_temperature_c,
        curvature=args.curvature,
        target_sgr_percent=args.target_sgr_percent,
        reference_weight_g=args.reference_weight_g,
        calibration_temperature_c=args.calibration_temperature_c,
        days=args.days,
        temperatures_c=parse_temperatures(args.temperatures_c),
    )
    result = run_and_export(
        config,
        args.output_dir,
        write_svg_chart=not args.skip_svg,
        write_html_chart=not args.skip_html,
    )
    summary = result["summary"]
    best_curve = summary["best_temperature_by_final_weight"]
    failed_validations = [item for item in summary["validation"] if not item["passed"]]

    print(
        f"Best final weight: {best_curve['temperature_label']} -> {best_curve['final_weight_g']:.3f} g"
    )
    print(f"Calibrated alpha: {summary['calibrated_alpha']:.8f}")
    for name, artifact_path in result["artifacts"].items():
        print(f"{name}: {artifact_path}")

    if failed_validations:
        print("Validation failed:")
        for item in failed_validations:
            print(json.dumps(item, ensure_ascii=True))
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
