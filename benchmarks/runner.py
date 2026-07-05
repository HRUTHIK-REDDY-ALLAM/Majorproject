"""Benchmark runner: executes Detective AI over the synthetic scenario suite
and reports quantitative accuracy, calibration, and critic-effectiveness metrics.

Modes:
    offline — deterministic rule-based pipeline (no API calls; used in CI)
    llm     — full LangGraph multi-agent pipeline via Groq
    auto    — llm if GROQ_API_KEY is set, otherwise offline

Usage:
    python -m benchmarks.runner --mode offline --count 10
    python -m benchmarks.runner --mode llm --count 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.generator import generate_all_scenarios

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ── Evidence formatting (ground truth is NEVER included) ─────────────────────


def build_evidence_summary(scenario: dict[str, Any], max_items: int = 60) -> str:
    """Format scenario evidence as an LLM-digestible summary."""
    ev = scenario["evidence"]
    lines: list[str] = ["## Camera Sightings (person IDs from ReID clustering)"]
    for s in ev["camera_sightings"][:max_items]:
        lines.append(
            f"- [{s['timestamp']}] {s['camera_id']}: person {s['person_id']} "
            f"(detection confidence {s['detection_confidence']}) — {s['description']}"
        )
    lines.append("\n## Access Control Logs")
    for log in ev["access_logs"][:max_items]:
        lines.append(
            f"- [{log['timestamp']}] {log['person_name']} ({log['person_id']}) "
            f"{log['action']} at {log['location']}"
        )
    lines.append("\n## Witness Statements")
    for st in ev["witness_statements"]:
        lines.append(
            f"- {st['source']} (assessed reliability {st['reliability_score']}): "
            f"\"{st['text']}\""
        )
    lines.append("\n## Camera Topology")
    for conn in scenario["topology"]["connections"]:
        blind = " [BLIND SPOT]" if conn.get("via_blind_spot") else ""
        lines.append(
            f"- {conn['from']} → {conn['to']}: ~{conn['travel_time']}s{blind}"
        )
    return "\n".join(lines)


# ── Pipeline execution ────────────────────────────────────────────────────────


def run_offline(scenario: dict[str, Any]) -> dict[str, Any]:
    from detective_ai.pipeline.offline import OfflineInvestigationPipeline

    pipeline = OfflineInvestigationPipeline(scenario["topology"])
    return pipeline.investigate(
        case_id=scenario["id"],
        case_title=scenario["title"],
        case_description=scenario["description"],
        camera_sightings=scenario["evidence"]["camera_sightings"],
        access_logs=scenario["evidence"]["access_logs"],
        witness_statements=scenario["evidence"]["witness_statements"],
    )


def run_llm(scenario: dict[str, Any], max_rounds: int = 3) -> dict[str, Any]:
    from detective_ai.agents.graph import run_investigation

    summary = build_evidence_summary(scenario)
    evidence_count = (
        len(scenario["evidence"]["camera_sightings"])
        + len(scenario["evidence"]["access_logs"])
        + len(scenario["evidence"]["witness_statements"])
    )
    return asyncio.run(
        run_investigation(
            case_id=scenario["id"],
            case_title=scenario["title"],
            case_description=scenario["description"],
            evidence_summary=summary,
            evidence_count=evidence_count,
            max_rounds=max_rounds,
        )
    )


# ── Optional MLflow tracking ──────────────────────────────────────────────────


def _log_to_mlflow(mode: str, aggregate: dict[str, Any], results_path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed — skipping experiment tracking "
                    "(pip install 'detective-ai[tracking]').")
        return
    mlflow.set_experiment("detective-ai-benchmark")
    with mlflow.start_run(run_name=f"{mode}-{datetime.now():%Y%m%d-%H%M%S}"):
        mlflow.log_param("mode", mode)
        mlflow.log_param("total_scenarios", aggregate["total_scenarios"])
        mlflow.log_metric("suspect_accuracy", aggregate["suspect_identification"]["accuracy"])
        mlflow.log_metric("timeline_accuracy_mean", aggregate["timeline_accuracy"]["mean"])
        mlflow.log_metric("ece", aggregate["confidence_calibration"]["expected_calibration_error"])
        mlflow.log_metric("critic_effectiveness", aggregate["critic_effectiveness"]["mean"])
        mlflow.log_metric("mean_resolution_seconds", aggregate["resolution_time"]["mean_seconds"])
        mlflow.log_artifact(str(results_path))
    logger.info("Benchmark metrics logged to MLflow.")


# ── Main ──────────────────────────────────────────────────────────────────────


def run_benchmark(
    mode: str = "offline",
    count: int = 10,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the benchmark suite and return aggregate metrics."""
    scenarios = generate_all_scenarios(count=count)
    evaluator = BenchmarkEvaluator()

    print(f"\n{'=' * 70}")
    print(f"Detective AI Benchmark :: mode={mode}, {len(scenarios)} scenarios")
    print(f"{'=' * 70}")

    for scenario in scenarios:
        start = time.perf_counter()
        try:
            if mode == "llm":
                report = run_llm(scenario)
            else:
                report = run_offline(scenario)
        except Exception as e:
            logger.error(f"{scenario['id']} failed: {e}")
            report = {"error": str(e)}
        elapsed = time.perf_counter() - start

        result = evaluator.evaluate_scenario(scenario, report, elapsed)
        status = "[OK]  " if result["suspect_correct"] else "[MISS]"
        print(
            f"  {status} {result['scenario_id']}: "
            f"predicted={result['suspect_predicted']} "
            f"actual={result['suspect_actual']} "
            f"timeline={result['timeline_accuracy']:.2f} "
            f"conf={result['reported_confidence']:.2f} "
            f"critic={result['critic_effectiveness']:.2f} "
            f"({elapsed:.1f}s)"
        )

    aggregate = evaluator.compute_aggregate_metrics()

    # Post-hoc calibration fit (Platt scaling) from this run's outcomes
    calibration_fit = evaluator.confidence_engine.auto_calibrate() or {}

    print(f"\n{'-' * 70}")
    print("Aggregate metrics:")
    print(f"  Suspect identification accuracy : "
          f"{aggregate['suspect_identification']['accuracy']:.1%}")
    print(f"  Timeline reconstruction (IoU)   : "
          f"{aggregate['timeline_accuracy']['mean']:.3f} "
          f"± {aggregate['timeline_accuracy']['std']:.3f}")
    print(f"  Expected Calibration Error      : "
          f"{aggregate['confidence_calibration']['expected_calibration_error']:.4f}")
    print(f"  Critic effectiveness            : "
          f"{aggregate['critic_effectiveness']['mean']:.1%}")
    print(f"  Mean time-to-resolution         : "
          f"{aggregate['resolution_time']['mean_seconds']:.1f}s")
    print(f"{'-' * 70}\n")

    output_path = output_path or (
        RESULTS_DIR / f"benchmark_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    evaluator.save_results(output_path)
    if calibration_fit:
        with open(output_path) as f:
            data = json.load(f)
        data["calibration_fit"] = calibration_fit
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    _log_to_mlflow(mode, aggregate, output_path)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Detective AI benchmark suite")
    parser.add_argument("--mode", choices=["offline", "llm", "auto"], default="offline")
    parser.add_argument("--count", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--output", type=str, default=None, help="Results JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    mode = args.mode
    if mode == "auto":
        from detective_ai.config import settings

        mode = "llm" if settings.groq_api_key else "offline"

    run_benchmark(
        mode=mode,
        count=args.count,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
