"""Analyze one battery folder: layer-2 metrics report plus PNG charts.

Usage:

    python analyze_libero_battery.py --battery-root ../evaluation_runs/battery_x

Writes ``analysis/report.json`` and charts inside the battery folder. Reads
only persisted artifacts; no simulator or GPU required.
"""

import argparse
import json
from pathlib import Path

from embodiedbench.evaluator.libero_battery_analysis import (
    build_report,
    render_charts,
)
from embodiedbench.planner.harness.trace_io import write_json_atomic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <battery-root>/analysis",
    )
    parser.add_argument("--no-charts", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.battery_root / "analysis")
    report = build_report(args.battery_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "report.json", report)
    charts = [] if args.no_charts else render_charts(report, output_dir)
    print(json.dumps(
        {
            "success": report["success"],
            "protocol_health": report["protocol_health"],
            "latency_component_share": report["latency"].get("component_share"),
            "phase_funnel": report["phase_funnel"],
        },
        indent=2,
    ))
    print("REPORT: %s" % (output_dir / "report.json"))
    for chart in charts:
        print("CHART: %s" % chart)


if __name__ == "__main__":
    main()
