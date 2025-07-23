"""Entry point for the new fast scheduling pipeline."""

from __future__ import annotations

import argparse
from typing import Any, Optional

from fast.preprocessing import preprocess
from fast.model import build_feasible_model, build_optimized_model
from fast.search import solve_model
from fast.html import generate_html


def run(config_path: str, html_path: str | None = None) -> None:
    """Run preprocessing, solving and report generation."""
    data = preprocess(config_path)
    model = build_feasible_model(data)
    solver = solve_model(model)
    if solver is None:
        raise RuntimeError("No feasible schedule found")
    model = build_optimized_model(model, data)
    solver = solve_model(model)
    if solver is None:
        raise RuntimeError("Optimisation failed")
    html = html_path or "schedule.html"
    generate_html({}, data.config, html)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="School schedule optimiser")
    parser.add_argument("config", nargs="?", default="schedule-config.json")
    parser.add_argument("html", nargs="?", default="schedule.html")
    args = parser.parse_args(argv)
    run(args.config, args.html)


if __name__ == "__main__":
    main()
