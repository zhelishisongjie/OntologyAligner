from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ablation import config, experiments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen OntologyAligner ablation protocol v1"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--experiment", choices=("E1", "E2", "E3", "E4", "E5", "E6"))
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=("all", *config.BACKBONE_BY_KEY),
        default="all",
        help="E5 backbone selector; all assembles the formal E5 workbook",
    )
    parser.add_argument(
        "--llm-backbone",
        choices=tuple(config.E4_LLM_CONFIG_KEYS),
        default="claude-opus-5",
        help="E4 reranker model; ignored by other experiments",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.report_only:
        path = experiments.generate_summary()
        print(f"Generated {path}", flush=True)
        return
    if args.all:
        for path in experiments.run_all():
            print(f"Completed {path}", flush=True)
        return
    path = experiments.run_experiment(
        args.experiment, args.backbone, args.llm_backbone
    )
    print(f"Completed {path}", flush=True)


if __name__ == "__main__":
    main()
