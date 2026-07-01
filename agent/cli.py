"""laplace CLI (spec §9): `laplace ask "<question>" --scenario <id>`"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .runner import ClaudeAgentRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer an operational question by experiment")
    ask.add_argument("question")
    ask.add_argument("--scenario", required=True, help="scenario_id (dev scenarios only for development)")
    ask.add_argument("--rollout-budget", type=int, default=200)
    ask.add_argument("--render-budget", type=int, default=4)
    ask.add_argument("--tool-call-budget", type=int, default=25)
    ask.add_argument("--seed-base", type=int, default=0)
    ask.add_argument("--runs-dir", default="runs")
    ask.add_argument("--model", default=None, help="optional model override")
    ask.add_argument("--max-workers", type=int, default=4)

    args = parser.parse_args(argv)

    runner = ClaudeAgentRunner(runs_dir=args.runs_dir, model=args.model,
                               max_workers=args.max_workers)
    result = asyncio.run(runner.run(
        args.question, args.scenario,
        budgets={"rollouts": args.rollout_budget,
                 "renders": args.render_budget,
                 "tool_calls": args.tool_call_budget},
        seed_base=args.seed_base))

    print(json.dumps({
        "accepted": result.accepted,
        "recommendation": (result.report or {}).get("recommendation"),
        "confidence": (result.report or {}).get("confidence"),
        "violations": result.violations,
        "num_turns": result.num_turns,
        "duration_s": round(result.duration_s, 1),
        "cost_usd": result.cost_usd,
        "trace": result.trace_path,
        "episode_dir": result.episode_dir,
        "error": result.error,
    }, indent=2))

    if result.report:
        print("\n--- report ---")
        print(json.dumps(result.report, indent=2))
    return 0 if (result.accepted and not result.error) else 1


if __name__ == "__main__":
    sys.exit(main())
