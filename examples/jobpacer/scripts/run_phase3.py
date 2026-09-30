"""Unified user entry point for the Phase 3 GPU experiment workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="generate and freeze pilot/formal candidate inputs")
    prepare_modes = prepare.add_subparsers(dest="prepare_action", required=True)
    generate = prepare_modes.add_parser("generate", help="generate workload templates and candidate samples")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--scope", choices=("formal", "pilot"), default="formal")
    generate.add_argument("--workload-seeds")
    freeze = prepare_modes.add_parser("freeze", help="bind calibrated profiles and static orders to candidates")
    freeze.add_argument("--input-dir", type=Path, required=True)
    freeze.add_argument("--compute-profile", type=Path, required=True)
    freeze.add_argument("--comm-profile", type=Path, required=True)
    freeze.add_argument("--calibration-input-dir", type=Path)
    freeze.add_argument("--world-size", type=int, default=2)
    order = prepare_modes.add_parser("order", help="write a deterministic block/arm order table")
    order.add_argument("--suite-manifest", type=Path, required=True)
    order.add_argument("--output", type=Path, required=True)
    order.add_argument("--order-seed", type=int)
    order.add_argument("--repeats", type=int, default=5)
    order.add_argument("--arms")
    preview = prepare_modes.add_parser("preview", help="estimate batch wall time and disk use from history")
    preview.add_argument("--order", type=Path, required=True)
    preview.add_argument("--history-summary", type=Path, action="append", default=[])

    qualify = commands.add_parser("qualify", help="run bare/backend normal and fault-path qualification")
    qualify.add_argument("--mechanism-evidence", type=Path, required=True)
    qualify.add_argument("--suite-manifest", type=Path, required=True)
    qualify.add_argument("--output-dir", type=Path, required=True)
    qualify.add_argument("--timeout", type=float, default=60.0)
    qualify.add_argument("--setup-timeout", type=float, default=30.0)
    qualify.add_argument("--warmup", type=int, default=5)

    check = commands.add_parser("check", help="read readiness evidence or explicitly run a diagnostic")
    check_mode = check.add_mutually_exclusive_group(required=True)
    check_mode.add_argument("--status", action="store_true", help="read existing evidence only; starts no replay")
    check_mode.add_argument("--kind", choices=("measurement", "recovery", "mechanism"),
                            help="explicitly select a diagnostic; some actions run GPU replays")
    check.add_argument("--suite-manifest", type=Path)
    check.add_argument("--qualification", type=Path)
    check.add_argument("--audit", action="append", default=[], metavar="GATE=PATH")
    check.add_argument("--action", choices=("plan", "run", "analyze"),
                       help="measurement action; run launches replay, plan/analyze do not")
    check.add_argument("--output", type=Path)
    check.add_argument("--pairs", type=int, default=5)
    check.add_argument("--timeout", type=float, default=60.0)
    check.add_argument("--message-bytes", type=int, default=1 << 20)
    check.add_argument("--repeats", type=int, default=5)
    check.add_argument("--warmup", type=int, default=5)

    run = commands.add_parser("run", help="plan, execute, or resume a paired batch")
    run.add_argument("--plan-only", action="store_true", help="create a frozen batch plan without replay")
    run.add_argument("--suite-manifest", type=Path,
                     help="required with --plan-only; input suite to freeze into the batch")
    run.add_argument("--batch-dir", "--output-dir", dest="batch_dir", type=Path, required=True)
    run.add_argument("--order-seed", type=int)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--arms")
    run.add_argument("--max-blocks", type=int,
                     help="pilot-only cap for a smoke or seven-arm rehearsal")
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--warmup-iterations", type=int, default=5)
    run.add_argument("--allow-pilot", "--allow-six-arm-pilot", dest="allow_pilot", action="store_true")
    run.add_argument("--allow-formal-matrix", action="store_true",
                     help="explicitly unlock the formal matrix after ready-for-formal approval")
    run.add_argument("--resume", action="store_true",
                     help="resume the exact frozen batch; incomplete blocks restart as whole blocks")

    analyze = commands.add_parser("analyze", help="validate and summarize an existing batch; starts no replay")
    analyze.add_argument("--batch-dir", type=Path, required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument("--analysis-seed", type=int, default=20260928)

    finalize = commands.add_parser("finalize", help="validate pilot/readiness evidence and seal formal inputs")
    finalize_modes = finalize.add_subparsers(dest="finalize_action", required=True)
    pilot = finalize_modes.add_parser("pilot", help="attach an accepted pilot to formal candidate inputs")
    pilot.add_argument("--suite-dir", type=Path, required=True)
    pilot.add_argument("--pilot-batch-dir", type=Path, required=True)
    formal = finalize_modes.add_parser("formal", help="attach all readiness audits and seal formal inputs")
    formal.add_argument("--suite-manifest", type=Path, required=True)
    formal.add_argument("--qualification", type=Path, required=True)
    formal.add_argument("--audit", action="append", required=True, metavar="GATE=PATH")
    return parser


def _audit_paths(items: Sequence[str], parser: argparse.ArgumentParser, *, expected: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            parser.error(f"audit must use GATE=PATH: {item!r}")
        gate, path = item.split("=", 1)
        if gate not in expected:
            parser.error(f"unknown gate {gate!r}; expected one of {sorted(expected)}")
        if gate in result:
            parser.error(f"duplicate gate: {gate}")
        result[gate] = Path(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        from examples.jobpacer.experiments.phase3 import prepare
        try:
            if args.prepare_action == "generate":
                result = prepare.prepare_candidates(args.output_dir, scope=args.scope,
                                                    workload_seeds=args.workload_seeds)
            elif args.prepare_action == "freeze":
                result = prepare.freeze_candidates(
                    args.input_dir, args.compute_profile, args.comm_profile,
                    calibration_input_dir=args.calibration_input_dir, world_size=args.world_size)
            elif args.prepare_action == "order":
                from examples.jobpacer.experiments.phase3.suite import ARMS, ORDER_SEED
                result = prepare.write_order(
                    args.suite_manifest, args.output,
                    order_seed=ORDER_SEED if args.order_seed is None else args.order_seed,
                    repeats=args.repeats, arms=args.arms or ",".join(ARMS))
            else:
                result = prepare.preview_order(args.order, args.history_summary)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "qualify":
        if args.timeout <= 0 or args.setup_timeout <= 0 or args.warmup < 5:
            parser.error("timeouts must be positive and G1 warmup must be at least five iterations")
        from examples.jobpacer.experiments.phase3.qualify import qualify_g1
        audit = qualify_g1(args.suite_manifest, args.output_dir, timeout_s=args.timeout,
                           setup_timeout_s=args.setup_timeout, warmup=args.warmup,
                           mechanism_evidence=args.mechanism_evidence)
        print(json.dumps(audit, sort_keys=True))
        return 0 if audit["passed"] else 1

    if args.command == "check":
        from examples.jobpacer.experiments.phase3 import checks
        if args.status:
            if args.suite_manifest is None:
                parser.error("check --status requires --suite-manifest")
            from examples.jobpacer.experiments.phase3.gates import READINESS_GATES
            audits = _audit_paths(args.audit, parser, expected=set(READINESS_GATES))
            result = checks.check_status(args.suite_manifest, audit_paths=audits,
                                         qualification_path=args.qualification)
        elif args.kind == "measurement":
            if args.action is None or args.output is None:
                parser.error("check --kind measurement requires --action and --output")
            if args.action == "plan" and args.suite_manifest is None:
                parser.error("check --kind measurement --action plan requires --suite-manifest")
            result = checks.run_measurement_check(
                args.action, output=args.output, suite_manifest=args.suite_manifest,
                pairs=args.pairs, timeout=args.timeout)
        elif args.kind == "recovery":
            if args.action not in (None, "run") or args.suite_manifest is None or args.output is None:
                parser.error("check --kind recovery requires --suite-manifest and --output; it runs the rehearsal")
            result = checks.run_recovery_check(args.suite_manifest, args.output)
            result = {"passed": result["passed"], "checks": result["checks"]}
        else:
            if args.action not in (None, "run") or args.output is None:
                parser.error("check --kind mechanism requires --output; it runs the NCCL mechanism diagnostic")
            result = checks.run_mechanism_check(
                args.output, message_bytes=args.message_bytes, repeats=args.repeats,
                warmup=args.warmup, timeout=args.timeout)
        print(json.dumps(result, sort_keys=True))
        return 1 if result.get("passed") is False else 0

    if args.command == "run":
        if args.timeout <= 0 or args.warmup_iterations < 0:
            parser.error("timeout must be positive and warmup non-negative")
        from examples.jobpacer.experiments.phase3 import batch
        if args.plan_only:
            if args.suite_manifest is None or args.resume:
                parser.error("run --plan-only requires --suite-manifest and cannot be combined with --resume")
            from examples.jobpacer.experiments.phase3.suite import ARMS, ORDER_SEED
            arms = tuple(part.strip() for part in (args.arms or ",".join(ARMS)).split(",") if part.strip())
            try:
                manifest = batch.create_batch(
                    args.suite_manifest, args.batch_dir,
                    order_seed=ORDER_SEED if args.order_seed is None else args.order_seed,
                    repeats=args.repeats, arms=arms, timeout_s=args.timeout,
                    max_blocks=args.max_blocks)
            except (OSError, ValueError, KeyError) as exc:
                parser.error(str(exc))
            print(json.dumps({"batch_dir": str(args.batch_dir.resolve()), "stage": manifest["stage"],
                              "blocks": manifest["planned_blocks"], "runs": manifest["planned_runs"],
                              "arms": list(arms)}, sort_keys=True))
            return 0
        if args.suite_manifest is not None or args.max_blocks is not None:
            parser.error("--suite-manifest and --max-blocks are only valid with --plan-only")
        try:
            return batch.run_batch(args.batch_dir, timeout=args.timeout,
                                   warmup=args.warmup_iterations, allow_pilot=args.allow_pilot,
                                   resume=args.resume,
                                   allow_formal_matrix=args.allow_formal_matrix)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"Phase 3 run stopped: {exc}", file=sys.stderr)
            return 2

    if args.command == "analyze":
        if args.bootstrap_samples < 0:
            parser.error("bootstrap samples must be non-negative")
        from examples.jobpacer.analysis.phase3_results import analyze_batch
        try:
            analysis = analyze_batch(args.batch_dir, bootstrap_samples=args.bootstrap_samples,
                                     analysis_seed=args.analysis_seed)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        print(json.dumps({"accepted_blocks": analysis["accepted_complete_blocks"],
                          "planned_blocks": analysis["planned_blocks"],
                          "complete_formal_matrix": analysis["complete_formal_matrix"],
                          "validation_errors": len(analysis["validation_errors"])}, sort_keys=True))
        return 0 if not analysis["validation_errors"] else 1

    from examples.jobpacer.experiments.phase3 import finalize
    if args.finalize_action == "pilot":
        result = finalize.seal_formal_inputs(args.suite_dir, args.pilot_batch_dir)
        print(json.dumps({"suite_manifest": str((args.suite_dir / "suite-manifest.json").resolve()),
                          "status": result["status"],
                          "bare": result.get("bare", {}).get("status")}, sort_keys=True))
        return 0
    from examples.jobpacer.experiments.phase3.gates import READINESS_GATES
    audits = _audit_paths(args.audit, parser, expected=set(READINESS_GATES))
    if set(audits) != set(READINESS_GATES):
        parser.error(f"required audits: {READINESS_GATES}")
    result = finalize.finalize(args.suite_manifest, args.qualification, audits)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
