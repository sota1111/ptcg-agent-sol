"""Command-line entry point."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ptcg_agent.agent import choose_action
from ptcg_agent.config import RuntimeConfig, load_config
from ptcg_agent.training import evaluate, load_checkpoint, train


def cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    torch = importlib.import_module("torch")
    cuda = getattr(torch, "cuda")  # noqa: B009 - optional dependency is loaded dynamically
    return bool(cuda.is_available())


def check_device(config: RuntimeConfig) -> None:
    if config.device == "cuda" and not cuda_available():
        raise RuntimeError("CUDA was requested but is unavailable; see README GPU setup")


def run_stream(config: RuntimeConfig, checkpoint: Path | None = None) -> int:
    check_device(config)
    policy = load_checkpoint(checkpoint, config.seed)[0] if checkpoint else None
    started = time.monotonic()
    budget_seconds = config.max_hours * 60 * 60
    for line_number, line in enumerate(sys.stdin, start=1):
        if time.monotonic() - started >= budget_seconds:
            print("compute budget exhausted", file=sys.stderr)
            return 124
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            print(json.dumps(choose_action(request, policy), separators=(",", ":")), flush=True)
        except (json.JSONDecodeError, ValueError) as error:
            print(f"line {line_number}: {error}", file=sys.stderr)
            return 2
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ptcg-agent")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "smoke", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/cpu.toml"))
        command.add_argument("--max-hours", type=float)
        if name == "run":
            command.add_argument("--checkpoint", type=Path)
    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    download = data_sub.add_parser("download")
    download.add_argument("--competition", required=True)
    download.add_argument("--output", type=Path, default=Path("data/raw"))
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, default=Path("configs/cpu.toml"))
    training.add_argument("--max-hours", type=float)
    training.add_argument("--episodes", type=int, default=10_000)
    training.add_argument("--checkpoint", type=Path, default=Path("artifacts/policy.json"))
    training.add_argument("--checkpoint-every", type=int, default=100)
    training.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--config", type=Path, default=Path("configs/cpu.toml"))
    evaluation.add_argument("--max-hours", type=float)
    evaluation.add_argument("--episodes", type=int, default=1_000)
    evaluation.add_argument("--checkpoint", type=Path, default=Path("artifacts/policy.json"))
    return root


def _load(args: argparse.Namespace) -> RuntimeConfig:
    return load_config(args.config, args.max_hours)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            config = _load(args)
            check_device(config)
            status = {"status": "ok", "device": config.device, "max_hours": config.max_hours}
            print(json.dumps(status))
            return 0
        if args.command == "smoke":
            config = _load(args)
            check_device(config)
            result = choose_action({"request_id": "smoke", "legal_actions": ["pass"]})
            print(json.dumps({"status": "ok", "config": config.device, "response": result}))
            return 0
        if args.command == "run":
            return run_stream(_load(args), args.checkpoint)
        if args.command == "train":
            config = _load(args)
            check_device(config)
            _, training_result = train(
                seed=config.seed,
                episodes=args.episodes,
                checkpoint=args.checkpoint,
                max_seconds=config.max_hours * 60 * 60,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
            )
            print(
                json.dumps(
                    {
                        "status": "interrupted" if training_result.interrupted else "complete",
                        **training_result.__dict__,
                    }
                )
            )
            return 124 if training_result.interrupted else 0
        if args.command == "evaluate":
            config = _load(args)
            check_device(config)
            policy, _ = load_checkpoint(args.checkpoint, config.seed)
            print(json.dumps(evaluate(policy, config.seed, args.episodes), sort_keys=True))
            return 0
        if args.command == "data" and args.data_command == "download":
            args.output.mkdir(parents=True, exist_ok=True)
            return subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    args.competition,
                    "-p",
                    str(args.output),
                ],
                check=False,
            ).returncode
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
