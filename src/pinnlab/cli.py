"""Explicit train, audit, replay, and render commands; no cloud services."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import webbrowser
from pathlib import Path


def _source_demo() -> Path:
    candidates = [
        Path.cwd() / "examples/taylor-green",
        Path(__file__).resolve().parents[2] / "examples/taylor-green",
    ]
    for candidate in candidates:
        if (candidate / "checkpoint.pt").is_file():
            return candidate
    raise FileNotFoundError(
        "Bundled demo not found. Run from a repository checkout or pass --run PATH."
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural fluid lab: simulate, compare, verify.")
    p.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = p.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train from PDE and initial conditions, without future labels."
    )
    train.add_argument("--output", type=Path, default=Path("runs/train"))
    train.add_argument("--nu", type=float, default=0.1)
    train.add_argument("--tmax", type=float, default=1.0)
    train.add_argument("--mode", type=int, default=1)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--depth", type=int, default=3)
    train.add_argument("--adam-steps", type=int, default=1500)
    train.add_argument("--lbfgs-steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=1)
    train.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    train.add_argument("--device", default="cpu")
    train.add_argument(
        "--soft-ic", action="store_true", help="Use an IC loss instead of a hard constraint."
    )
    train.add_argument("--overwrite", action="store_true")

    audit = commands.add_parser("evaluate", help="Audit a saved checkpoint on independent points.")
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--grid", type=int, default=48)
    audit.add_argument("--frames", type=int, default=21)
    audit.add_argument("--residual-points", type=int, default=2048)
    audit.add_argument("--dt", type=float, default=0.02)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--threads", type=int, default=1)

    render = commands.add_parser("render", help="Render existing audited arrays without training.")
    render.add_argument("--run", type=Path, required=True)
    render.add_argument("--no-html", action="store_true")
    render.add_argument("--no-gif", action="store_true")

    demo = commands.add_parser(
        "demo", help="Run the bundled trained model and open an offline animation."
    )
    demo.add_argument("--run", type=Path, help="Alternate audited run to replay.")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument(
        "--no-open", action="store_true", help="Headless operation: only write artifacts."
    )
    demo.add_argument(
        "--interactive",
        action="store_true",
        help="Open local Matplotlib quantity/reference controls.",
    )
    demo.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "train":
            from .train import TrainConfig, train

            if (args.output / "checkpoint.pt").exists() and not args.overwrite:
                raise FileExistsError(
                    "Checkpoint exists; use a new --output or explicitly pass --overwrite."
                )
            cfg = TrainConfig(
                nu=args.nu,
                tmax=args.tmax,
                mode=args.mode,
                width=args.width,
                depth=args.depth,
                hard_ic=not args.soft_ic,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                threads=args.threads,
                dtype=args.dtype,
                device=args.device,
            )
            result = train(cfg, args.output)
            print(json.dumps(result, indent=2))
        elif args.command == "evaluate":
            import torch
            from .evaluate import evaluate

            if args.threads < 1:
                raise ValueError("--threads must be positive")
            torch.set_num_threads(args.threads)
            result = evaluate(
                args.run,
                n=args.grid,
                frames=args.frames,
                residual_points=args.residual_points,
                dt=args.dt,
                device=args.device,
            )
            print(
                json.dumps(
                    {
                        "audit": result["audit"],
                        "final_time": result["per_time"][-1],
                        "evaluation": result["evaluation"],
                    },
                    indent=2,
                )
            )
        elif args.command == "render":
            from .visualize import render_report

            for path in render_report(args.run, html=not args.no_html, gif=not args.no_gif):
                print(path.resolve())
        elif args.command == "demo":
            from .visualize import render_report, show

            source = args.run if args.run is not None else _source_demo()
            required = ["checkpoint.pt", "training.json", "history.csv"]
            for name in required:
                if not (source / name).is_file():
                    raise FileNotFoundError(f"Missing {source / name}; train the run first.")
            if args.output.resolve() == source.resolve():
                raise ValueError("Demo output must differ from the source run.")
            if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
                raise FileExistsError(
                    "Demo output is not empty; use a new --output or --overwrite."
                )
            args.output.mkdir(parents=True, exist_ok=True)
            for name in required:
                shutil.copy2(source / name, args.output / name)
            import torch
            from .evaluate import evaluate

            torch.set_num_threads(1)
            # The demo actually loads and evaluates the network. It does not
            # simply replay cached exact fields or previously exported frames.
            metrics = evaluate(args.output)
            print(
                json.dumps(
                    {
                        "checkpoint_sha256": metrics["checkpoint_sha256"],
                        "final_velocity_relative_l2": metrics["per_time"][-1][
                            "pinn_velocity_relative_l2"
                        ],
                    },
                    indent=2,
                )
            )
            for path in render_report(args.output):
                print(path.resolve())
            if args.interactive:
                show(args.output)
            elif not args.no_open:
                webbrowser.open((args.output / "comparison.html").resolve().as_uri())
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        print(f"pinnlab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
