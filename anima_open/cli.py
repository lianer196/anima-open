"""Command line entry point: ``python3 -m anima_open``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .clip_bpe import CONTEXT, ClipBpe
from .lcm import Schedule

DEFAULT_VOCAB = Path("assets/clip/vocab.json")
DEFAULT_MERGES = Path("assets/clip/merges.txt")


def _tokenize(args: argparse.Namespace) -> int:
    tok = ClipBpe.from_files(args.vocab, args.merges)
    ids = tok.encode(args.text, context=CONTEXT)
    if args.json:
        print(json.dumps({"text": args.text, "ids": ids, "len": len(ids)}))
        return 0
    inv = {v: k for k, v in tok.encoder.items()}
    print(f"text   {args.text!r}")
    print(f"len    {len(ids)} / {CONTEXT}")
    for i, tid in enumerate(ids):
        print(f"  [{i:2d}] {tid:6d}  {inv.get(tid, '?')}")
    return 0


def _schedule(args: argparse.Namespace) -> int:
    sched = Schedule.from_json(args.schedule)
    if args.steps is None:
        print("supported step counts:", sched.supported())
        return 0
    rows = sched.rows(args.steps)
    print(f"steps={args.steps} init_sigma={sched.init_sigma()} rows={len(rows)}")
    for i, r in enumerate(rows):
        print(f"  [{i}] t={r.t} aT={r.a_t:.6f} aPrev={r.a_prev:.6f} cSkip={r.c_skip:.6f} cOut={r.c_out:.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anima_open", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokenize", help="tokenize a prompt to 77 CLIP ids")
    t.add_argument("text")
    t.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    t.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=_tokenize)

    s = sub.add_parser("schedule", help="print a sampling schedule table")
    s.add_argument("--schedule", type=Path, required=True)
    s.add_argument("--steps", type=int)
    s.set_defaults(func=_schedule)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
