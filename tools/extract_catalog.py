#!/usr/bin/env python3
"""Extract the model catalogue out of a decompiled ``Catalog.java``.

The reference application keeps its model table as a single static block of
``new ModelDef(...)`` calls.  This script walks those calls and emits a
normalised ``catalog.json`` so the data can be reviewed, diffed and reused by
an independent client.

Usage
-----
    python3 tools/extract_catalog.py path/to/Catalog.java -o docs/catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STR = r'"((?:[^"\\]|\\.)*)"'
ASSET = re.compile(r'new Asset\(\s*' + STR)
PAIR = re.compile(r'm100to\(\s*(\d+)\s*,\s*(\d+)\s*\)')
INT_LIST = re.compile(r'Integer\[\]\s*\{([^}]*)\}')
VAE_CALL = re.compile(r'vaeAsset\(\s*' + STR)


def split_args(call: str) -> list[str]:
    """Split the argument list of a constructor call at top-level commas."""
    args, depth, cur, in_str, esc = [], 0, [], False, False
    for ch in call:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def find_calls(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"new ModelDef\(", text):
        i = m.end()
        depth = 1
        in_str = esc = False
        start = i
        while i < len(text) and depth:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(text[start:i])
    return out


def parse_call(call: str) -> dict:
    a = split_args(call)
    lit = lambda idx: (re.fullmatch(STR, a[idx]).group(1) if idx < len(a) and re.fullmatch(STR, a[idx]) else None)
    ints = lambda idx: (int(a[idx]) if idx < len(a) and re.fullmatch(r"-?\d+", a[idx]) else None)

    steps = INT_LIST.search(a[5]) if len(a) > 5 else None
    vae = VAE_CALL.search(a[8]) if len(a) > 8 else None
    text_asset = ASSET.search(a[9]) if len(a) > 9 else None
    rope = None
    for arg in a[16:20]:
        hit = ASSET.search(arg) if arg else None
        if hit and hit.group(1).startswith("rope"):
            rope = hit.group(1)
            break

    return {
        "id": lit(0),
        "title": lit(1),
        "subtitle": lit(2),
        "arch": lit(3),
        "native_steps": ints(4),
        "step_choices": [int(x) for x in steps.group(1).split(",") if x.strip()] if steps else [],
        "resolutions": [[int(h), int(w)] for h, w in PAIR.findall(a[6])] if len(a) > 6 else [],
        "dit_graphs": [m.group(1) for m in ASSET.finditer(a[7])] if len(a) > 7 else [],
        "vae": vae.group(1) if vae else None,
        "text_weights": text_asset.group(1) if text_asset else None,
        "mods_prefix": lit(11),
        "grid_h": ints(12),
        "grid_w": ints(13),
        "rope": rope,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog_java", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("docs/catalog.json"))
    args = ap.parse_args()

    text = args.catalog_java.read_text(encoding="utf-8", errors="replace")
    models = [parse_call(c) for c in find_calls(text)]
    models = [m for m in models if m["id"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(models)} models -> {args.out}")
    for m in models:
        print(f"  {m['id']:32s} steps={m['step_choices']} res={m['resolutions'][:2]} graphs={len(m['dit_graphs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
