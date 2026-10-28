"""LCM / DDIM sampling schedule (clean-room implementation).

The reference application ships a per-step coefficient table (``sxsched.json``)
with one row per sampling step:

===========  =====================================================
field        meaning
===========  =====================================================
``t``        timestep index
``aT``       ``alpha_bar`` at this step
``aPrev``    ``alpha_bar`` at the previous step
``cSkip``    LCM "skip" coefficient for the current sample
``cOut``     LCM "out" coefficient for the model prediction
===========  =====================================================

Given the model output the scheduler performs either a ``v``-prediction
denoise or an ``eps``-prediction denoise, then applies either the DDIM update
or the LCM update (with optional per-step gaussian noise on all but the last
step).  The arithmetic below is a from-scratch reimplementation of that
contract; the coefficient table itself is data, supplied by the caller.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["Row", "Schedule", "epsilon"]


def epsilon(alpha_bar: float) -> float:
    return math.sqrt(1.0 - alpha_bar)


@dataclass(frozen=True)
class Row:
    t: int
    a_t: float
    a_prev: float
    c_skip: float
    c_out: float


class Schedule:
    """Coefficient table keyed by step count."""

    def __init__(self, table: dict[int, list[Row]]):
        if not table:
            raise ValueError("empty schedule table")
        self.table = {int(k): list(v) for k, v in table.items()}

    # ------------------------------------------------------------------ load
    @classmethod
    def from_json(cls, path: str | Path) -> "Schedule":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = raw["steps"]
        table: dict[int, list[Row]] = {}
        for n, rows in steps.items():
            table[int(n)] = [
                Row(
                    t=int(r["t"]),
                    a_t=float(r["aT"]),
                    a_prev=float(r["aPrev"]),
                    c_skip=float(r["cSkip"]),
                    c_out=float(r["cOut"]),
                )
                for r in rows
            ]
        return cls(table)

    # ------------------------------------------------------------------ query
    def supported(self) -> list[int]:
        return sorted(self.table)

    def rows(self, steps: int) -> list[Row]:
        try:
            return self.table[int(steps)]
        except KeyError:
            raise KeyError(f"no schedule for {steps} steps (have {self.supported()})") from None

    def timesteps(self, steps: int) -> list[int]:
        return [r.t for r in self.rows(steps)]

    def init_sigma(self) -> float:
        return 1.0

    # ------------------------------------------------------------------- step
    def step(
        self,
        steps: int,
        index: int,
        x: list[float],
        out: Sequence[float],
        rnd: random.Random | None = None,
        *,
        v_pred: bool = False,
        ddim: bool = False,
    ) -> list[float]:
        """Advance ``x`` by one step in place-friendly fashion (returns new list)."""
        if len(x) != len(out):
            raise ValueError(f"shape mismatch: x={len(x)} out={len(out)}")
        rows = self.rows(steps)
        row = rows[index]
        rnd = rnd or random.Random()
        sqrt_at = math.sqrt(row.a_t)
        sqrt_one_minus_at = epsilon(row.a_t)
        sqrt_prev = math.sqrt(row.a_prev)
        sqrt_one_minus_prev = epsilon(row.a_prev)
        last = index == len(rows) - 1

        y = [0.0] * len(x)
        for i, (xi, oi) in enumerate(zip(x, out)):
            if v_pred:
                pred_x0 = xi * sqrt_at - oi * sqrt_one_minus_at
            else:
                pred_x0 = (xi - oi * sqrt_one_minus_at) / sqrt_at
            if ddim:
                direction = (xi * sqrt_one_minus_at + oi * sqrt_at) if v_pred else oi
                y[i] = pred_x0 * sqrt_prev + direction * sqrt_one_minus_prev
            else:
                v = pred_x0 * row.c_out + xi * row.c_skip
                if not last:
                    v = v * sqrt_prev + rnd.gauss(0.0, 1.0) * sqrt_one_minus_prev
                y[i] = v
        return y
