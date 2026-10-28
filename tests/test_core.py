"""Behavioural tests for the clean-room core.

Run with ``python3 -m unittest discover -s tests -v`` (no third-party deps).
The CLIP fixtures live in ``tests/`` and are checked in as *expected ids only*
derived from a public tokenizer artefact; the vocabulary files themselves are
not redistributed with this repository -- drop your own copy next to the tests
to enable them.
"""

from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anima_open.clip_bpe import BOS, CONTEXT, EOS, PAD, ClipBpe, pre_tokenize  # noqa: E402
from anima_open.lcm import Row, Schedule  # noqa: E402

TESTS = Path(__file__).resolve().parent
# User-supplied inputs (never committed): point ANIMA_DATA at a directory holding
# vocab.json + merges.txt, or drop them next to this file.
DATA = Path(os.environ.get("ANIMA_DATA", TESTS))


class PreTokenizeTest(unittest.TestCase):
    def test_contraction_stays_whole(self):
        self.assertEqual(pre_tokenize("it's a cat"), ["it", "'s", "a", "cat"])

    def test_punctuation_is_its_own_word(self):
        self.assertEqual(pre_tokenize("hi!!"), ["hi", "!!"])

    def test_digits_split_per_char(self):
        self.assertEqual(pre_tokenize("3.14"), ["3", ".", "1", "4"])

    def test_normalises_and_lowercases(self):
        self.assertEqual(pre_tokenize("  CAFÉ  "), ["café"])


@unittest.skipUnless((DATA / "vocab.json").exists(), "vocabulary not present (set ANIMA_DATA)")
class ClipBpeFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ClipBpe.from_files(DATA / "vocab.json", DATA / "merges.txt")
        cls.fixtures = json.loads((DATA / "clip_tok_vectors.json").read_text(encoding="utf-8"))

    def test_context_geometry(self):
        ids = self.tok.encode("a photo of a cat")
        self.assertEqual(len(ids), CONTEXT)
        self.assertEqual(ids[0], BOS)
        self.assertEqual(ids[-1], PAD)
        self.assertIn(EOS, ids)

    def test_fixtures(self):
        for case in self.fixtures["cases"]:
            with self.subTest(text=case["text"][:40]):
                self.assertEqual(self.tok.encode(case["text"]), case["l"])

    def test_overlong_prompt_is_truncated_not_crashed(self):
        ids = self.tok.encode("cat, " * 200)
        self.assertEqual(len(ids), CONTEXT)
        self.assertEqual(ids[0], BOS)
        self.assertEqual(ids[-1], EOS)


class ClipBpeSyntheticTest(unittest.TestCase):
    """Runs with no vocabulary on disk: exercises the merge engine itself."""

    @classmethod
    def setUpClass(cls):
        vocab = {
            "h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hell": 6,
            "hello": 7, "hello</w>": 8, "w": 9, "o</w>": 10,
            "<|startoftext|>": 49406, "<|endoftext|>": 49407,
        }
        merges = ["#version: 0.2", "h e", "l l", "he ll", "hell o", "hell o</w>"]
        cls.tok = ClipBpe(vocab, merges)

    def test_merges_apply_in_rank_order(self):
        self.assertEqual(self.tok.encode_word("hello"), [8])

    def test_unknown_symbol_falls_back_to_bytes(self):
        # 'x' is absent from the synthetic vocab -> byte-level fallback path
        ids = self.tok.encode_word("hx")
        self.assertTrue(ids)
        self.assertIsInstance(ids[0], int)

    def test_context_padding(self):
        ids = self.tok.encode("hello", context=5)
        self.assertEqual(ids, [49406, 8, 49407, 49407, 49407])

    def test_special_tokens_round_trip(self):
        self.assertEqual(self.tok.encode_word("<|endoftext|>"), [49407])


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        # synthetic two-step table: alpha_bar 1.0 -> 0.5 -> 0.0
        self.sched = Schedule(
            {
                2: [
                    Row(t=999, a_t=1.0, a_prev=0.5, c_skip=0.5, c_out=0.5),
                    Row(t=0, a_t=0.5, a_prev=0.0, c_skip=0.0, c_out=1.0),
                ]
            }
        )

    def test_rows_and_timesteps(self):
        self.assertEqual(self.sched.supported(), [2])
        self.assertEqual(self.sched.timesteps(2), [999, 0])

    def test_unknown_step_count_raises(self):
        with self.assertRaises(KeyError):
            self.sched.rows(4)

    def test_last_step_has_no_noise(self):
        x = [0.0, 0.0]
        out = [1.0, 1.0]
        y = self.sched.step(2, 1, x, out, random.Random(0))
        # pred_x0 = (0 - 1*sqrt(1-0.5))/sqrt(0.5) = -1 ; c_out=1, c_skip=0 -> -1
        for v in y:
            self.assertAlmostEqual(v, -1.0, places=6)

    def test_non_last_step_reads_noise_from_supplied_rng(self):
        a = self.sched.step(2, 0, [0.0], [1.0], random.Random(7))
        b = self.sched.step(2, 0, [0.0], [1.0], random.Random(7))
        c = self.sched.step(2, 0, [0.0], [1.0], random.Random(8))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_ddim_matches_closed_form(self):
        x, out = [2.0], [0.5]
        y = self.sched.step(2, 1, x, out, random.Random(0), ddim=True)
        # eps-prediction, a_t=0.5: pred_x0 = (2 - 0.5*sqrt(0.5))/sqrt(0.5)
        import math

        pred = (2.0 - 0.5 * math.sqrt(0.5)) / math.sqrt(0.5)
        self.assertAlmostEqual(y[0], pred * math.sqrt(0.0) + 0.5 * math.sqrt(1.0), places=6)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.sched.step(2, 0, [0.0], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
