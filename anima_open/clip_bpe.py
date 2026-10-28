"""CLIP-style byte-level BPE tokenizer (clean-room implementation).

Vocabulary layout observed in the reference application:

* ``vocab.json``  -- 49_408 entries, token -> id, end-of-word marker is the
  literal suffix ``</w>`` (not the ``Ġ`` convention used by GPT-2).
* ``merges.txt``  -- first line is ``#version: 0.2``, then ``left right`` pairs
  ordered by merge priority.
* padding / special ids: ``bos=49406``, ``eos=49407``, ``pad=49407`` with the
  context length fixed to 77 (bos + 75 content + eos, right padded).

Only the algorithm is reproduced here; no upstream file is shipped with this
package.  Point the loader at your own copy of the vocabulary.
"""

from __future__ import annotations

import gzip
import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["ClipBpe", "bytes_to_unicode"]

END_OF_WORD = "</w>"
PAD = 49407
BOS = 49406
EOS = 49407
CONTEXT = 77


def bytes_to_unicode() -> dict[int, str]:
    """Reversible byte -> printable unicode map (GPT-2 lineage, 256 entries)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def pre_tokenize(text: str) -> list[str]:
    """Split raw text into BPE input words.

    Mirrors the reference behaviour: NFC normalisation, contractions kept as a
    unit, letter runs / digit runs / punctuation runs each kept together, and
    every word carrying the ``</w>`` suffix.
    """
    text = unicodedata.normalize("NFC", text).lower().strip()
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "<" and text.startswith("<|startoftext|>", i):
            out.append("<|startoftext|>")
            i += len("<|startoftext|>")
            continue
        if ch == "<" and text.startswith("<|endoftext|>", i):
            out.append("<|endoftext|>")
            i += len("<|endoftext|>")
            continue
        if ch == "'":
            for c in _CONTRACTIONS:
                if text.startswith(c, i):
                    out.append(c)
                    i += len(c)
                    break
            else:
                j = i
                while j < n and not text[j].isspace() and not _is_letter(text[j]):
                    j += 1
                out.append(text[i:j])
                i = j
            continue
        if _is_letter(ch):
            j = i
            while j < n and _is_letter(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if _is_number(ch):
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and not _is_letter(text[j]) and not _is_number(text[j]):
            j += 1
        out.append(text[i:j])
        i = j
    return out


class ClipBpe:
    """Merge-rule BPE with an explicit rank table."""

    def __init__(self, vocab: dict[str, int], merges: Iterable[str]):
        self.encoder = dict(vocab)
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, line in enumerate(merges):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            a, _, b = line.partition(" ")
            self.ranks[(a, b)] = rank
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, tuple[str, ...]] = {}

    # ------------------------------------------------------------------ load
    @classmethod
    def from_files(cls, vocab_path: str | Path, merges_path: str | Path) -> "ClipBpe":
        vp, mp = Path(vocab_path), Path(merges_path)
        opener = gzip.open if vp.suffix == ".gz" else open
        with opener(vp, "rt", encoding="utf-8") as fh:
            vocab = json.load(fh)
        opener = gzip.open if mp.suffix == ".gz" else open
        with opener(mp, "rt", encoding="utf-8") as fh:
            merges = fh.read().splitlines()
        return cls(vocab, merges)

    # ------------------------------------------------------------------ core
    def bpe(self, word: str) -> tuple[str, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        if word in ("<|startoftext|>", "<|endoftext|>"):
            result: tuple[str, ...] = (word,)
            self._cache[word] = result
            return result
        # byte-level: every unicode codepoint becomes one or more byte symbols
        symbols = [self.byte_encoder[b] for b in word.encode("utf-8")]
        symbols[-1] = symbols[-1] + END_OF_WORD
        while len(symbols) > 1:
            best: tuple[str, str] | None = None
            best_rank = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            a, b = best
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            if len(symbols) == 1:
                break
        result = tuple(symbols)
        self._cache[word] = result
        return result

    def encode_word(self, word: str) -> list[int]:
        if word in ("<|startoftext|>", "<|endoftext|>"):
            return [self.encoder[word]]
        ids: list[int] = []
        for piece in self.bpe(word):
            tid = self.encoder.get(piece)
            if tid is None:
                # fall back to byte-level symbols so unknown text still tokenises
                ids.extend(
                    self.encoder[self.byte_encoder[b]]
                    for b in piece.encode("utf-8")
                    if self.byte_encoder[b] in self.encoder
                )
            else:
                ids.append(tid)
        return ids

    def encode(self, text: str, context: int = CONTEXT) -> list[int]:
        ids: list[int] = [BOS]
        for word in pre_tokenize(text):
            ids.extend(self.encode_word(word))
        ids.append(EOS)
        if len(ids) > context:
            ids = ids[:context]
            ids[-1] = EOS
        else:
            ids.extend([PAD] * (context - len(ids)))
        return ids
