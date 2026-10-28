# How the contract was derived

Everything in `docs/CONTRACT.md` was reconstructed from observable behaviour of
a distributed Android application plus the public algorithm literature. This
file records the derivation so a reader can judge how much to trust each claim,
and explicitly states what was **not** copied.

## Provenance of each fact

| Fact | How it was established | Confidence |
| --- | --- | --- |
| Context length 77, BOS 49406, EOS/PAD 49407 | Recorded token-id vectors for 8 prompts | High — reproduced exactly |
| `</w>` end-of-word marker, byte-level fallback | Vocabulary structure (`girl</w>` as one entry) + merge table | High — 8/8 fixtures |
| Per-character digit splitting | Fixture `3.14 100%` diverges unless digits split | High |
| Contraction grouping | Fixture `don't it's` | High |
| NFC + lowercase on the BPE side, NFKC on the wordpiece side | Behavioural comparison of mixed-width CJK input | Medium |
| Table-driven LCM update, `init_sigma = 1.0` | Coefficient table shape, final-step noise suppression, closed-form check | High for the algebra, medium for the intended table values |
| `cSkip`/`cOut` semantics (skip/out, not `orig_prev`) | Algebraic fit against the DDIM branch | Medium |
| SoC- and runtime-keyed artefacts, refuse-on-mismatch | Distinct user-facing error messages for "artefact/runtime major mismatch" | High |
| Fixed graph-count split (7 DiT + 1 decoder) | Catalogue entries list 7 sequential graphs + VAE graph | High |

## What this repository deliberately does not contain

- No decompiled or transpiled code from the reference application. The
  implementation here was written against the contract, not transcribed.
- No model weights, `.npug`/`.dla`/`.tflite` artefacts, tokenizer vocabulary,
  fonts, icons, or artwork.
- No server endpoints, credentials, signing certificates, or community content.
- No obfuscation-defeat, licence-check bypass, or signature-spoofing logic. The
  reference application checks its own signing certificate and refuses to run
  when repackaged; that check is a technical measure and was left alone.

## Reproducing the data inputs

Both data inputs are obtained independently of this repository:

1. **Tokenizer vocabulary** — export any CLIP BPE vocabulary with the
   `</w>` end-of-word convention into `vocab.json` + `merges.txt`.
2. **Sampling schedule** — a per-step coefficient table. If you have a table
   in another format, convert it to the JSON shape used by
   `Schedule.from_json`:

   ```json
   { "steps": { "8": [ { "t": 999, "aT": 1.0, "aPrev": 0.9, "cSkip": 0.1, "cOut": 0.9 } ] } }
   ```

   A table can be generated for any `alpha_bar` schedule by evaluating the
   LCM consistency coefficients at each step; the formula is in the LCM paper.

## Catalog extraction

`tools/extract_catalog.py` reads a **decompiled** model table and normalises it.
It exists to make a data table reviewable and diffable. Its output contains
model ids, resolutions, step counts and artefact filenames — configuration
metadata, not code. The output is a local artefact and is git-ignored; do not
commit it, and never commit checkpoint blobs or anything loaded into those
graphs.

## Known gaps

- The second text tower's wordpiece merge order is not verified; only its
  whitespace splitting is.
- The exact RNG stream (draw order across tensors) is unverified, so seeded
  reproducibility across implementations is not guaranteed.
- Classifier-free guidance scaling is applied outside these two modules; the
  contract stops at the per-step update.
