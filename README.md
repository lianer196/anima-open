# anima-open

A clean-room, dependency-free reference implementation of the *prompt → LCM
sampling* half of an on-device anime diffusion pipeline, plus the tooling used
to document the behavioural contract it was derived from.

This repository contains **only** code written from scratch against observed
behaviour and published algorithm descriptions. It ships **no** binary, model
weights, vocabulary file, artwork or upstream source of any kind.

---

## Why this exists

An on-device diffusion app (`com.anima.npu`, marketed as *MTK Paint*) generates
anime images locally on MediaTek Dimensity NPUs. Its user-visible behaviour —
prompt tokenisation, the LCM coefficient schedule, the conditioning/timestep
contract between the text encoder and the DiT — is not documented anywhere, so
independent clients and ports have no way to interoperate.

This project reconstructs that contract from first principles and publishes it
in executable form: the two pieces that are pure, pure-ish maths and therefore
verifiable, namely **tokenisation** and **scheduling**. The NPU graph execution
itself stays out of scope — it is hardware- and artefact-specific.

## What is in the box

| Path | Contents |
| --- | --- |
| `anima_open/clip_bpe.py` | CLIP-style byte-level BPE tokenizer, 77-token context, `</w>` end-of-word marker, special-token passthrough |
| `anima_open/lcm.py` | LCM / DDIM step update from a per-step coefficient table (`t`, `aT`, `aPrev`, `cSkip`, `cOut`) |
| `anima_open/cli.py` | `python3 -m anima_open tokenize` / `schedule` for eyeballing both |
| `tools/extract_catalog.py` | Reads a decompiled model table and emits normalised `docs/catalog.json` |
| `docs/CONTRACT.md` | The reconstructed behavioural contract, field by field |
| `docs/FINDINGS.md` | How the contract was derived, and what was deliberately not reproduced |
| `tests/` | 17 tests; the tokenizer is asserted against 8 recorded vectors |

## Install / run

```bash
git clone https://github.com/<you>/anima-open && cd anima-open
python3 -m unittest discover -s tests -v   # no dependencies required

python3 -m anima_open tokenize "1girl, silver hair, night city"
python3 -m anima_open schedule --steps 8 --json path/to/sxsched.json
```

The tokenizer needs a CLIP vocabulary (`vocab.json` + `merges.txt`, 49 408
tokens) and the scheduler needs a coefficient table. Neither is redistributed
here; see [docs/FINDINGS.md](docs/FINDINGS.md) for how to produce your own from
public artefacts.

## Correctness

```
$ python3 -m unittest discover -s tests -v
Ran 17 tests in 0.052s
OK
```

Without a vocabulary on disk the suite still runs and reports
`Ran 14 tests ... OK (skipped=3)`; the three skipped cases are the ones needing
real CLIP ids. `test_fixtures` compares against 8 recorded token-id vectors
covering CJK + emoji, contractions, per-character digit splitting, tab/space
runs, oversized inputs and literal special tokens.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one hard rule: **no upstream
artefacts**. Patches that add decompiled code, model files, vocabularies or
scraped assets will be rejected.
