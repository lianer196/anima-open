# Contributing

Thanks for looking. Three rules, in order of importance.

## 1. No upstream artefacts

This repository must stay legally clean. Do **not** open a PR containing:

- decompiled, disassembled or transpiled code from any other app
- model weights, NPU graph artefacts (`.npug`, `.dla`, `.tflite`), tokenizer
  vocabularies, fonts, icons, screenshots or artwork
- server endpoints, API keys, certificates, or user data
- anything that defeats a licence check, signature check or integrity check

`ci.yml` has a `guard` job that fails the build if a binary or vocabulary file
is tracked. If your change needs test data, generate it in the test at runtime
or use a tiny synthetic fixture.

## 2. One behaviour per PR

The contract in `docs/CONTRACT.md` is the specification. A PR should either

- fix the implementation to match the contract, **with a test that fails before
  and passes after**, or
- change the contract, **with an updated `docs/FINDINGS.md` row explaining how
  the new behaviour was established**.

Behavioural claims need evidence. "It felt right" is not evidence; a fixture, a
closed-form derivation, or a reproducible observation is.

## 3. Keep it dependency-free

`anima_open/` runs on the standard library alone. `tools/` may use more, but
should degrade to a helpful error when the dependency is missing. Tests use
`unittest`, not `pytest`.

## Workflow

```bash
python3 -m unittest discover -s tests -v
python3 -m anima_open tokenize "1girl, night city"   # needs a vocabulary
git checkout -b fix/my-thing
```

Commit messages: `fix:`, `feat:`, `docs:`, `test:`, `chore:`.

## Reporting a contract divergence

Open an issue with:

1. the input (prompt, step count, or coefficient row),
2. what this implementation produced,
3. what the reference produced,
4. how you observed it.

We will either fix the code or record the gap in `docs/FINDINGS.md` under
*Known gaps*.
