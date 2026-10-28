# Behavioural contract

This is the reconstructed interface between the three stages of the reference
pipeline: **text → conditioning**, **conditioning → latent loop**, **latent →
image**. Only the parts implemented in `anima_open/` are normative here; the
NPU graph boundary is described so an independent backend can be plugged in.

## 1. Tokenisation

Fixed geometry: context length **77**.

```text
ids = [BOS] + Σ tokenize(word) + [EOS]      # BOS=49406, EOS=PAD=49407
if len(ids) > 77: truncate to 77, force ids[76] = EOS
else:            right-pad with PAD to 77
```

Two tokenizers run side by side — a CLIP BPE and a sentencepiece-style encoder
for the second text tower. Both apply Unicode normalisation first (NFC for the
BPE side, NFKC for the wordpiece side), then lowercase.

### 1.1 Word splitting (BPE side)

Rules, in order:

1. Whitespace runs collapse and act as word separators; leading/trailing
   whitespace is dropped.
2. A contraction from `'s 't 're 've 'm 'll 'd` is emitted as its own word.
3. A run of letters is one word.
4. **A digit is its own word, one character at a time.** `3.14` → `3`, `.`, `1`, `4`.
5. A run of non-space, non-letter, non-digit characters is one word.
6. Literal `<|startoftext|>` / `<|endoftext|>` are passed through as specials.

### 1.2 BPE application

Each word is mapped to bytes via the reversible 256-entry byte↔unicode table,
then `</w>` is appended to the *final byte symbol* and merges are applied in
rank order until no ranked pair remains. The end-of-word marker is the literal
string `</w>`, **not** the `Ġ` convention — vocabularies built for this
pipeline store `girl</w>` as a single token.

Unknown symbols fall back to their constituent byte symbols rather than
emitting `<unk>`.

### 1.3 Wordpiece tower

Whitespace-delimited pieces, each prefixed with `▁`; empty input yields `▁`.

## 2. Sampling schedule

Per-step coefficient table, one row per step, supplied as data:

| Field | Meaning |
| --- | --- |
| `t` | timestep index passed to the DiT |
| `aT` | `alpha_bar` at this step |
| `aPrev` | `alpha_bar` at the previous step |
| `cSkip` | LCM skip coefficient |
| `cOut` | LCM out coefficient |

Step update for each latent element `x`, model output `o`:

```text
sqrt_aT   = sqrt(aT)          sqrt_1m_aT = sqrt(1 - aT)
sqrt_prev = sqrt(aPrev)       sqrt_1m_prev = sqrt(1 - aPrev)

# prediction
pred_x0 = v-prediction ? x*sqrt_aT - o*sqrt_1m_aT
                       : (x - o*sqrt_1m_aT) / sqrt_aT

# DDIM branch
dir     = v-prediction ? x*sqrt_1m_aT + o*sqrt_aT : o
x'      = pred_x0*sqrt_prev + dir*sqrt_1m_prev

# LCM branch (default)
v       = pred_x0*cOut + x*cSkip
x'      = last_step ? v : v*sqrt_prev + N(0,1)*sqrt_1m_prev
```

Notes:

- `init_sigma` is `1.0`: the loop starts from unit-variance noise.
- Noise is drawn **per element** from the supplied `Random`, only on non-final
  steps, so a seeded run is reproducible only if the RNG consumption order
  matches exactly.
- The final step suppresses noise entirely.

## 3. NPU graph boundary (not implemented here)

The reference implementation splits the DiT into 7 sequential graph artefacts
plus a separate decoder graph, dispatches them through a JNI surface of 29
`native*` entry points, and streams progress back through a callback
(`step`, `steps`, `stage`, `perStep`, `eta`). An independent backend conforming
to this repository only needs to honour §1 and §2; how it executes the graphs
is its own business.

Compiled artefacts are keyed by SoC (`mt6985`, `mt6989`, `mt6993`, …) and by
NPU runtime major version, so an implementation must refuse to load an artefact
whose recorded `soc` / `runtimeMajor` do not match the device — the original
does exactly that and surfaces a distinct error rather than crashing.
