# jevk5 (`jevk5-v1`)

**alibiserikbay/JevK5** v0.3 is Qwen3.5-4B with a distilled LoRA merged into the weights (Apache-2.0).
It reads the next-token logits of the answer letters after a SemIf-style prompt: a fixed system
instruction and the decision as JSON (evidence, criterion, lettered options). The author publishes
GGUF builds in **alibiserikbay/JevK5-GGUF** and a runtime at https://github.com/allebee/jevk5.
Ollaya ships the 4B Q8_0 file as `jevk5:4b` (also `jevk5:latest`).

| | |
|---|---|
| Repo / commit | `alibiserikbay/JevK5-GGUF@ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a` |
| File | `jevk5-4b-v0.3-Q8_0.gguf` 4.48 GB, sha256 `aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30` (matches the repo's `SHA256SUMS`) |
| Source model | `alibiserikbay/JevK5@c4f7fdb3aeab5582336406e78d3bef11bf98833d` (v0.3, bf16), on Qwen3.5-4B |
| License | Apache-2.0, with the author's `NOTICE` in the GitHub repo: Qwen3.5 (Apache-2.0), the prompt and readout adapted from SemIf (MIT). The NOTICE also says 14,138 of v0.3's training questions were written and checked by an OpenAI model, "subject to OpenAI's terms". Ollaya does not redistribute the file; it is fetched from the author's repo. |
| Reference code | `allebee/jevk5@f944fe37ff1d5ed3830aa4c8d88b7189c8c1268a` (v0.3.3): `jevk5/prompt.py` (prompt, options, knockout), `jevk5/gguf.py` (`JevK5GGUF`, the llama.cpp client) |
| Temperature | `1.22`, the author's value for the v0.3 4B files (JevK5-GGUF card, "Run it"; `jevk5_config.json` in `alibiserikbay/JevK5`). The same config's `knockout_temperature` 0.93 applies only above 16 options, which `jevk5-v1` does not read. |
| Architecture | `qwen35` (Gated DeltaNet and attention layers). The author made the file with llama.cpp `9575389` and `--no-mtp`; Ollaya's pinned build (b11146) loads it unpatched. |

## Engine: llama.cpp in the runner

As for [winnow](winnow.md), the runner builds the prompt, tokenizes it with the GGUF's vocabulary and
reads the label logits at the last token with `llama_get_logits_ith`; see
[llm-logits.md](llm-logits.md) for the engine and [decisions/0003-llama-cpp-runtime.md](../decisions/0003-llama-cpp-runtime.md).

- **Plan: `cold`.** Qwen3.5's recurrent layers keep a running state, not per-token cache entries, so
  llama.cpp cannot cut the cache back to a shared prefix. Every question is one cold pass
  (`decision.json`: `"plan": "cold"`), as the author's client does (`cache_prompt: false`). The
  numbers never depend on what ran before.
- **Context.** 16,384 tokens (`n_ctx`). The author's example uses 8,192; the model reads far more, and
  the attention cache for 16,384 tokens is small on this architecture.

## Prompt (port of `jevk5/prompt.py`)

```
user   = json.dumps({"evidence": state, "criterion": instructions,
                     "options": [{"letter": "A", "description": text_0}, ...]}, ensure_ascii=False)
pre    = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
post   = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompt = pre + user + post                                  (the author's text, byte for byte)
ids    = tokenize(pre, parse_special=true) ⧺ tokenize(user, parse_special=false)
         ⧺ tokenize(post, parse_special=true)               (no BOS)

SYSTEM = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option.
          Respond with only its uppercase letter, with no explanation or reasoning."    (one line)
```

- `json.dumps` is Python's default (`", "` / `": "`, non-ASCII kept), as in `pyjson.rs`. The state and
  instructions go in as JSON values, whatever their type.
- **Control tokens.** Special tokens are parsed only in Ollaya's own template pieces; the user
  message (state, question and options) is tokenized with special parsing off, as in `winnow-v1` and
  `llm-logits-v1`. A state containing `<|im_end|>` therefore stays text and can never close the user
  turn. This departs from the author: `JevK5GGUF` and the transformers tokenizer parse specials in the
  whole prompt. The ids differ only when the request contains control-token text: on the pinned
  tokenizer, 447 of the 450 fixture prompts have identical ids both ways, and the 3 that differ are
  the `dec/control_tokens` case (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<|fim_*|>` in the
  state). All template pieces end in `\n` or a special token, so no BPE merge crosses a boundary.
- No BOS: Qwen3.5 adds none, and the reference tokenizes with `add_special=false`.

### Options (`decision_options`)

Texts are `"{id}: {description}"`, where a description that is not a string is written as Python
`str()` would (`pyrepr.rs`).

| Type | Prompt order (letters) | Description | Wire order |
|---|---|---|---|
| noul | `true` (A), `false` (B) | the criterion's description if truthy, else `The proposition is {k}.` | `[false, true]` = `[z_B, z_A]` |
| choice | criteria keys in order (a list of labels is `dict.fromkeys`) | the description if truthy, else the key | key order |
| score | levels `0..K-1` | the level as `str()` (`None` for null) | level order |

- **Labels.** `A` to `P`, each one token in this GGUF (ids 32 to 47). `decision.json` stores them and
  the runner checks them against the GGUF at load time.
- **Option logits.** The letters' logits at the last token. The author's calibration is a softmax over
  them at T = 1.22 (`calibration.json`: `[1.22, 1.22, 1.22]`); Ollaya's daemon applies it as for every
  family.
- **More than 16 options.** The author's runtime reads them in several passes (groups of up to 16,
  then a final of the best, `prompt.spread` "knockout") and sharpens the combined distribution with a
  second temperature. That combination is not ported: `jevk5-v1` answers more than 16 options with
  422 TOO_MANY_OPTIONS. A single-option choice needs no model call.

### Validation (Ollaya's)

The reference raises on malformed input rather than validating it. `jevk5-v1` answers these with 400:
an unknown type; noul criteria that are not an object; choice criteria that are neither an object nor a
list of strings; score criteria that are not an array; a question with no options; and a prompt that
does not fit the context (the state is never cut, because it sits inside the JSON payload).

## Reference implementation

`convert/ollaya_convert/families/jevk5/ref.py` ports `prompt_text` and `decision_options` and adds
the validation above. `prompt_goldens.py --jevk5-upstream <checkout>` checks every fixture prompt
against the author's own `jevk5.prompt` (450 prompts, byte-identical).

## Parity / quality

Goldens: `export_llama.py jevk5` sends each test request through `ref.py` to a stock `llama-server` of
the pinned build (b11146) loaded with the author's GGUF, one cold pass per question, reading the
letters with the bias trick ([llm-logits.md](llm-logits.md)). The runner must match every decision,
with option logits within 1e-3.

Measured on choso-wsl, 2026-09-26 (123 cases: 40 typed-decisions rows, Laya's and the decoder edge
cases, and the extra cases of `export_llama.py`; 4 are rejected by both, 593 questions are scored):

| Model | Device | Questions | Decisions | Option logits max | Probabilities max |
|---|---|---|---|---|---|
| 4b Q8_0 | CUDA, RTX 4090 | 593 | 593/593 | 7.7e-6 | 1.6e-6 |

- **Rejected cases.** Three questions with 30, 40 and 77 options (TOO_MANY_OPTIONS) and the 9,000-word
  state (35,073 tokens, beyond the context).
- **Latency (measured here, in the runner without HTTP, five questions, short state).** RTX 4090:
  288 ms at the median (one cold pass per question).
- **Not run.** The CPU, Metal, Windows, typed-decisions quality and a comparison with the author's
  transformers runtime (bf16).

Published numbers (JevK5-GGUF card, the author's runs on JevBench's 231 public items, not official
JevBench results): the Q8_0 file gives the same answer as bf16 on 229 of 231, with tier accuracies
easy 1.000, standard 0.944, hard 0.784, the same as bf16.

## Limits

- **Options.** 1 to 16 per question; more is TOO_MANY_OPTIONS (the author's knockout is not ported).
- **Context.** State, question and options share 16,384 tokens; a longer prompt is rejected, not cut.
- **Control tokens.** Text in the state, question and options is never parsed as a control token
  (unlike the author's runtime).
- **Cost.** One cold pass per question: the state is evaluated again for every question.
