"""Reference implementation of `jevk5-v1`: alibiserikbay/JevK5's typed-decision prompt, ported from
`jevk5/prompt.py` of the author's runtime (https://github.com/allebee/jevk5 at v0.3.3,
f944fe37ff1d5ed3830aa4c8d88b7189c8c1268a), which follows SemIf (TheoLeeCJ/SemIf, MIT).

    user   = json.dumps({"evidence": state, "criterion": instructions,
                         "options": [{"letter": "A", "description": text_0}, ...]}, ensure_ascii=False)
    prompt = "<|im_start|>system\\n" + SYSTEM + "<|im_end|>\\n<|im_start|>user\\n" + user
             + "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
    ids    = tokenize(PRE, parse_special=True) + tokenize(user) + tokenize(POST, parse_special=True)

Option texts are "id: description" (`decision_options`): noul reads `true` (A) then `false` (B), with
"The proposition is {k}." for a missing description; a choice falls back to its id; a score level is
its index. The option logits are the letters' logits at the last token; the author's calibration is
one temperature over them (`jevk5_config.json`).

The text is the author's prompt byte for byte. The author tokenizes it whole with special parsing, so
`<|im_end|>` inside a state would become a control token; Ollaya parses specials only in its own
template pieces (PRE, POST), as winnow-v1 and llm-logits-v1 do. Without control-token text in the
request the ids are the same.

The author's runtime reads more than 16 options in several passes combined by a knockout
(`prompt.spread`) with a second temperature. `jevk5-v1` covers up to 16 options, one pass per
question; more options are TOO_MANY_OPTIONS.
"""
from __future__ import annotations

import json

LAYOUT = "jevk5-v1"
LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
PRE = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
assert CHAT_TEMPLATE.format(system=SYSTEM, user="\x00") == PRE + "\x00" + POST
UPSTREAM = {"runtime": "https://github.com/allebee/jevk5", "commit": "f944fe37ff1d5ed3830aa4c8d88b7189c8c1268a",
            "source": "jevk5/prompt.py"}


class JevK5Error(ValueError):
    """A question the layout rejects (HTTP 400)."""


class TooManyOptions(JevK5Error):
    """More options than one pass reads (HTTP 422)."""


def render_state(state) -> str:
    """The state as `json.dumps` writes it inside the user message (its token count is reported)."""
    return json.dumps(state, ensure_ascii=False)


def user_message(state, criterion, texts) -> str:
    """The user message of `prompt.messages`: the decision as JSON."""
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(texts)],
    }
    return json.dumps(payload, ensure_ascii=False)


def prompt_text(state, criterion, texts) -> str:
    """`prompt.prompt_text`: the full prompt, chat template included."""
    return CHAT_TEMPLATE.format(system=SYSTEM, user=user_message(state, criterion, texts))


def compile_question(qid, q):
    """`prompt.decision_options` plus Ollaya's validation -> (type, keys in wire order, option texts in
    prompt order, prompt position of each wire option)."""
    if not isinstance(q, dict):
        raise JevK5Error("question %r: invalid named question" % qid)
    kind = q.get("type")
    crit = q.get("criteria")
    if kind == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise JevK5Error("question %r: noul criteria must be an object" % qid)
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
        keys, wire = ["false", "true"], [1, 0]
    elif kind == "choice":
        if isinstance(crit, list):
            if not all(isinstance(k, str) for k in crit):
                raise JevK5Error("question %r: choice labels must be strings" % qid)
            crit = dict.fromkeys(crit)
        if not isinstance(crit, dict):
            raise JevK5Error("question %r: choice criteria must be an object" % qid)
        pairs = [(k, v or k) for k, v in crit.items()]
        keys, wire = [k for k, _ in pairs], list(range(len(pairs)))
    elif kind == "score":
        if not isinstance(crit, list):
            raise JevK5Error("question %r: score criteria must be an ordered array" % qid)
        pairs = [(str(i), level) for i, level in enumerate(crit)]
        keys, wire = [k for k, _ in pairs], list(range(len(pairs)))
    else:
        raise JevK5Error("question %r: unknown question type" % qid)
    if not pairs:
        raise JevK5Error("question %r: the question has no options" % qid)
    if len(pairs) > len(LETTERS):
        raise TooManyOptions("question %r: %d options; jevk5-v1 reads at most %d" % (qid, len(pairs), len(LETTERS)))
    return kind, keys, [f"{k}: {d}" for k, d in pairs], wire


def compile_request(state, questions):
    """Every question's (qid, type, keys, user message, wire order), in request order. The request is
    rejected as a whole when any question is."""
    if not isinstance(questions, dict) or not questions:
        raise JevK5Error("questions must contain at least one named question")
    out = []
    for qid, q in questions.items():
        kind, keys, texts, wire = compile_question(qid, q)
        out.append((qid, kind, keys, user_message(state, q.get("instructions"), texts), wire))
    return out
