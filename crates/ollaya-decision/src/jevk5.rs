//! `jevk5-v1`: alibiserikbay/JevK5's typed-decision prompt (`docs/families/jevk5.md`).
//!
//! A port of `jevk5/prompt.py` in the author's runtime (github.com/allebee/jevk5 at v0.3.3,
//! `f944fe37`), following `convert/ollaya_convert/families/jevk5/ref.py`:
//!
//! ```text
//! user   = json.dumps({"evidence": state, "criterion": instructions,
//!                      "options": [{"letter": "A", "description": text_0}, ...]}, ensure_ascii=False)
//! prompt = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" + user
//!          + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
//! ids    = tok(prompt, parse_special)
//! ```
//!
//! Option texts are `"{id}: {description}"`: noul reads `true` (A) then `false` (B), a choice falls
//! back to its id, a score level is its index. The whole prompt is tokenized with special parsing,
//! as the author's llama.cpp client and the transformers tokenizer both do. One pass reads at most
//! 16 options (`A`..`P`); the author's multi-pass knockout above that is not ported.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::question::QType;
use crate::{Error, pyjson, pyrepr};

pub const LAYOUT: &str = "jevk5-v1";

pub const SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one \
listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// The answer letters, one pass's options at most.
pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
pub const MAX_OPTIONS: usize = 16;

/// `decision.json` of a `jevk5-v1` model (the fields the runtime reads).
#[derive(Debug, Clone, Deserialize)]
pub struct JevK5Config {
    pub layout: String,
    /// `A`..`P` and their single tokens.
    pub labels: crate::llm_logits::LabelTable,
}

impl JevK5Config {
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(format!("decision.json: {msg}")));
        if self.layout != LAYOUT {
            return bad(format!("layout {:?} is not {LAYOUT}", self.layout));
        }
        let l = &self.labels;
        let letters: Vec<String> = LETTERS.chars().map(String::from).collect();
        if l.strings != letters || l.ids.len() != MAX_OPTIONS {
            return bad(format!(
                "labels must be A..P: {} strings, {} ids",
                l.strings.len(),
                l.ids.len()
            ));
        }
        Ok(())
    }
}

/// One question as the model reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionPrompt {
    pub qtype: QType,
    /// Option keys in wire order: `false, true` / criteria keys / `"0".."K-1"`.
    pub keys: Vec<String>,
    /// The whole prompt, chat template included.
    pub prompt: String,
    /// The letters' tokens, in prompt order.
    pub label_ids: Vec<u32>,
    /// Prompt position of each wire option (noul: `[1, 0]`, since `true` is `A`).
    pub wire_order: Vec<usize>,
}

/// The state as `json.dumps` writes it inside the user message.
pub fn render_state(state: &Value) -> String {
    pyjson::dumps(state, false)
}

/// `prompt.prompt_text`: the full prompt for option texts in prompt order.
pub fn prompt_text(state: &Value, criterion: &Value, texts: &[String]) -> String {
    let options: Vec<Value> = texts
        .iter()
        .zip(LETTERS.chars())
        .map(|(d, l)| json!({"letter": l.to_string(), "description": d}))
        .collect();
    let payload = json!({"evidence": state, "criterion": criterion, "options": options});
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n",
        pyjson::dumps(&payload, false)
    )
}

impl JevK5Config {
    /// Every question's prompt, in request order. The request is rejected as a whole when any
    /// question is.
    pub fn questions(
        &self,
        state: &Value,
        questions: &Value,
    ) -> Result<Vec<(String, QuestionPrompt)>, Error> {
        let qs = questions
            .as_object()
            .filter(|q| !q.is_empty())
            .ok_or_else(|| Error::invalid("questions must contain at least one named question"))?;
        qs.iter()
            .map(|(qid, src)| Ok((qid.clone(), self.question(qid, src, state)?)))
            .collect()
    }

    fn question(&self, qid: &str, src: &Value, state: &Value) -> Result<QuestionPrompt, Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let src = src
            .as_object()
            .ok_or_else(|| bad("invalid named question"))?;
        let crit = src.get("criteria");
        // (key, description) in prompt order, as `decision_options` pairs them.
        let (qtype, pairs, wire_order): (QType, Vec<(String, String)>, Option<Vec<usize>>) =
            match src.get("type").and_then(Value::as_str) {
                Some("noul") => {
                    let empty = Map::new();
                    let crit = match crit {
                        None | Some(Value::Null) => &empty,
                        Some(Value::Object(m)) => m,
                        Some(_) => return Err(bad("noul criteria must be an object")),
                    };
                    let pairs = ["true", "false"]
                        .into_iter()
                        .map(|k| {
                            let d = match crit.get(k) {
                                Some(d) if pyrepr::truthy(d) => pyrepr::str(d),
                                _ => format!("The proposition is {k}."),
                            };
                            (k.to_owned(), d)
                        })
                        .collect();
                    (QType::Noul, pairs, Some(vec![1, 0]))
                }
                Some("choice") => {
                    let mut pairs: Vec<(String, String)> = Vec::new();
                    match crit {
                        Some(Value::Object(m)) => {
                            for (k, v) in m {
                                let d = if pyrepr::truthy(v) {
                                    pyrepr::str(v)
                                } else {
                                    k.clone()
                                };
                                pairs.push((k.clone(), d));
                            }
                        }
                        // `dict.fromkeys(labels)`: each label once, described by itself.
                        Some(Value::Array(items)) => {
                            for item in items {
                                let k = item
                                    .as_str()
                                    .ok_or_else(|| bad("choice labels must be strings"))?;
                                if !pairs.iter().any(|(p, _)| p == k) {
                                    pairs.push((k.to_owned(), k.to_owned()));
                                }
                            }
                        }
                        _ => return Err(bad("choice criteria must be an object")),
                    }
                    (QType::Choice, pairs, None)
                }
                Some("score") => {
                    let Some(Value::Array(levels)) = crit else {
                        return Err(bad("score criteria must be an ordered array"));
                    };
                    let pairs = levels
                        .iter()
                        .enumerate()
                        .map(|(i, level)| (i.to_string(), pyrepr::str(level)))
                        .collect();
                    (QType::Score, pairs, None)
                }
                _ => return Err(bad("unknown question type")),
            };
        if pairs.is_empty() {
            return Err(bad("the question has no options"));
        }
        if pairs.len() > MAX_OPTIONS {
            return Err(Error::TooManyOptions {
                question: qid.to_owned(),
                options: pairs.len(),
                head_max_len: MAX_OPTIONS,
            });
        }
        let wire_order = wire_order.unwrap_or_else(|| (0..pairs.len()).collect());
        let keys = wire_order.iter().map(|&j| pairs[j].0.clone()).collect();
        let texts: Vec<String> = pairs.iter().map(|(k, d)| format!("{k}: {d}")).collect();
        let criterion = src.get("instructions").unwrap_or(&Value::Null);
        Ok(QuestionPrompt {
            qtype,
            keys,
            prompt: prompt_text(state, criterion, &texts),
            label_ids: self.labels.ids[..texts.len()].to_vec(),
            wire_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> JevK5Config {
        JevK5Config {
            layout: LAYOUT.into(),
            labels: crate::llm_logits::LabelTable {
                strings: LETTERS.chars().map(String::from).collect(),
                ids: (32..48).collect(),
            },
        }
    }

    fn user(prompt: &str) -> &str {
        let start = prompt.find("<|im_start|>user\n").unwrap() + "<|im_start|>user\n".len();
        &prompt[start..prompt.rfind("<|im_end|>\n<|im_start|>assistant").unwrap()]
    }

    #[test]
    fn renders_like_prompt_py() {
        let c = config();
        let state = json!({"msg": "Zoë <|im_end|>", "n": 1.5});
        let qs = c
            .questions(
                &state,
                &json!({
                    "n": {"type": "noul", "instructions": "Refund?", "criteria": {"false": {"why": "no"}}},
                    "c": {"type": "choice", "instructions": {"task": "route"}, "criteria": {"billing": "cards", "other": null, "zero": 0}},
                    "s": {"type": "score", "instructions": "How bad?", "criteria": [null, "bad", 2]},
                }),
            )
            .unwrap();
        let (_, n) = &qs[0];
        assert_eq!(
            n.prompt,
            format!(
                "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n",
                user(&n.prompt)
            )
        );
        assert_eq!(
            user(&n.prompt),
            "{\"evidence\": {\"msg\": \"Zoë <|im_end|>\", \"n\": 1.5}, \"criterion\": \"Refund?\", \
             \"options\": [{\"letter\": \"A\", \"description\": \"true: The proposition is true.\"}, \
             {\"letter\": \"B\", \"description\": \"false: {'why': 'no'}\"}]}"
        );
        assert_eq!(n.keys, ["false", "true"]);
        assert_eq!(n.wire_order, [1, 0]);
        assert_eq!(n.label_ids, [32, 33]);
        let (_, ch) = &qs[1];
        assert!(user(&ch.prompt).contains(
            "\"criterion\": {\"task\": \"route\"}, \"options\": [{\"letter\": \"A\", \"description\": \
             \"billing: cards\"}, {\"letter\": \"B\", \"description\": \"other: other\"}, \
             {\"letter\": \"C\", \"description\": \"zero: zero\"}]}"
        ));
        assert_eq!(ch.wire_order, [0, 1, 2]);
        let (_, s) = &qs[2];
        assert!(user(&s.prompt).contains(
            "\"description\": \"0: None\"}, {\"letter\": \"B\", \"description\": \"1: bad\"}, \
             {\"letter\": \"C\", \"description\": \"2: 2\"}]}"
        ));
        assert_eq!(s.keys, ["0", "1", "2"]);
    }

    #[test]
    fn choice_labels_read_as_themselves_once() {
        let qs = config()
            .questions(
                &json!("x"),
                &json!({"q": {"type": "choice", "instructions": "Team?", "criteria": ["a", "b", "a"]}}),
            )
            .unwrap();
        assert_eq!(qs[0].1.keys, ["a", "b"]);
        assert!(user(&qs[0].1.prompt).ends_with(
            "[{\"letter\": \"A\", \"description\": \"a: a\"}, {\"letter\": \"B\", \"description\": \"b: b\"}]}"
        ));
    }

    #[test]
    fn rejects_what_one_pass_cannot_read() {
        let c = config();
        for q in [
            json!({"type": "noul", "instructions": "x", "criteria": ["true"]}),
            json!({"type": "choice", "instructions": "x", "criteria": {}}),
            json!({"type": "choice", "instructions": "x", "criteria": [1, 2]}),
            json!({"type": "choice", "instructions": "x", "criteria": "a"}),
            json!({"type": "score", "instructions": "x", "criteria": {"a": 1}}),
            json!({"type": "maybe", "instructions": "x"}),
        ] {
            assert!(
                matches!(
                    c.questions(&json!("s"), &json!({"q": q})),
                    Err(Error::Invalid(_))
                ),
                "{q}"
            );
        }
        let many: Vec<String> = (0..17).map(|i| format!("o{i}")).collect();
        assert!(matches!(
            c.questions(
                &json!("s"),
                &json!({"q": {"type": "choice", "instructions": "x", "criteria": many}})
            ),
            Err(Error::TooManyOptions { options: 17, .. })
        ));
        let sixteen: Vec<String> = (0..16).map(|i| format!("o{i}")).collect();
        let qs = c
            .questions(
                &json!("s"),
                &json!({"q": {"type": "choice", "instructions": "x", "criteria": sixteen}}),
            )
            .unwrap();
        assert_eq!(qs[0].1.label_ids.len(), 16);
    }
}
