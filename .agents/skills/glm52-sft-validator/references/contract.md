# GLM-5.2 SFT validation contract

## Input

The file is UTF-8 without BOM. Each non-empty line is one JSON object with exactly these top-level fields:

- `messages` (required, non-empty array)
- `tools` (optional, non-empty array)
- `chat_template_kwargs` (optional object)
- `meta` (optional free-form object; use `meta.id` or `meta.source_id` for report identity)

Unknown training fields are fatal. Put provenance and sampling metadata under `meta`.

Assistant reasoning has two accepted representations:

```json
{"role":"assistant","content":[{"type":"thinking","thinking":"reason"},{"type":"text","text":"answer"}]}
```

```json
{"role":"assistant","reasoning_content":"reason","content":"answer"}
```

Never put both representations on one message. New data should use the first form. Handwritten `<think>` or role/tool protocol markers are rejected because the renderer owns those boundaries.

Tool-call `function.arguments` may be a JSON object (legacy dataset form) or a JSON string that decodes to an object (native Tinker form). Tool definitions may use OpenAI wrapped or Tinker bare form. Names must be unique, every call/reference must be declared, and N calls must be followed immediately by N tool messages in the same order.

Supported `chat_template_kwargs` are `enable_thinking: boolean` and `reasoning_effort: high|max`. The old `clear_thinking` option is intentionally rejected because the new renderer preserves historical reasoning; silently accepting it would change training semantics.

## Layers

| Layer | Gate |
| --- | --- |
| L0 | UTF-8/no-BOM JSONL, no blank lines, valid JSON, no duplicate keys |
| L1 | Exact schema, content parts, reasoning exclusivity, protocol safety, tools and call/response pairing |
| L2 | The actual GLM-5.2 Tinker renderer accepts the normalized record |
| L4 | Token and supervision-weight lengths match; weights are binary |
| L5 | Right truncation uses the configured training limit and leaves at least one supervised token |
| L6 | Duplicate, supervised-ratio, and near-limit dataset warnings |

L3 byte parity against the pinned historical Jinja template is maintained by the repository's separate integration parity tests, not recomputed for every dataset row.

## Exit codes

- `0`: at least one trainable record and no fatal findings.
- `1`: fatal data finding, or zero trainable records.
- `2`: missing/invalid `max_seq_len`, missing renderer dependency, unreadable path/report, or tokenizer load failure.

The JSON report is the machine-readable source of truth. Warnings do not change exit code `0`, but truncation and duplicate warnings should be reviewed before expensive training.
