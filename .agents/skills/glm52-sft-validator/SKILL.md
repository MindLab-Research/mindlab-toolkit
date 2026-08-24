---
name: glm52-sft-validator
description: Strictly validate GLM-5.2 SFT JSONL datasets with the repository's real Tinker renderer before training. Use for dataset validation, training-readiness checks, GLM-5.2 JSONL audits, tool/thinking/mask failures, or requests mentioning 训练前校验 or 严格数据校验. Do not use for unrelated model families or generic JSON linting.
---

# GLM-5.2 SFT Validator

Run a fail-closed training-readiness gate. Structural JSON checks alone are not a passing result: a valid record must also render and produce supervised tokens through the repository's GLM-5.2 Tinker renderer.

## Workflow

1. Identify the JSONL path and the exact `max_seq_len` used by training. Never guess the length limit. If it is unavailable, stop and report the missing configuration.
2. Install the renderer extra when needed, then run from the repository root:

   ```bash
   python -m pip install -e '.[glm52-renderer]'
   mint-validate-glm52-sft \
       --data <dataset.jsonl> \
       --max-seq-len <training-limit> \
       --report <validation-report.json>
   ```

   During repository development, `uv run --extra test python scripts/validate_glm52_sft.py ...` is equivalent.
3. Interpret exit codes strictly:

   - `0`: all records passed; warnings still require review.
   - `1`: fatal data violations; do not start training.
   - `2`: validator environment/configuration failure; this says nothing about data validity.
4. Summarize totals and findings grouped by layer/rule. Give line, record id, JSON path, and corrective action. Do not print full reasoning content or unrelated dataset rows.
5. Do not rewrite, delete, truncate, or deduplicate the dataset unless the user separately authorizes data mutation. Re-run the same command after any authorized repair.

Read [references/contract.md](references/contract.md) when diagnosing a finding, changing validator rules, or converting legacy data.

## Invariants

- Use the same tokenizer/model and sequence limit as training.
- Treat `ThinkingPart` and legacy `reasoning_content` as mutually exclusive representations; prefer `ThinkingPart` for new data.
- Treat `enable_thinking=false` plus any reasoning-bearing assistant message as fatal.
- Require tool calls and immediately following tool responses to pair one-for-one and reference declared tools.
- Never downgrade renderer rejection, missing supervision, mask-length mismatch, or unknown configuration to a warning.
