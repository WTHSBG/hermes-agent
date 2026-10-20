# Reasoning replay on compatible endpoints

This fork is based on the deployed v0.18.1 / v2026.7.7 commit
`f9eca7e15f1c2bfe5194aae5aa489af53c0a1a23`. It retains GritPath's deployed
tool call ID normalization.

An endpoint may accept historical `reasoning_content` without requiring an
empty-field placeholder. Enable replay of real reasoning for every model on
your compatible endpoint in `config.yaml`:

```yaml
model:
  provider: custom
  base_url: ${CUSTOM_BASE_URL}
  reasoning_replay: preserve
```

For additional endpoints, including fallback destinations:

```yaml
providers:
  inference:
    api: http://inference.example:35996/v1
    reasoning_replay: preserve
```

The legacy `custom_providers` list accepts the same field. The setting applies
only to the exact base URL (ignoring a trailing slash), across all model names.
An explicit provider setting wins over `model.reasoning_replay`.

- `auto` (default): retain existing upstream provider compatibility rules.
- `preserve`: preserve real reasoning in both request-history reconstruction
  and the final pre-request reconciliation. Existing mandatory padding rules
  remain separate. Missing reasoning is not fabricated.
- `strip`: omit the field for endpoints that reject it.

Destination changes re-resolve the policy. Internal history and session DB
records are not rewritten. Both outer and execution images must contain this
fix; execution image configuration templates must opt in as well. Already
missing historical reasoning cannot be recovered by this patch.
