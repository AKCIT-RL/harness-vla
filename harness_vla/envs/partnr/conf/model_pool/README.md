# PARTNR beta model pool

This Hydra group is the single source for the model, thinking flags, and
context window used by the PARTNR beta experiment. Select a cell with
`model_pool=<profile>`.

Available cells:

- `gemma4_12b_off`
- `gemma4_12b_on`
- `gemma3_4b_off`
- `gemma3_1b_off`

Thinking-on cells for gemma3:4b and gemma3:1b intentionally do
not exist because those installed models do not support thinking mode. Every
off cell explicitly sets `enable_thinking: false` and
`disable_thinking: true`; the gemma4 on cell sets the inverse.

Both planner paths use a uniform 1024-token output budget. Ollama counts
thinking and final content against the same `num_predict` limit, so the prior
128/250-token limits could end a thinking response before it emitted an action.
These pool runs therefore form a new protocol. Qwen is excluded from the active
experiment; its earlier runs are retained only as historical artifacts.