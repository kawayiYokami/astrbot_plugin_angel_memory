# 提示词拆分说明

本文件不再作为反思提示词的真相源。

当前反思提示词已经拆分到：

- `llm_memory/prompts/sections/00_intro.md`
- `llm_memory/prompts/sections/10_output_schema.md`
- `llm_memory/prompts/sections/20_actions.md`
- `llm_memory/prompts/sections/30_memory_fields.md`
- `llm_memory/prompts/sections/40_user_profiles.md`
- `llm_memory/prompts/sections/50_generation_rules.md`
- `llm_memory/prompts/sections/60_examples.md`
- `llm_memory/prompts/sections/70_checklist.md`

运行时由 `llm_memory/prompts/prompt_assembler.py` 按固定顺序组装。

不要再直接手改本文件，否则不会影响真实运行提示词。
