Fork of https://github.com/wmt-conference/wmt-collect-translations (2024 version)

Last upstream commit: https://github.com/wmt-conference/wmt-collect-translations/commit/704b3825730f93a3ee3a0fda44af9414937b6d5a

## Preprocessing the src data
`python -m scripts.export_data_to_wmt_codebase` (in the parent dir)

## Collecting translations
Define `OPENAI_API_KEY` in `.env` file.

For Gemini and Llama (which we also queried via the Google Vertex API), we used a LiteLLM proxy. For this, we defined a `LITELLM_PROXY_API_KEY` and the `LITELLM_PROXY_API_BASE`.

```bash
model="Gemini-2.5-Flash"  # Llama-3.3-70b  # GPT-4o
for variety in rm_rumgr rm_sursilv rm_sutsilv rm_surmiran rm_puter rm_vallader; do
  python main_romansh.py --system=${model} --lp=${variety}-de --no_testsuites
  python main_romansh.py --system=${model} --lp=de-${variety} --no_testsuites
done
```
