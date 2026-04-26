# nuke.fm Trader Bot

LLM forecast trader for nuke.fm scalar Bags markets.

## Run

Set secrets in the environment:

```bash
export OPENROUTER_API_KEY=...
export NUKEFM_BOT_API_KEY=...
uv run python -m nukefm_trader_bot --config config.json
```

Omit `--config` to use the default `config.json` in the current working directory.

The bot reads the public Bags token board, asks OpenRouter `moonshotai/kimi-k2.6` for a cited forecast with web search enabled, publishes the token rationale through the private nuke.fm API, converts that forecast into the market's scalar LONG target, and trades inside configured risk caps.

## Tests

```bash
uv run pytest
```
