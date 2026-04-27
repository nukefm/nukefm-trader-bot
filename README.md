# nuke.fm Trader Bot

LLM forecast trader for nuke.fm scalar bags.fm token markets.

## Run

Set secrets in the environment:

```bash
export OPENROUTER_API_KEY=...
export NUKEFM_BOT_API_KEY=...
uv run --project bots/trader python -m nukefm_trader_bot --config bots/trader/config.json
```

The bot reads the public Bags token board, asks OpenRouter `moonshotai/kimi-k2.6` for a cited market-end-date price forecast with web search enabled, publishes the token rationale through the private nuke.fm API, converts that forecast into the market's scalar LONG target, and trades inside configured risk caps.

## Tests

```bash
cd bots/trader
uv run pytest
```
