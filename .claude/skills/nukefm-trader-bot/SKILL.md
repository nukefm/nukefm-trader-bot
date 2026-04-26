---
name: nukefm-trader-bot
description: Use when maintaining or operating the nuke.fm trader bot that forecasts Bags token expiry prices with OpenRouter Kimi K2.6 web search and trades scalar LONG/SHORT markets through the nuke.fm API.
---

# nuke.fm Trader Bot

Use this skill for changes to `bots/trader` or for operating the LLM trading loop.

## Model

- The bot trades nuke.fm scalar LONG/SHORT markets for Bags tokens.
- It does not trade the underlying Bags token.
- It asks OpenRouter `moonshotai/kimi-k2.6` for a cited USD price forecast at the market expiry.
- Web search must use the `openrouter:web_search` server tool.
- Forecast output must be strict JSON with `forecast_price_usd`, `confidence`, `rationale`, and `sources`.

## Trading Rules

- Trade only markets with `state == "open"` and non-null prediction liquidity.
- Convert forecast price to LONG target with the same log-space scalar range used by nuke.fm.
- Buy LONG when the target LONG price exceeds the current LONG price by the configured edge.
- Buy SHORT when the target LONG price is below the current LONG price by the configured edge.
- Never exceed configured account, daily spend, per-market exposure, or per-trade caps.
- Missing or invalid forecasts are no-trade outcomes, not fallbacks.

## Secrets

- Read only `OPENROUTER_API_KEY` and `NUKEFM_BOT_API_KEY` from environment variables.
- Do not log secret values or request headers.
- Do not load private keys in this bot.

## Validation

- Run `uv run --project bots/trader pytest` after code changes.
- Run the root app tests if API fields or scalar market semantics change.
