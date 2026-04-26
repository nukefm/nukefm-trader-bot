from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from nukefm_trader_bot.bot import (
    BotConfig,
    BotStore,
    Forecast,
    OpenRouterForecaster,
    TradeDecision,
    TraderBot,
    forecast_response_format,
    parse_forecast,
)


class FakeMarketApi:
    def __init__(self, *, token: dict, account: dict, current_long_price: Decimal = Decimal("0.50")) -> None:
        self._token = token
        self._account = account
        self._current_long_price = current_long_price
        self.executed_trades: list[dict] = []
        self.quote_calls: list[dict] = []
        self.rationales: list[dict] = []

    def list_tokens(self) -> list[dict]:
        return [self._token]

    def get_account(self) -> dict:
        return self._account

    def quote_trade(self, *, market_id: int, outcome: str, amount_usdc: Decimal) -> dict:
        self.quote_calls.append({"market_id": market_id, "outcome": outcome, "amount_usdc": amount_usdc})
        price_move = amount_usdc * Decimal("0.10")
        after_long_price = (
            self._current_long_price + price_move
            if outcome == "long"
            else self._current_long_price - price_move
        )
        after_long_price = max(Decimal("0.01"), min(Decimal("0.99"), after_long_price))
        return {"after_long_price_usd": str(after_long_price)}

    def execute_trade(self, *, market_id: int, outcome: str, amount_usdc: Decimal) -> dict:
        trade = {"market_id": market_id, "outcome": outcome, "amount_usdc": amount_usdc}
        self.executed_trades.append(trade)
        return trade

    def submit_rationale(self, *, mint: str, forecast: Forecast) -> dict:
        rationale = {"mint": mint, "forecast": forecast}
        self.rationales.append(rationale)
        return rationale


class FakeForecaster:
    def __init__(self, forecast_price_usd: Decimal) -> None:
        self._forecast_price_usd = forecast_price_usd
        self.calls = 0

    def forecast(self, token: dict) -> Forecast:
        self.calls += 1
        return Forecast(
            forecast_price_usd=self._forecast_price_usd,
            confidence=Decimal("0.7"),
            rationale="Forecast from test fixture.",
            sources=("https://example.test/source",),
            created_at=datetime(2026, 4, 26, tzinfo=UTC),
        )


def config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        nukefm_api_base_url="https://nukefm.test",
        openrouter_model="moonshotai/kimi-k2.6",
        max_trade_usdc=Decimal("1"),
        max_daily_spend_usdc=Decimal("5"),
        max_per_market_exposure_usdc=Decimal("2"),
        min_forecast_edge=Decimal("0.01"),
        forecast_ttl_seconds=3600,
        max_search_results=5,
        log_path=tmp_path / "trader.log",
        state_path=tmp_path / "trader.sqlite3",
    )


def token_fixture(*, state: str = "open", liquidity: str | None = "10") -> dict:
    return {
        "mint": "Mint333",
        "symbol": "GAMMA",
        "name": "Gamma",
        "bags_token_url": "https://bags.fm/Mint333",
        "current_market_chart": {"points": []},
        "current_market": {
            "id": 7,
            "state": state,
            "question": "What will GAMMA trade at by 2026-07-14?",
            "expiry": "2026-07-14T12:00:00+00:00",
            "long_price_usd": "0.50",
            "short_price_usd": "0.50",
            "implied_price_usd": "1",
            "min_price_usd": "0.1",
            "max_price_usd": "10",
            "total_liquidity_usdc": liquidity,
            "pm_volume_24h_usdc": "1",
            "underlying_volume_24h_usd": "100",
            "underlying_market_cap_usd": "1000",
        },
    }


def account_fixture(*, balance: str = "10", exposure: str = "0") -> dict:
    return {
        "account_balance_usdc": balance,
        "open_positions": [
            {
                "market_id": 7,
                "marked_value_usdc": exposure,
            }
        ],
    }


def test_parse_forecast_requires_positive_price() -> None:
    with pytest.raises(ValueError, match="forecast_price_usd"):
        parse_forecast(
            '{"forecast_price_usd": "0", "confidence": "0.7", "rationale": "x", "sources": []}',
            created_at=datetime(2026, 4, 26, tzinfo=UTC),
        )


def test_parse_forecast_requires_content() -> None:
    with pytest.raises(ValueError, match="no forecast content"):
        parse_forecast(None, created_at=datetime(2026, 4, 26, tzinfo=UTC))


def test_bot_trades_long_toward_forecast(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "state.sqlite3")
    api = FakeMarketApi(token=token_fixture(), account=account_fixture())
    bot = TraderBot(
        config=config(tmp_path),
        market_api=api,
        forecaster=FakeForecaster(Decimal("2")),
        store=store,
    )

    decisions = bot.run_once()

    assert decisions[0].outcome == "long"
    assert api.executed_trades[0]["outcome"] == "long"
    assert api.rationales[0]["mint"] == "Mint333"
    assert Decimal("0") < api.executed_trades[0]["amount_usdc"] <= Decimal("1")


def test_bot_trades_short_toward_forecast(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "state.sqlite3")
    api = FakeMarketApi(token=token_fixture(), account=account_fixture())
    bot = TraderBot(
        config=config(tmp_path),
        market_api=api,
        forecaster=FakeForecaster(Decimal("0.5")),
        store=store,
    )

    decisions = bot.run_once()

    assert decisions[0].outcome == "short"
    assert api.executed_trades[0]["outcome"] == "short"


def test_bot_skips_when_market_is_not_tradeable(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "state.sqlite3")
    forecaster = FakeForecaster(Decimal("2"))
    api = FakeMarketApi(
        token=token_fixture(state="awaiting_liquidity", liquidity=None),
        account=account_fixture(),
    )
    bot = TraderBot(config=config(tmp_path), market_api=api, forecaster=forecaster, store=store)

    assert bot.run_once() == []
    assert forecaster.calls == 0
    assert api.executed_trades == []
    assert api.rationales == []


def test_bot_skips_when_risk_cap_is_spent(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "state.sqlite3")
    store.initialize()
    store.record_trade_attempt(
        TradeDecision(
            market_id=7,
            mint="Mint333",
            symbol="GAMMA",
            outcome="long",
            amount_usdc=Decimal("5"),
            reason="seed spend",
        ),
        state="executed",
        now=datetime.now(UTC),
    )
    api = FakeMarketApi(token=token_fixture(), account=account_fixture())
    bot = TraderBot(config=config(tmp_path), market_api=api, forecaster=FakeForecaster(Decimal("2")), store=store)

    decisions = bot.run_once()

    assert decisions[0].outcome is None
    assert decisions[0].reason == "Risk caps or account balance leave no tradable amount."
    assert api.executed_trades == []


def test_cached_forecast_survives_new_store_instance(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite3"
    first_store = BotStore(state_path)
    first_store.initialize()
    forecast = Forecast(
        forecast_price_usd=Decimal("2"),
        confidence=Decimal("0.8"),
        rationale="Cached fixture.",
        sources=("https://example.test",),
        created_at=datetime.now(UTC),
    )
    first_store.record_forecast(
        market_id=7,
        mint="Mint333",
        forecast=forecast,
        target_long_price=Decimal("0.65"),
    )

    second_store = BotStore(state_path)
    assert second_store.latest_forecast(7, ttl_seconds=3600, now=datetime.now(UTC)) == forecast


def test_openrouter_forecaster_uses_kimi_with_web_search(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"forecast_price_usd": "2", "confidence": "0.7", '
                                '"rationale": "searched", "sources": ["https://example.test"]}'
                            )
                        }
                    }
                ]
            }

    class FakeSession:
        def __init__(self) -> None:
            self.payload = None

        def post(self, url: str, *, headers: dict, json: dict, timeout: int) -> FakeResponse:
            self.payload = json
            return FakeResponse()

    fake_session = FakeSession()
    monkeypatch.setattr("nukefm_trader_bot.bot.requests.Session", lambda: fake_session)

    forecaster = OpenRouterForecaster(
        api_key="test-key",
        model="moonshotai/kimi-k2.6",
        max_search_results=3,
    )
    forecast = forecaster.forecast(token_fixture())

    assert forecast.forecast_price_usd == Decimal("2")
    assert fake_session.payload["model"] == "moonshotai/kimi-k2.6"
    assert fake_session.payload["tools"] == [
        {
            "type": "openrouter:web_search",
            "parameters": {
                "max_results": 3,
                "max_total_results": 3,
                "search_context_size": "medium",
            },
        }
    ]
    assert fake_session.payload["response_format"] == forecast_response_format()
