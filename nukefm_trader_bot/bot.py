from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Protocol
from urllib.parse import quote

from loguru import logger
import requests


USDC_ATOMIC = Decimal("1000000")
ONE = Decimal("1")
ZERO = Decimal("0")

HTTP_TRANSIENT_STATUSES = frozenset({429, 502, 503, 504})
HTTP_MAX_ATTEMPTS = 3


def _is_transient_request_error(error: BaseException) -> bool:
    if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(error, requests.exceptions.ChunkedEncodingError):
        return True
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        return error.response.status_code in HTTP_TRANSIENT_STATUSES
    return False


FORECAST_SYSTEM_PROMPT = (
    "Forecast the Bags token's USD price at the listed market expiry. "
    "The nuke.fm/Bags token context in the user message is the canonical price, market-cap, mint, and unit source. "
    "Use web search only for catalysts or corroborating context. "
    "Never replace the canonical Bags reference price or market cap with an external page unless that page verifies the exact same Solana mint. "
    "Ticker/name matches are not enough because many meme tokens share symbols. "
    "If external data conflicts with the canonical Bags context or cannot be matched to the mint, ignore that external price data and mention the mismatch briefly. "
    "Return only JSON with keys forecast_price_usd, confidence, rationale, sources. "
    "forecast_price_usd and confidence must be JSON numbers. "
    "forecast_price_usd must use the same USD price units as reference_price_usd and chart.underlying_price_usd. "
    "confidence must be between 0 and 1; never use words like low, medium, or high. "
    "rationale must be 2-4 concise sentences."
)


@dataclass(frozen=True)
class BotConfig:
    nukefm_api_base_url: str
    openrouter_model: str
    max_trade_usdc: Decimal
    max_daily_spend_usdc: Decimal
    max_per_market_exposure_usdc: Decimal
    min_forecast_edge: Decimal
    forecast_ttl_seconds: int
    max_search_results: int
    log_path: Path
    state_path: Path

    @classmethod
    def load(cls, path: Path) -> "BotConfig":
        data = json.loads(path.read_text())
        return cls(
            nukefm_api_base_url=data["nukefm_api_base_url"].rstrip("/"),
            openrouter_model=data.get("openrouter_model", "moonshotai/kimi-k2.6"),
            max_trade_usdc=Decimal(str(data["max_trade_usdc"])),
            max_daily_spend_usdc=Decimal(str(data["max_daily_spend_usdc"])),
            max_per_market_exposure_usdc=Decimal(str(data["max_per_market_exposure_usdc"])),
            min_forecast_edge=Decimal(str(data["min_forecast_edge"])),
            forecast_ttl_seconds=int(data["forecast_ttl_seconds"]),
            max_search_results=int(data["max_search_results"]),
            log_path=Path(data["log_path"]),
            state_path=Path(data["state_path"]),
        )


@dataclass(frozen=True)
class Forecast:
    forecast_price_usd: Decimal
    confidence: Decimal
    rationale: str
    sources: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True)
class TradeDecision:
    market_id: int
    mint: str
    symbol: str
    outcome: str | None
    amount_usdc: Decimal
    reason: str
    target_long_price: Decimal | None = None
    current_long_price: Decimal | None = None


class MarketApi(Protocol):
    def list_tokens(self) -> list[dict]:
        ...

    def get_account(self) -> dict:
        ...

    def quote_trade(self, *, market_id: int, outcome: str, amount_usdc: Decimal) -> dict:
        ...

    def execute_trade(self, *, market_id: int, outcome: str, amount_usdc: Decimal) -> dict:
        ...

    def submit_rationale(self, *, mint: str, forecast: Forecast) -> dict:
        ...


class Forecaster(Protocol):
    def forecast(self, token: dict) -> Forecast:
        ...


class NukefmApiClient:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._session = requests.Session()

    def list_tokens(self) -> list[dict]:
        return self._request("GET", "/v1/public/tokens")["tokens"]

    def get_account(self) -> dict:
        return self._request("GET", "/v1/private/account", private=True)

    def quote_trade(self, *, market_id: int, outcome: str, amount_usdc: Decimal) -> dict:
        return self._request(
            "POST",
            "/v1/private/trades/quote",
            private=True,
            json_body={
                "market_id": market_id,
                "outcome": outcome,
                "side": "buy",
                "amount_usdc": format_usdc(amount_usdc),
            },
        )

    def execute_trade(self, *, market_id: int, outcome: str, amount_usdc: Decimal) -> dict:
        return self._request(
            "POST",
            "/v1/private/trades",
            private=True,
            json_body={
                "market_id": market_id,
                "outcome": outcome,
                "side": "buy",
                "amount_usdc": format_usdc(amount_usdc),
            },
        )

    def submit_rationale(self, *, mint: str, forecast: Forecast) -> dict:
        safe_mint = quote(mint, safe="")
        return self._request(
            "POST",
            f"/v1/private/tokens/{safe_mint}/rationale",
            private=True,
            json_body={
                "forecast_price_usd": str(forecast.forecast_price_usd),
                "confidence": str(forecast.confidence),
                "rationale": forecast.rationale,
                "sources": list(forecast.sources),
            },
        )

    def _request(self, method: str, path: str, *, private: bool = False, json_body: dict | None = None) -> dict:
        headers = {"X-API-Key": self._api_key} if private else None
        url = f"{self._base_url}{path}"
        for attempt in range(HTTP_MAX_ATTEMPTS):
            try:
                response = self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=30,
                )
            except requests.RequestException as error:
                if attempt < HTTP_MAX_ATTEMPTS - 1 and _is_transient_request_error(error):
                    time.sleep(2**attempt)
                    continue
                raise
            if response.status_code in HTTP_TRANSIENT_STATUSES and attempt < HTTP_MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()


class OpenRouterForecaster:
    def __init__(self, *, api_key: str, model: str, max_search_results: int) -> None:
        self._api_key = api_key
        self._model = model
        self._max_search_results = max_search_results
        self._session = requests.Session()

    def forecast(self, token: dict) -> Forecast:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": FORECAST_SYSTEM_PROMPT,
                },
                {"role": "user", "content": json.dumps(forecast_context(token), sort_keys=True)},
            ],
            "tools": [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "max_results": self._max_search_results,
                        "max_total_results": self._max_search_results,
                        "search_context_size": "medium",
                    },
                }
            ],
            "response_format": forecast_response_format(),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(HTTP_MAX_ATTEMPTS):
            try:
                response = self._session.post(url, headers=headers, json=payload, timeout=120)
            except requests.RequestException as error:
                if attempt < HTTP_MAX_ATTEMPTS - 1 and _is_transient_request_error(error):
                    time.sleep(2**attempt)
                    continue
                raise
            if response.status_code in HTTP_TRANSIENT_STATUSES and attempt < HTTP_MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_forecast(content, created_at=utc_now())


class BotStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id INTEGER NOT NULL,
                    mint TEXT NOT NULL,
                    forecast_price_usd TEXT NOT NULL,
                    target_long_price TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id INTEGER NOT NULL,
                    mint TEXT NOT NULL,
                    outcome TEXT,
                    amount_usdc TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def latest_forecast(self, market_id: int, *, ttl_seconds: int, now: datetime) -> Forecast | None:
        cutoff = now - timedelta(seconds=ttl_seconds)
        with sqlite3.connect(self._path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT *
                FROM forecasts
                WHERE market_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [market_id, cutoff.isoformat()],
            ).fetchone()
        if row is None:
            return None
        return Forecast(
            forecast_price_usd=Decimal(row["forecast_price_usd"]),
            confidence=Decimal(row["confidence"]),
            rationale=row["rationale"],
            sources=tuple(json.loads(row["sources_json"])),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def record_forecast(self, *, market_id: int, mint: str, forecast: Forecast, target_long_price: Decimal) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                INSERT INTO forecasts (
                    market_id,
                    mint,
                    forecast_price_usd,
                    target_long_price,
                    confidence,
                    rationale,
                    sources_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    market_id,
                    mint,
                    str(forecast.forecast_price_usd),
                    str(target_long_price),
                    str(forecast.confidence),
                    forecast.rationale,
                    json.dumps(list(forecast.sources)),
                    forecast.created_at.isoformat(),
                ],
            )

    def record_trade_attempt(self, decision: TradeDecision, *, state: str, now: datetime) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                INSERT INTO trade_attempts (
                    market_id,
                    mint,
                    outcome,
                    amount_usdc,
                    state,
                    reason,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    decision.market_id,
                    decision.mint,
                    decision.outcome,
                    str(decision.amount_usdc),
                    state,
                    decision.reason,
                    now.isoformat(),
                ],
            )

    def spent_today(self, *, now: datetime) -> Decimal:
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with sqlite3.connect(self._path) as connection:
            rows = connection.execute(
                """
                SELECT amount_usdc
                FROM trade_attempts
                WHERE state = 'executed' AND created_at >= ?
                """,
                [day_start.isoformat()],
            ).fetchall()
        return sum((Decimal(row[0]) for row in rows), ZERO)


class TraderBot:
    def __init__(self, *, config: BotConfig, market_api: MarketApi, forecaster: Forecaster, store: BotStore) -> None:
        self._config = config
        self._market_api = market_api
        self._forecaster = forecaster
        self._store = store

    def run_once(self) -> list[TradeDecision]:
        now = utc_now()
        self._store.initialize()
        tokens = self._market_api.list_tokens()
        account = self._market_api.get_account()
        decisions: list[TradeDecision] = []

        for token in tokens:
            market = token["current_market"]
            if market["state"] != "open" or market["total_liquidity_usdc"] is None:
                continue

            decision = self._decision_for_token(token, account=account, now=now)
            decisions.append(decision)
            if decision.outcome is None:
                self._store.record_trade_attempt(decision, state="skipped", now=now)
                continue

            try:
                self._market_api.execute_trade(
                    market_id=decision.market_id,
                    outcome=decision.outcome,
                    amount_usdc=decision.amount_usdc,
                )
            except Exception as error:
                failed = TradeDecision(
                    market_id=decision.market_id,
                    mint=decision.mint,
                    symbol=decision.symbol,
                    outcome=decision.outcome,
                    amount_usdc=decision.amount_usdc,
                    reason=f"Trade failed: {error}",
                    target_long_price=decision.target_long_price,
                    current_long_price=decision.current_long_price,
                )
                self._store.record_trade_attempt(failed, state="failed", now=now)
                decisions[-1] = failed
                logger.warning("Trade failed for {} market {}: {}", decision.symbol, decision.market_id, error)
                continue

            self._store.record_trade_attempt(decision, state="executed", now=now)
            logger.info(
                "Executed {} buy of {} USDC for {} market {}.",
                decision.outcome,
                format_usdc(decision.amount_usdc),
                decision.symbol,
                decision.market_id,
            )

        return decisions

    def _decision_for_token(self, token: dict, *, account: dict, now: datetime) -> TradeDecision:
        market = token["current_market"]
        market_id = int(market["id"])
        base = {
            "market_id": market_id,
            "mint": token["mint"],
            "symbol": token["symbol"],
        }

        try:
            current_long_price = Decimal(str(market["long_price_usd"]))
            forecast = self._forecast(token, now)
        except Exception as error:
            return TradeDecision(**base, outcome=None, amount_usdc=ZERO, reason=f"No usable forecast: {error}")

        try:
            self._market_api.submit_rationale(mint=token["mint"], forecast=forecast)
        except Exception as error:
            return TradeDecision(**base, outcome=None, amount_usdc=ZERO, reason=f"Rationale submission failed: {error}")

        target_long_price = self._target_long_price(forecast, market)
        edge = target_long_price - current_long_price
        if abs(edge) < self._config.min_forecast_edge:
            return TradeDecision(
                **base,
                outcome=None,
                amount_usdc=ZERO,
                reason="Forecast edge below minimum.",
                target_long_price=target_long_price,
                current_long_price=current_long_price,
            )

        outcome = "long" if edge > 0 else "short"
        cap = self._trade_cap_usdc(market_id=market_id, account=account, now=now)
        if cap <= ZERO:
            return TradeDecision(
                **base,
                outcome=None,
                amount_usdc=ZERO,
                reason="Risk caps or account balance leave no tradable amount.",
                target_long_price=target_long_price,
                current_long_price=current_long_price,
            )

        amount = self._sized_trade_amount(
            market_id=market_id,
            outcome=outcome,
            target_long_price=target_long_price,
            cap_usdc=cap,
        )
        if amount <= ZERO:
            return TradeDecision(
                **base,
                outcome=None,
                amount_usdc=ZERO,
                reason="Smallest quote would overshoot forecast target.",
                target_long_price=target_long_price,
                current_long_price=current_long_price,
            )

        return TradeDecision(
            **base,
            outcome=outcome,
            amount_usdc=amount,
            reason="Forecast edge met.",
            target_long_price=target_long_price,
            current_long_price=current_long_price,
        )

    def _forecast(self, token: dict, now: datetime) -> Forecast:
        market_id = int(token["current_market"]["id"])
        cached = self._store.latest_forecast(market_id, ttl_seconds=self._config.forecast_ttl_seconds, now=now)
        if cached is not None:
            return cached

        forecast = self._forecaster.forecast(token)
        target_long_price = self._target_long_price(forecast, token["current_market"])
        self._store.record_forecast(
            market_id=market_id,
            mint=token["mint"],
            forecast=forecast,
            target_long_price=target_long_price,
        )
        return forecast

    @staticmethod
    def _target_long_price(forecast: Forecast, market: dict) -> Decimal:
        min_price = Decimal(str(market["min_price_usd"]))
        max_price = Decimal(str(market["max_price_usd"]))
        forecast_price = forecast.forecast_price_usd
        if min_price <= ZERO or max_price <= min_price or forecast_price <= ZERO:
            raise ValueError("Invalid scalar price inputs.")

        clamped_price = min(max_price, max(min_price, forecast_price))
        return (clamped_price.ln() - min_price.ln()) / (max_price.ln() - min_price.ln())

    def _trade_cap_usdc(self, *, market_id: int, account: dict, now: datetime) -> Decimal:
        account_balance = Decimal(str(account["account_balance_usdc"]))
        daily_remaining = self._config.max_daily_spend_usdc - self._store.spent_today(now=now)
        market_exposure = self._market_exposure_usdc(account, market_id)
        exposure_remaining = self._config.max_per_market_exposure_usdc - market_exposure
        return min(self._config.max_trade_usdc, account_balance, daily_remaining, exposure_remaining)

    @staticmethod
    def _market_exposure_usdc(account: dict, market_id: int) -> Decimal:
        positions = account.get("open_positions") or []
        for position in positions:
            if int(position["market_id"]) == market_id:
                return Decimal(str(position["marked_value_usdc"]))
        return ZERO

    def _sized_trade_amount(
        self,
        *,
        market_id: int,
        outcome: str,
        target_long_price: Decimal,
        cap_usdc: Decimal,
    ) -> Decimal:
        low = 1
        high = usdc_to_atomic(cap_usdc)
        best = 0
        while low <= high:
            candidate = (low + high) // 2
            quote = self._market_api.quote_trade(
                market_id=market_id,
                outcome=outcome,
                amount_usdc=atomic_to_usdc(candidate),
            )
            after_long_price = Decimal(str(quote["after_long_price_usd"]))
            if (outcome == "long" and after_long_price <= target_long_price) or (
                outcome == "short" and after_long_price >= target_long_price
            ):
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1
        return atomic_to_usdc(best)


def forecast_context(token: dict) -> dict:
    market = token["current_market"]
    return {
        "token": {
            "symbol": token["symbol"],
            "name": token["name"],
            "mint": token["mint"],
            "bags_token_url": token.get("bags_token_url"),
        },
        "market": {
            "id": market["id"],
            "state": market["state"],
            "expiry": market["expiry"],
            "question": market["question"],
            "long_price_usd": market.get("long_price_usd"),
            "short_price_usd": market.get("short_price_usd"),
            "implied_price_usd": market.get("implied_price_usd"),
            "reference_price_usd": market.get("reference_price_usd"),
            "min_price_usd": market.get("min_price_usd"),
            "max_price_usd": market.get("max_price_usd"),
            "total_liquidity_usdc": market.get("total_liquidity_usdc"),
            "pm_volume_24h_usdc": market.get("pm_volume_24h_usdc"),
            "underlying_volume_24h_usd": market.get("underlying_volume_24h_usd"),
            "underlying_market_cap_usd": market.get("underlying_market_cap_usd"),
            "market_cap_kind": market.get("market_cap_kind"),
        },
        "forecasting_rules": {
            "canonical_price_field": "market.reference_price_usd",
            "forecast_price_units": "same USD token-price units as market.reference_price_usd and chart.underlying_price_usd",
            "external_source_rule": "Only use external price or market-cap data if it verifies the exact token.mint; ticker/name matches alone are invalid.",
        },
        "chart": token.get("current_market_chart", {}).get("points", [])[-24:],
    }


def forecast_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "nukefm_price_forecast",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "forecast_price_usd": {
                        "type": "number",
                        "description": "Positive USD price forecast at market expiry.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence from 0 to 1.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short rationale for the forecast.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["forecast_price_usd", "confidence", "rationale", "sources"],
            },
        },
    }


def _strip_markdown_code_fence(text: str) -> str:
    """Extract JSON from an LLM reply; handles optional fences and missing closing markers."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    first_break = t.find("\n")
    if first_break == -1:
        return t[3:].strip()
    body = t[first_break + 1 :]
    trimmed = body.rstrip()
    if trimmed.endswith("```"):
        end_fence = trimmed.rfind("```")
        body = trimmed[:end_fence]
    return body.strip()


def parse_forecast(content: str | None, *, created_at: datetime) -> Forecast:
    if content is None:
        raise ValueError("OpenRouter returned no forecast content.")

    text = _strip_markdown_code_fence(content)
    data = json.loads(text)
    forecast_price = Decimal(str(data["forecast_price_usd"]))
    confidence = Decimal(str(data["confidence"]))
    if forecast_price <= ZERO:
        raise ValueError("forecast_price_usd must be positive.")
    if confidence < ZERO or confidence > ONE:
        raise ValueError("confidence must be inside [0, 1].")
    sources = tuple(str(source) for source in data.get("sources", []))
    return Forecast(
        forecast_price_usd=forecast_price,
        confidence=confidence,
        rationale=str(data["rationale"]),
        sources=sources,
        created_at=created_at,
    )


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_path, rotation="10 MB", retention=5)


def build_bot(config_path: Path) -> TraderBot:
    config = BotConfig.load(config_path)
    configure_logging(config.log_path)
    openrouter_api_key = required_env("OPENROUTER_API_KEY")
    nukefm_api_key = required_env("NUKEFM_BOT_API_KEY")
    return TraderBot(
        config=config,
        market_api=NukefmApiClient(base_url=config.nukefm_api_base_url, api_key=nukefm_api_key),
        forecaster=OpenRouterForecaster(
            api_key=openrouter_api_key,
            model=config.openrouter_model,
            max_search_results=config.max_search_results,
        ),
        store=BotStore(config.state_path),
    )


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def utc_now() -> datetime:
    return datetime.now(UTC)


def usdc_to_atomic(amount_usdc: Decimal) -> int:
    return int((amount_usdc * USDC_ATOMIC).to_integral_value())


def atomic_to_usdc(amount_atomic: int) -> Decimal:
    return Decimal(amount_atomic) / USDC_ATOMIC


def format_usdc(amount_usdc: Decimal) -> str:
    text = format(amount_usdc.quantize(Decimal("0.000001")), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
