"""
Alpaca Trading Client for Cerebro

Async wrapper around the Alpaca REST API for:
  - Account info (balance, buying power, portfolio value)
  - Order placement (market, limit, stop, stop-limit, trailing stop)
  - Position tracking (open positions, P&L)
  - Order management (list, cancel)
  - Market data (quotes, bars)

Supports both paper and live trading via ALPACA_PAPER=true/false.
API keys come from environment: ALPACA_API_KEY, ALPACA_SECRET_KEY.
"""

import os
import json
import uuid
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base URLs
# ---------------------------------------------------------------------------
PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


class AlpacaClient:
    """
    Async HTTP client for the Alpaca Trading API.

    All methods return plain dicts (JSON responses from Alpaca).
    Raises aiohttp.ClientResponseError on 4xx/5xx.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.paper = paper if paper is not None else os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        self.trade_base = PAPER_BASE if self.paper else LIVE_BASE
        self.data_base = DATA_BASE
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def configured(self) -> bool:
        """True if API keys are set."""
        return bool(self.api_key and self.secret_key)

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self._headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, url: str, params: Dict = None) -> Any:
        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, url: str, payload: Dict = None) -> Any:
        session = await self._get_session()
        async with session.post(url, json=payload or {}) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _delete(self, url: str) -> Any:
        session = await self._get_session()
        async with session.delete(url) as resp:
            resp.raise_for_status()
            if resp.content_length and resp.content_length > 0:
                return await resp.json()
            return {"status": "ok"}

    # ── Account ────────────────────────────────────────────────────

    async def get_account(self) -> Dict[str, Any]:
        """Get account details: balance, buying power, portfolio value, etc."""
        return await self._get(f"{self.trade_base}/v2/account")

    async def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> Dict[str, Any]:
        """Get portfolio value history."""
        return await self._get(f"{self.trade_base}/v2/account/portfolio/history", {
            "period": period,
            "timeframe": timeframe,
        })

    # ── Orders ─────────────────────────────────────────────────────

    async def submit_order(
        self,
        symbol: str,
        qty: float = None,
        notional: float = None,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float = None,
        stop_price: float = None,
        trail_percent: float = None,
        trail_price: float = None,
    ) -> Dict[str, Any]:
        """
        Submit a new order.

        Args:
            symbol: Ticker symbol (e.g. "AAPL", "BTC/USD")
            qty: Number of shares/units (use qty OR notional, not both)
            notional: Dollar amount to buy (fractional shares)
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit", "trailing_stop"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Required for limit/stop_limit orders
            stop_price: Required for stop/stop_limit orders
            trail_percent: For trailing_stop orders (e.g. 5.0 = 5%)
            trail_price: For trailing_stop orders (dollar amount)

        Returns:
            Order dict with id, status, filled_qty, etc.
        """
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if qty is not None:
            payload["qty"] = str(qty)
        elif notional is not None:
            payload["notional"] = str(notional)
        else:
            raise ValueError("Must provide either qty or notional")

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
        if trail_percent is not None:
            payload["trail_percent"] = str(trail_percent)
        if trail_price is not None:
            payload["trail_price"] = str(trail_price)

        return await self._post(f"{self.trade_base}/v2/orders", payload)

    async def list_orders(self, status: str = "open", limit: int = 50) -> List[Dict]:
        """List orders. status: open, closed, all."""
        return await self._get(f"{self.trade_base}/v2/orders", {
            "status": status,
            "limit": limit,
        })

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get a specific order by ID."""
        return await self._get(f"{self.trade_base}/v2/orders/{order_id}")

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a specific order."""
        return await self._delete(f"{self.trade_base}/v2/orders/{order_id}")

    async def cancel_all_orders(self) -> Any:
        """Cancel all open orders."""
        return await self._delete(f"{self.trade_base}/v2/orders")

    # ── Positions ──────────────────────────────────────────────────

    async def list_positions(self) -> List[Dict]:
        """List all open positions."""
        return await self._get(f"{self.trade_base}/v2/positions")

    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for a specific symbol."""
        return await self._get(f"{self.trade_base}/v2/positions/{symbol}")

    async def close_position(self, symbol: str, qty: float = None, percentage: float = None) -> Dict[str, Any]:
        """Close a position (fully or partially)."""
        params = {}
        if qty is not None:
            params["qty"] = str(qty)
        elif percentage is not None:
            params["percentage"] = str(percentage)
        return await self._delete(f"{self.trade_base}/v2/positions/{symbol}")

    async def close_all_positions(self) -> Any:
        """Close all open positions."""
        return await self._delete(f"{self.trade_base}/v2/positions")

    # ── Market Data ────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for a symbol."""
        return await self._get(f"{self.data_base}/v2/stocks/{symbol}/quotes/latest")

    async def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 30) -> Dict[str, Any]:
        """Get historical bars (OHLCV) for a symbol."""
        return await self._get(f"{self.data_base}/v2/stocks/{symbol}/bars", {
            "timeframe": timeframe,
            "limit": limit,
        })

    async def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get full snapshot (quote + trade + bar) for a symbol."""
        return await self._get(f"{self.data_base}/v2/stocks/{symbol}/snapshot")

    # ── Crypto Market Data ─────────────────────────────────────────

    async def get_crypto_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest crypto quote (e.g. BTC/USD)."""
        return await self._get(f"{self.data_base}/v1beta3/crypto/us/latest/quotes", {
            "symbols": symbol,
        })

    async def get_crypto_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 30) -> Dict[str, Any]:
        """Get historical crypto bars."""
        return await self._get(f"{self.data_base}/v1beta3/crypto/us/bars", {
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": limit,
        })

    # ── Watchlists ─────────────────────────────────────────────────

    async def get_clock(self) -> Dict[str, Any]:
        """Get market clock (open/close times, is_open)."""
        return await self._get(f"{self.trade_base}/v2/clock")

    async def get_calendar(self, start: str = None, end: str = None) -> List[Dict]:
        """Get market calendar."""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._get(f"{self.trade_base}/v2/calendar", params)

    # ── Health ─────────────────────────────────────────────────────

    async def check_health(self) -> Dict[str, Any]:
        """
        Check if Alpaca is reachable and keys are valid.
        Returns a summary dict.
        """
        if not self.configured:
            return {
                "available": False,
                "reason": "API keys not configured",
                "paper": self.paper,
            }
        try:
            account = await self.get_account()
            return {
                "available": True,
                "paper": self.paper,
                "account_id": account.get("id", ""),
                "status": account.get("status", ""),
                "buying_power": account.get("buying_power", "0"),
                "portfolio_value": account.get("portfolio_value", "0"),
                "cash": account.get("cash", "0"),
            }
        except aiohttp.ClientResponseError as e:
            return {
                "available": False,
                "reason": f"API error: {e.status} {e.message}",
                "paper": self.paper,
            }
        except Exception as e:
            return {
                "available": False,
                "reason": str(e),
                "paper": self.paper,
            }


# ── Trade Log (persisted to AI_MEMORY) ─────────────────────────────────────

TRADE_LOG_PATH = os.path.join(
    os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")),
    "cerebro", "trade_log.json"
)


def _load_trade_log() -> List[Dict]:
    try:
        if os.path.exists(TRADE_LOG_PATH):
            with open(TRADE_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_trade_log(log: List[Dict]):
    try:
        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
        with open(TRADE_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save trade log: %s", e)


def log_trade(order: Dict[str, Any], source: str = "manual") -> Dict:
    """Log an executed trade for history and XP tracking."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order.get("id", ""),
        "symbol": order.get("symbol", ""),
        "side": order.get("side", ""),
        "qty": order.get("qty", ""),
        "notional": order.get("notional", ""),
        "type": order.get("type", ""),
        "status": order.get("status", ""),
        "filled_avg_price": order.get("filled_avg_price", ""),
        "source": source,
    }
    log = _load_trade_log()
    log.append(entry)
    # Keep last 1000 trades
    if len(log) > 1000:
        log = log[-1000:]
    _save_trade_log(log)

    # Also log to unified wallet
    side = order.get("side", "")
    symbol = order.get("symbol", "")
    qty = order.get("qty", "")
    notional = order.get("notional", "")
    desc = f"{side.upper()} {qty or notional} {symbol}"
    log_wallet_entry(
        category="trade",
        description=desc,
        pnl=0.0,
        symbol=symbol,
        side=side,
        qty=str(qty),
        notional=str(notional),
        source="alpaca",
        metadata={"order_id": order.get("id", ""), "type": order.get("type", ""), "status": order.get("status", "")},
    )

    return entry


# ── Wallet Log (unified financial activity tracker) ───────────────────────

WALLET_LOG_PATH = os.path.join(
    os.environ.get("AI_MEMORY_PATH", os.path.expanduser("~/.cerebro/data")),
    "cerebro", "wallet_log.json"
)


def _load_wallet_log() -> List[Dict]:
    try:
        if os.path.exists(WALLET_LOG_PATH):
            with open(WALLET_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_wallet_log(log: List[Dict]):
    try:
        os.makedirs(os.path.dirname(WALLET_LOG_PATH), exist_ok=True)
        with open(WALLET_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save wallet log: %s", e)


def log_wallet_entry(
    category: str = "other",
    description: str = "",
    pnl: float = 0.0,
    symbol: str = "",
    side: str = "",
    qty: str = "",
    notional: str = "",
    source: str = "manual",
    metadata: Optional[Dict] = None,
) -> Dict:
    """Log a financial activity entry to the unified wallet."""
    entry = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "description": description,
        "pnl": pnl,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "notional": notional,
        "source": source,
        "metadata": metadata or {},
    }
    log = _load_wallet_log()
    log.append(entry)
    if len(log) > 2000:
        log = log[-2000:]
    _save_wallet_log(log)
    return entry


# ── Singleton ──────────────────────────────────────────────────────────────

_client_instance: Optional[AlpacaClient] = None


def get_alpaca_client(
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: Optional[bool] = None,
) -> AlpacaClient:
    """Get or create the Alpaca client singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = AlpacaClient(api_key, secret_key, paper)
    return _client_instance
