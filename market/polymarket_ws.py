"""
Helios Weather Lab - Polymarket WebSocket Client (SEARA-Grade)
Real-time market data via WebSocket connection to Polymarket CLOB.
Implements Orderbook Mirroring for ultra-low latency data access.

Connects to: wss://ws-subscriptions-clob.polymarket.com/ws/market
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
import time

import websockets
from .orderbook import LocalOrderBook

logger = logging.getLogger("polymarket_ws")

# CLOB WebSocket URL
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# File logging (optional but enabled by default)
_LOG_DIR = Path(__file__).parent.parent / "logs"
_DEFAULT_LOG_PATH = _LOG_DIR / "polymarket_ws.log"

def _ensure_file_logger():
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = Path(os.environ.get("POLYMARKET_WS_LOG_FILE", str(_DEFAULT_LOG_PATH)))

        # Avoid duplicate handlers on reload
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
                return

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # Never fail just because logging failed
        pass

_ensure_file_logger()

@dataclass
class MarketPrice:
    """Real-time price for a market outcome."""
    token_id: str
    bracket: str
    price: float  # 0-1
    last_trade_time: datetime = field(default_factory=datetime.now)

@dataclass 
class LiveMarketState:
    """Current state of all tracked markets."""
    prices: Dict[str, MarketPrice] = field(default_factory=dict)  # token_id -> MarketPrice
    orderbooks: Dict[str, LocalOrderBook] = field(default_factory=dict) # token_id -> OrderBook
    last_update: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, int] = field(default_factory=lambda: {
        "reconnect_count": 0,
        "resync_count": 0,
        "publisher_lag_ms": 0,
        "last_reconnect_ts": 0
    })

class PolymarketWebSocketClient:
    """
    WebSocket client for real-time Polymarket data with Orderbook Mirroring.
    """
    
    def __init__(self):
        self.ws = None
        self.state = LiveMarketState()
        self.market_info: Dict[str, str] = {}  # token_id -> bracket name
        self._running = False
        self._callbacks: List[Callable] = []
        self._publisher_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._subscribed_assets: set[str] = set()
        self._subscription_initialized: bool = False
        self._debug_ws = os.environ.get("POLYMARKET_WS_DEBUG", "").strip().lower() in {"1", "true", "yes"}
        try:
            self._debug_max_messages = int(os.environ.get("POLYMARKET_WS_DEBUG_MAX", "20"))
        except ValueError:
            self._debug_max_messages = 20
        self._debug_message_count = 0
        
    def add_callback(self, callback: Callable):
        """Add a callback to be called when state updates."""
        self._callbacks.append(callback)

    def _ws_is_open(self) -> bool:
        if self.ws is None:
            return False
        state = getattr(self.ws, "state", None)
        if state is not None:
            name = getattr(state, "name", None)
            if isinstance(name, str):
                return name == "OPEN"
            try:
                from websockets.protocol import State
                return state == State.OPEN
            except Exception:
                pass
        closed = getattr(self.ws, "closed", None)
        if isinstance(closed, bool):
            return not closed
        close_code = getattr(self.ws, "close_code", None)
        if close_code is not None:
            return False
        return True

    def _update_price(self, asset_id: str, price: float) -> None:
        if asset_id not in self.state.prices:
            bracket = self.market_info.get(asset_id, "")
            self.state.prices[asset_id] = MarketPrice(token_id=asset_id, bracket=bracket, price=price)
        else:
            self.state.prices[asset_id].price = price
        self.state.prices[asset_id].last_trade_time = datetime.now()
        
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass
            self.ws = await websockets.connect(
                POLYMARKET_WS_URL,
                ping_interval=30,
                ping_timeout=10,
                open_timeout=20
            )
            self._running = True
            self._subscription_initialized = False
            self._subscribed_assets.clear()
            self.state.metrics["last_reconnect_ts"] = int(time.time())
            self.state.metrics["reconnect_count"] += 1
            logger.info(f"✓ Connected to Polymarket WebSocket: {POLYMARKET_WS_URL}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Polymarket WS: {e}")
            return False
            
    async def subscribe_to_markets(self, market_ids: List[str]):
        """
        Subscribe to book and trades for specific markets.
        """
        if not self.ws:
            logger.error("Not connected to WebSocket")
            return False

        # Initialize orderbooks for new markets
        new_ids = []
        for mid in market_ids:
            if mid not in self._subscribed_assets:
                new_ids.append(mid)
            if mid not in self.state.orderbooks:
                self.state.orderbooks[mid] = LocalOrderBook(mid)

        if not new_ids:
            return True

        if not self._subscription_initialized:
            subscribe_msg = {
                "type": "market",
                "assets_ids": new_ids,
                "custom_feature_enabled": True
            }
            self._subscription_initialized = True
        else:
            subscribe_msg = {
                "operation": "subscribe",
                "assets_ids": new_ids,
                "custom_feature_enabled": True
            }

        if self._debug_ws:
            logger.info("WS subscribe payload: %s", subscribe_msg)
        await self.ws.send(json.dumps(subscribe_msg))
        for mid in new_ids:
            self._subscribed_assets.add(mid)
        logger.info(f"✓ Subscribed to {len(new_ids)} markets (CLOB market channel)")
        return True

    async def _publisher_loop(self):
        """
        Background task: Flushes 'Dirty' Write Buffer to 'Clean' Read Buffer.
        Runs every 100ms (10Hz).
        This decouples the high-speed WS ingestion from the API read path.
        """
        logger.info("Starting Publisher Loop (Double Buffer Commit)...")
        while self._running:
            start_ts = time.time()
            try:
                # Commit all dirty books
                for book in self.state.orderbooks.values():
                    book.commit_snapshot()
                
                # Metric: How long did commit take?
                duration = (time.time() - start_ts) * 1000
                if duration > 50:
                    logger.warning(f"Publisher loop slow: {duration:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error in publisher loop: {e}")
            
            # Sleep remainder of 100ms interval
            # await asyncio.sleep(0.1) 
            # Better: sleep fixed amount, or adaptive? Fixed for distinct ticks is fine.
            await asyncio.sleep(0.1)

    async def _heartbeat_loop(self):
        """
        Polymarket WS expects a text 'PING' keepalive.
        """
        logger.info("Starting WS heartbeat loop (PING)...")
        while self._running:
            try:
                if self._ws_is_open():
                    await self.ws.send("PING")
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def listen(self):
        """
        Listen for incoming messages and update state.
        Blocking call. Handles auto-reconnect.
        """
        # Start Publisher Task
        if not self._publisher_task or self._publisher_task.done():
            self._publisher_task = asyncio.create_task(self._publisher_loop())
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._running = True
        while self._running:
            try:
                if not self._ws_is_open():
                    logger.info("Connecting to WebSocket...")
                    if await self.connect():
                        # Re-subscribe if we have existing markets
                        market_ids = list(self.state.orderbooks.keys())
                        if market_ids:
                            await self.subscribe_to_markets(market_ids)
                    else:
                        await asyncio.sleep(5)
                        continue

                async for message in self.ws:
                    await self._handle_message(message)
                logger.warning("WebSocket stream ended, reconnecting...")
                self.ws = None
                await asyncio.sleep(1)
                    
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self.ws = None
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket listener error: {e}")
                self.ws = None
                await asyncio.sleep(5)
            
    async def _handle_message(self, raw_message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            if self._debug_ws and self._debug_message_count < self._debug_max_messages:
                self._debug_message_count += 1
                sample = raw_message if len(raw_message) <= 1000 else raw_message[:1000] + "..."
                logger.info("WS raw[%d/%d]: %s", self._debug_message_count, self._debug_max_messages, sample)
            
            # Polymarket sends list of messages sometimes? No, usually distinct objs
            if isinstance(data, list):
                for item in data:
                    await self._dispatch_message(item)
            else:
                await self._dispatch_message(data)

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
    async def _dispatch_message(self, data: dict):
         # Message format variations: event_type or type
        msg_type = data.get("event_type") or data.get("type")
        
        if not msg_type:
            return

        if msg_type == "price_change":
            await self._handle_price_change(data)
        elif msg_type == "trade":
            await self._handle_trade(data)
        elif msg_type == "last_trade_price":
            await self._handle_last_trade_price(data)
        elif msg_type == "book":
            # Snapshot or Delta?
            # Polymarket legacy format: "book" + check content keys
            # New format often distinguishes, but let's check content
            # Actually standard CLOB format is simple list of changes unless snapshot
            await self._handle_book_update(data)
        elif msg_type == "best_bid_ask":
            await self._handle_best_bid_ask(data)
        elif msg_type == "new_market":
            if self._debug_ws:
                logger.info("WS new_market keys=%s", list(data.keys()))
        elif msg_type == "subscribed":
            if self._debug_ws:
                logger.info("WS subscribed ack: %s", data)
        elif msg_type == "error":
            logger.warning("WS error: %s", data)
        else:
            if self._debug_ws:
                logger.info("WS unknown msg_type=%s keys=%s", msg_type, list(data.keys()))
            pass

    async def _handle_price_change(self, data: dict):
        """
        Handle 'price_change' event. 
        In Polymarket CLOB 'book' channel, these ARE the deltas.
        """
        ts_raw = data.get("timestamp", "0")
        ts = int(ts_raw) if ts_raw else 0

        # New schema: "price_changes" list
        price_changes = data.get("price_changes")
        if price_changes:
            for change in price_changes:
                asset_id = change.get("asset_id") or data.get("asset_id")
                if asset_id and asset_id in self.state.orderbooks:
                    self.state.orderbooks[asset_id].apply_delta([change], timestamp=ts)
            return

        asset_id = data.get("asset_id")
        if not asset_id or asset_id not in self.state.orderbooks:
            return

        # Legacy format: { "price": "0.50", "size": "100", "side": "BUY", ... }
        self.state.orderbooks[asset_id].apply_delta([data], timestamp=ts)

    async def _handle_trade(self, data: dict):
        """Handle trade event."""
        asset_id = data.get("asset_id")
        price = data.get("price")
        if asset_id and price is not None:
            # Update last trade price
            self._update_price(asset_id, float(price))

    async def _handle_last_trade_price(self, data: dict):
        """Handle last_trade_price event."""
        asset_id = data.get("asset_id")
        price = data.get("price")
        if asset_id and price is not None:
            self._update_price(asset_id, float(price))

    async def _handle_best_bid_ask(self, data: dict):
        """Handle best_bid_ask event (used for lightweight price view)."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return
        best_bid = data.get("best_bid")
        best_ask = data.get("best_ask")
        try:
            bid_val = float(best_bid) if best_bid is not None else None
            ask_val = float(best_ask) if best_ask is not None else None
        except (TypeError, ValueError):
            return
        mid = None
        if bid_val is not None and ask_val is not None:
            mid = (bid_val + ask_val) / 2.0
        elif bid_val is not None:
            mid = bid_val
        elif ask_val is not None:
            mid = ask_val
        if mid is None:
            return
        self._update_price(asset_id, mid)

    async def _handle_book_update(self, data: dict):
        """
        Handle order book update (Snapshot or Delta).
        """
        # Timestamp
        ts_raw = data.get("timestamp", "0")
        ts = int(ts_raw) if ts_raw else 0

        # Check if it is a Snapshot (bids/asks or buys/sells lists)
        # Snapshot structure: { "asset_id": "...", "buys": [], "sells": [], "timestamp": ... }
        if (("bids" in data and "asks" in data) or ("buys" in data and "sells" in data)):
            await self._handle_snapshot(data, ts)
        else:
            # Fallback for deltas labeled as "book"
            if self._debug_ws:
                logger.info("WS book delta keys=%s", list(data.keys()))
            await self._handle_delta(data, ts)

    async def _handle_snapshot(self, data: dict, ts: int):
        asset_id = data.get("asset_id")
        if not asset_id or asset_id not in self.state.orderbooks:
            return

        raw_bids = data.get("bids") or data.get("buys") or []
        raw_asks = data.get("asks") or data.get("sells") or []
        bids = [(x["price"], x["size"]) for x in raw_bids if "price" in x and "size" in x]
        asks = [(x["price"], x["size"]) for x in raw_asks if "price" in x and "size" in x]
        
        self.state.orderbooks[asset_id].apply_snapshot(bids, asks, timestamp=ts)
        self.state.metrics["resync_count"] += 1 

    async def _handle_delta(self, data: dict, ts: int):
        # Delta structure: { "market": "...", "timestamp": "...", "price_changes": [...] } ??
        # Or sometimes just list of changes in root if using specific client lib
        # Let's assume standard CLOB structure as seen in SEARA
        
        # If data has "price_changes" (SEARA format)
        changes = data.get("changes") or data.get("price_changes")
        if not changes:
            return

        for change in changes:
            asset_id = change.get("asset_id")
            if asset_id and asset_id in self.state.orderbooks:
                # Wrap single change in list for our method
                # Change format: { "side": "BUY", "price": "0.55", "size": "0", "asset_id": "..." }
                self.state.orderbooks[asset_id].apply_delta([change], timestamp=ts)

    def get_market_snapshot(self, token_id: str) -> Optional[dict]:
        """Get instant snapshot from memory."""
        if token_id in self.state.orderbooks:
            # Reads from the Clean Buffer (O(1))
            return self.state.orderbooks[token_id].get_snapshot()
        return None

    def get_all_books(self) -> Dict[str, Dict[str, Dict[float, float]]]:
        """
        Get raw bid/ask maps for all tracked orderbooks.

        Used by diagnostic/evidence tooling (not hot path).
        """
        return {
            token_id: {
                "bids": {float(price): size for price, size in book.bids.items()},
                "asks": {float(price): size for price, size in book.asks.items()}
            }
            for token_id, book in self.state.orderbooks.items()
        }

    def get_market_info(self, token_id: str) -> Optional[str]:
        """Get registered market info for a token (station|bracket)."""
        return self.market_info.get(token_id)

    async def disconnect(self):
        """Close WebSocket connection."""
        self._running = False
        if self._publisher_task:
            self._publisher_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self.ws:
            await self.ws.close()
            logger.info("Disconnected from Polymarket WebSocket")
            
    @property
    def is_connected(self) -> bool:
        return self._running and self._ws_is_open()
        
    def set_market_info(self, token_id: str, bracket: str):
        """Register market info for a token."""
        self.market_info[token_id] = bracket

# Global client instance
_ws_client: Optional[PolymarketWebSocketClient] = None

async def get_ws_client() -> PolymarketWebSocketClient:
    """Get or create global WebSocket client (async version)."""
    global _ws_client
    if _ws_client is None:
        _ws_client = PolymarketWebSocketClient()
    return _ws_client

def get_ws_client_sync() -> Optional[PolymarketWebSocketClient]:
    """Get global WebSocket client synchronously (returns None if not yet created)."""
    return _ws_client




