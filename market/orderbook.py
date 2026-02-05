"""
Helios Weather Lab - Local OrderBook Mirror
Maintains a local copy of the Polymarket orderbook for a specific token.
Optimized for speed (O(1) lookups) using standard dictionaries.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class OrderBookLevel:
    price: float
    size: float

class LocalOrderBook:
    """
    In-memory OrderBook for a single token.
    Mimics the Rust `LocalOrderBook` logic but in Python.
    Optimized with Double Buffering for lock-free(ish) reads.
    """
    def __init__(self, token_id: str):
        self.token_id = token_id
        # Bids and Asks are dicts: {price_str: size_float}
        # We use string keys for prices to avoid float precision issues during lookups
        # WRITE BUFFER (Dirty)
        self.bids: Dict[str, float] = {}
        self.asks: Dict[str, float] = {}
        
        # Metadata
        self.last_update_ts: float = 0.0  # Local ingestion time
        self.last_provider_ts: int = 0    # Provider timestamp (ms)
        self.last_sequence_id: int = 0
        
        # READ BUFFER (Clean/Cached)
        self._cached_snapshot: Dict = {}
        self._needs_commit: bool = False
        self._commit_counter: int = 0

    def apply_snapshot(self, bids: List[Tuple[str, str]], asks: List[Tuple[str, str]], timestamp: int = 0):
        """
        Apply a full snapshot. Clears existing state.
        Args:
            bids: List of (price, size) as strings
            asks: List of (price, size) as strings
            timestamp: Provider timestamp in ms
        """
        self.bids.clear()
        self.asks.clear()

        for price, size in bids:
            self.bids[price] = float(size)
        
        for price, size in asks:
            self.asks[price] = float(size)
            
        self.last_provider_ts = timestamp
        self.last_update_ts = time.time()
        self._needs_commit = True
        self.commit_snapshot() # Immediate commit on full snapshot

    def apply_delta(self, changes: List[Dict], timestamp: int = 0):
        """
        Apply incremental updates (Deltas).
        Args:
            changes: List of dicts like {"side": "BUY", "price": "0.50", "size": "100"}
            timestamp: Provider timestamp in ms
        """
        # Monotonicity check (loose)
        # if timestamp > 0 and timestamp < self.last_provider_ts:
        #     # Ignore old updates? Or just log? Polymarket timestamp is reliable.
        #     pass 
        if timestamp > 0:
            self.last_provider_ts = timestamp

        for change in changes:
            side = change.get("side", "").upper()
            price = change.get("price")
            size_str = change.get("size", "0")
            size = float(size_str)

            if not price:
                continue

            target_dict = self.bids if side == "BUY" else self.asks

            if size == 0:
                # Remove level
                if price in target_dict:
                    del target_dict[price]
            else:
                # Update/Insert level
                target_dict[price] = size

        self.last_update_ts = time.time()
        self._needs_commit = True
        # NOTE: We do NOT commit here. We wait for the Publisher to call commit_snapshot().

    def commit_snapshot(self):
        """
        Promote current state to _cached_snapshot.
        This is where the O(N) sorting happens.
        """
        if not self._needs_commit and self._cached_snapshot:
            return

        # Calculate Best Bid/Ask & Spread
        best_bid = None
        best_ask = None
        spread = None
        
        # Bids: Max price is best
        if self.bids:
            # Sort keys (prices) descending
            sorted_bids = sorted(self.bids.keys(), key=float, reverse=True)
            best_bid = float(sorted_bids[0])
        
        # Asks: Min price is best
        if self.asks:
            # Sort keys (prices) ascending
            sorted_asks = sorted(self.asks.keys(), key=float)
            best_ask = float(sorted_asks[0])

        if best_bid is not None and best_ask is not None:
            spread = round(best_ask - best_bid, 4)

        # Update Cache
        self._cached_snapshot = {
            "token_id": self.token_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "ts_ingest_utc": self.last_update_ts,
            "ts_provider": self.last_provider_ts,
            "last_seq": self.last_sequence_id, # Placeholder if we find seq
            "staleness_ms": int((time.time() - self.last_update_ts) * 1000)
        }
        
        self._needs_commit = False
        self._commit_counter += 1

    def get_snapshot(self) -> Dict:
        """Return the cached O(1) snapshot."""
        if not self._cached_snapshot:
             # First time fallback
             self.commit_snapshot()
        
        # Update staleness on read (cheap)
        self._cached_snapshot["staleness_ms"] = int((time.time() - self.last_update_ts) * 1000)
        return self._cached_snapshot

    def get_l2_snapshot(self, top_n: int = 10) -> Dict:
        """
        Return a lightweight L2 snapshot (top N levels per side).

        This is used for recorder/replay and backtest. It trades a small
        O(N log N) sort for a richer view than the best bid/ask only.
        """
        if top_n <= 0:
            top_n = 1

        # Build top-N bids/asks from current buffers
        bids = []
        asks = []

        if self.bids:
            bid_prices = sorted(self.bids.keys(), key=float, reverse=True)[:top_n]
            for price in bid_prices:
                size = self.bids.get(price, 0.0)
                if size > 0:
                    bids.append({"price": float(price), "size": float(size)})

        if self.asks:
            ask_prices = sorted(self.asks.keys(), key=float)[:top_n]
            for price in ask_prices:
                size = self.asks.get(price, 0.0)
                if size > 0:
                    asks.append({"price": float(price), "size": float(size)})

        best_bid = bids[0]["price"] if bids else None
        best_ask = asks[0]["price"] if asks else None
        spread = round(best_ask - best_bid, 4) if best_bid is not None and best_ask is not None else None
        mid = round((best_bid + best_ask) / 2, 4) if best_bid is not None and best_ask is not None else None
        bid_depth = sum(level["size"] for level in bids) if bids else 0.0
        ask_depth = sum(level["size"] for level in asks) if asks else 0.0

        return {
            "token_id": self.token_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "mid": mid,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bids": bids,
            "asks": asks,
            "ts_ingest_utc": self.last_update_ts,
            "ts_provider": self.last_provider_ts,
            "last_seq": self.last_sequence_id,
            "staleness_ms": int((time.time() - self.last_update_ts) * 1000) if self.last_update_ts else None
        }
