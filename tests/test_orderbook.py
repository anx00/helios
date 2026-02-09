
import sys
import os
import unittest
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market.orderbook import LocalOrderBook, OrderBookLevel

class TestLocalOrderBook(unittest.TestCase):
    def setUp(self):
        self.book = LocalOrderBook("AMZN")

    def test_apply_snapshot(self):
        bids = [("100.0", "10"), ("99.0", "5")]
        asks = [("101.0", "10"), ("102.0", "5")]
        self.book.apply_snapshot(bids, asks)

        snap = self.book.get_snapshot()
        self.assertEqual(snap.get("best_bid"), 100.0)
        self.assertEqual(snap.get("best_ask"), 101.0)
        self.assertEqual(snap.get("spread"), 1.0)
        self.assertEqual(self.book.bids["100.0"], 10.0)

    def test_apply_delta_add(self):
        # Initial state
        self.book.apply_snapshot([], [])
        
        # Add a bid
        changes = [{"side": "BUY", "price": "100.0", "size": "10"}]
        self.book.apply_delta(changes)
        self.book.commit_snapshot()

        snap = self.book.get_snapshot()
        self.assertEqual(snap.get("best_bid"), 100.0)
        self.assertEqual(self.book.bids["100.0"], 10.0)

    def test_apply_delta_update(self):
        # Initial state with a bid
        self.book.apply_snapshot([("100.0", "10")], [])
        
        # Update size
        changes = [{"side": "BUY", "price": "100.0", "size": "20"}]
        self.book.apply_delta(changes)
        
        self.assertEqual(self.book.bids["100.0"], 20.0)

    def test_apply_delta_remove(self):
        # Initial state
        self.book.apply_snapshot([("100.0", "10")], [])
        
        # Remove level (size 0)
        changes = [{"side": "BUY", "price": "100.0", "size": "0"}]
        self.book.apply_delta(changes)
        self.book.commit_snapshot()

        snap = self.book.get_snapshot()
        self.assertIsNone(snap.get("best_bid"))
        self.assertNotIn("100.0", self.book.bids)

    def test_spread_calculation(self):
        self.book.apply_snapshot([("99.5", "10")], [("100.5", "10")])
        snap = self.book.get_snapshot()
        self.assertEqual(snap.get("spread"), 1.0)
        
        # Tighten spread
        changes = [{"side": "BUY", "price": "100.0", "size": "5"}]
        self.book.apply_delta(changes)
        self.book.commit_snapshot()

        snap = self.book.get_snapshot()
        self.assertEqual(snap.get("spread"), 0.5)

if __name__ == '__main__':
    unittest.main()
