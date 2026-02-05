
import sys
import os
import asyncio
import unittest
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market.polymarket_ws import PolymarketWebSocketClient

class TestPolymarketWS(unittest.IsolatedAsyncioTestCase):
    async def test_connection_and_subscription(self):
        client = PolymarketWebSocketClient()
        
        # Test Connection
        connected = await client.connect()
        self.assertTrue(connected, "Failed to connect to Polymarket WS")
        
        # Test Subscription (using a known market ID if possible, or random format)
        # Using a dummy ID just to see if it accepts the message without error
        # In real integration we'd use a real ID from Gamma, but for connectivity check this is enough
        # client.ws.send shouldn't raise exception
        
        try:
            # dummy market ID
            await client.subscribe_to_markets(["0x0000000000000000000000000000000000000000"])
            subscription_sent = True
        except Exception as e:
            subscription_sent = False
            print(f"Subscription error: {e}")
            
        self.assertTrue(subscription_sent, "Failed to send subscription message")
        
        # Clean up
        await client.disconnect()

if __name__ == '__main__':
    unittest.main()
