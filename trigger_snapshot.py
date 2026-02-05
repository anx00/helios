import asyncio
from monitor import snapshot_current_state

async def main():
    print("Triggering manual snapshot...")
    await snapshot_current_state()
    print("Snapshot complete.")

if __name__ == "__main__":
    asyncio.run(main())
