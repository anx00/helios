from py_clob_client.client import ClobClient

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
PRIVATE_KEY = "0x919a32d6de75a94940220eb05e783ac5d639f49f99d7760d74e88f6f94723a2a"
SIGNATURE_TYPE = 2
FUNDER = None  # o tu Profile/Wallet Address si usas proxy/safe

client = ClobClient(
    HOST,
    chain_id=CHAIN_ID,
    key=PRIVATE_KEY,
    signature_type=SIGNATURE_TYPE,
    funder=FUNDER,
)

creds = client.create_or_derive_api_creds()
print("API_KEY=", creds.api_key)
print("API_SECRET=", creds.api_secret)
print("API_PASSPHRASE=", creds.api_passphrase)