import logging
import os
import asyncio
import websockets
import re
import aiofiles
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
env_file = ".env.client"  # Change this to .env or .env.2 as needed
load_dotenv(env_file)

# Azure Blob Storage configuration
SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
CLIENT_ID = os.getenv("CLIENT_ID")
script_directory = os.path.dirname(os.path.realpath(__file__))
GLOBAL_MODEL_PATH = os.path.join(script_directory, "GLOBAL_MODELS")

if not SERVER_ACCOUNT_URL or not SERVER_CONTAINER_NAME or not CLIENT_ID:
    logging.error("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise
    
class WebSocketClient:
    def __init__(self, client_id, server_host, port=8000):
        self.client_id = client_id
        # self.server_url = f"ws://{server_host}:{port}/ws/{client_id}"
        self.server_url = f"wss://{server_host}/ws/{client_id}"
        print(self.server_url)
        self.websocket = None
        self.connected = False
        self.reconnect_delay = 5  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 300  # Maximum reconnect delay (5 minutes)
        self.last_downloaded_version = self.get_last_downloaded_version()

        # Configure logging
        log_dir = os.path.join(GLOBAL_MODEL_PATH, self.client_id)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/client_websocket.log"),
                logging.StreamHandler()
            ]
        )
        # print(f"{log_dir}/client_websocket.log")

    def get_last_downloaded_version(self):
        """Retrieve the last downloaded model version from local storage."""
        version_path = os.path.join(GLOBAL_MODEL_PATH, self.client_id, "last_downloaded_version.txt")
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                version = f.read().strip()
                match = re.search(r'g(\d+)\.h5', version)
                if match:
                    return int(match.group(1))
        return -1

    def update_last_downloaded_version(self, version):
        """Update the last downloaded version in local storage."""
        version_path = os.path.join(GLOBAL_MODEL_PATH, self.client_id, "last_downloaded_version.txt")
        os.makedirs(os.path.dirname(version_path), exist_ok=True)
        with open(version_path, "w") as f:
            f.write(version)
        logging.info(f"Updated last downloaded model version to {version}")

    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.server_url) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    self.reconnect_delay = 5
                    await self.handle_messages()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                await self.handle_disconnection()


    async def handle_disconnection(self):
        self.connected = False
        logging.info(f"Waiting {self.reconnect_delay} seconds before reconnecting...")
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def handle_messages(self):
        try:
            while True:
                message = await self.websocket.recv()
                if not message:
                    break
                    
                if message.startswith(("LATEST_MODEL:", "NEW_MODEL:")):
                    print(f"Received message: {message}")
                    version = message.split(":")[1]
                    match = re.search(r'g(\d+)\.h5', version)
                    if match and int(match.group(1)) > self.last_downloaded_version:
                        await self.retry_update_model(version)
                else:
                    print(f"Received message: {message}")
                        
        except Exception as e:
            logging.error(f"Message handling error: {e}")
            self.connected = False
            raise

    async def send_status(self):
        """Send periodic status updates to server"""
        while self.connected:
            try:
                status_message = f"STATUS:client_{self.client_id}_active"
                await self.websocket.send(status_message)
                await asyncio.sleep(60)  # Send status update every minute
            except Exception as e:
                logging.error(f"Error sending status update: {e}")

    async def update_model(self, filename):
        try:
            local_file_path = await self.download_model(filename)
            if local_file_path:
                logging.info(f"Successfully downloaded latest global model weights: {local_file_path}")
                self.update_last_downloaded_version(filename)
                return True
        except Exception as e:
            logging.error(f"Error updating model: {e}")
            return False
    
    async def retry_update_model(self, filename, retries=5, delay=10):
        attempt = 0
        while attempt < retries:
            try:
                success = await self.update_model(filename)
                if success:
                    # logging.info(f"Successfully updated model on attempt {attempt + 1}")
                    return True
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                logging.error(f"Transient error on attempt {attempt + 1}: {e}")
            except (KeyError, ValueError) as e:
                logging.error(f"Permanent error: {e}")
                break  # Don't retry for permanent errors
            
            attempt += 1
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

        logging.error(f"Failed to update model after {retries} attempts.")
        return False

    async def download_model(self, filename):
        try:
            os.makedirs(GLOBAL_MODEL_PATH, exist_ok=True)
            local_path = os.path.join(GLOBAL_MODEL_PATH, filename)

            blob_client = BLOB_SERVICE_CLIENT.get_blob_client(
                container=SERVER_CONTAINER_NAME, 
                blob=filename
            )

            async with aiofiles.open(local_path, "wb") as file:
                data = blob_client.download_blob().readall()
                await file.write(data)

            return local_path
        except Exception as e:
            logging.error(f"Download error: {e}")
            return None

async def run_websocket_service():
    """Independent WebSocket service."""
    host = os.getenv("SERVER_HOST")
    port = 8000

    while True:
        try:
            ws_client = WebSocketClient(CLIENT_ID, host, port)
            await ws_client.connect()
        except Exception as e:
            logging.error(f"WebSocket service error: {e}")
            await asyncio.sleep(5)  # Wait before restarting service

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_websocket_service())
    except KeyboardInterrupt:
        logging.info("Service stopped by user")
    finally:
        loop.close()