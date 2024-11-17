import requests
import os

class PinataHelper:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.pinata.cloud"

    def get_file_metadata(self, ipfs_hash: str) -> dict:
        url = f"{self.base_url}/data/pinList?hashContains={ipfs_hash}"
        headers = {
            "pinata_api_key": self.api_key,
            "pinata_secret_api_key": self.api_secret
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def download_file(self, ipfs_hash: str, save_path: str):
        url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
