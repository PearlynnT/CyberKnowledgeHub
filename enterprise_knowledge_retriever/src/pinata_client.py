import json
import requests
import os
import subprocess
import tempfile
from io import BytesIO
from PyPDF2 import PdfReader
from typing import List, Dict, Any


class PinataClient:
    def __init__(self):
        self.api_key = os.environ.get('PINATA_API_KEY')
        self.api_secret = os.environ.get('PINATA_SECRET_API_KEY')
        self.jwt = os.environ.get('PINATA_JWT')
        self.base_url = 'https://api.pinata.cloud'

    def list_files(self, **kwargs) -> List[Dict[str, Any]]:
        headers = {
            # 'Authorization': f'Bearer {self.jwt}'
            "pinata_api_key": self.api_key,
            "pinata_secret_api_key": self.api_secret
        }
        response = requests.get(f'{self.base_url}/data/pinList', headers=headers, params=kwargs)
        response.raise_for_status()
        return response.json()['rows']

    # PDF files
    def get_file_content(self, ipfs_hash: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.jwt}'
        }
        response = requests.get(f'https://gateway.pinata.cloud/ipfs/{ipfs_hash}', headers=headers)
        response.raise_for_status()
        # return response.content.decode('utf-8')
        # Create a BytesIO object from the response content
        pdf_file = BytesIO(response.content)

        # Use PyPDF2 to read the PDF
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    def pin_json_to_ipfs(self, json_content: str, name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/pinning/pinJSONToIPFS"
        headers = {
            'Content-Type': 'application/json',
            'pinata_api_key': self.api_key,
            'pinata_secret_api_key': self.api_secret
        }
        payload = {
            "pinataContent": json.loads(json_content),
            "pinataMetadata": {
                "name": name
            }
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
