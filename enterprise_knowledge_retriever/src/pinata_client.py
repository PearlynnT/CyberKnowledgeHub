import json
import requests
import os
import subprocess
import tempfile
from io import BytesIO
from PyPDF2 import PdfReader
from typing import List, Dict, Any

def reset_eof_of_pdf_return_stream(pdf_stream_in: bytes) -> bytes:
    # Split the binary stream into lines
    pdf_lines = pdf_stream_in.split(b'\n')

    # find the line position of the EOF
    actual_line = -1
    for i, x in enumerate(reversed(pdf_lines)):
        if b'%%EOF' in x:
            actual_line = len(pdf_lines) - i
            print(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
            break

    # Return the stream up to the EOF
    return b'\n'.join(pdf_lines[:actual_line])

# def decompress_pdf(temp_buffer: BytesIO) -> BytesIO:
#     temp_buffer.seek(0)  # Ensure we're at the start of the file.

#     # Write the in-memory BytesIO content to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as temp_input:
#         temp_input.write(temp_buffer.read())
#         temp_input_path = temp_input.name

#     # Create another temporary file for the output
#     with tempfile.NamedTemporaryFile(delete=False) as temp_output:
#         temp_output_path = temp_output.name

#     try:
#         # Run the qpdf command
#         process = subprocess.run(
#             ['qpdf', '--qdf', temp_input_path, temp_output_path],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )

#         if process.returncode != 0:
#             raise RuntimeError(f"Error decompressing PDF: {process.stderr.decode('utf-8')}")

#         # Read the decompressed content into a BytesIO object
#         with open(temp_output_path, 'rb') as output_file:
#             decompressed_data = BytesIO(output_file.read())

#         return decompressed_data
#     finally:
#         # Clean up temporary files
#         os.remove(temp_input_path)
#         os.remove(temp_output_path)


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

    def get_file_content(self, ipfs_hash: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.jwt}'
        }
        # response = requests.get(f'{self.base_url}/pinning/pinJobs/{ipfs_hash}', headers=headers)
        # response = requests.get(f'https://gateway.pinata.cloud/ipfs/{ipfs_hash}', headers=headers)
        response = requests.get(f'https://gateway.pinata.cloud/ipfs/QmXMjZxPfArGw4YNNxrt4Ny9fsqnJnE7guAsrCc3t55MhX', headers=headers)
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

        # # Load response content into BytesIO
        # input_buffer = BytesIO(response.content)

        # try:
        #     # Attempt to read PDF directly
        #     reader = PdfReader(input_buffer)
        # except Exception as e:
        #     print(f"Error reading PDF: {e}. Attempting to decompress.")
        #     # Decompress PDF if it cannot be read
        #     input_buffer = decompress_pdf(input_buffer)
        #     reader = PdfReader(input_buffer)

        # # Extract text from PDF
        # pdf_text = ""
        # for page in reader.pages:
        #     pdf_text += page.extract_text()

        # return pdf_text

        # # Save the PDF to disk
        # pdf_file_path = "downloaded_file.pdf"
        # with open(pdf_file_path, "wb") as f:
        #     f.write(response.content)

        # # Validate and extract text from the PDF
        # try:
        #     reader = PdfReader(pdf_file_path)
        #     pdf_text = "".join(page.extract_text() for page in reader.pages)
        #     return pdf_text
        # except Exception as e:
        #     raise ValueError(f"Failed to process PDF: {e}")


        # # Process the PDF stream
        # pdf_stream = reset_eof_of_pdf_return_stream(response.content)

        # # Write to a temporary file
        # temp_pdf_path = "temp.pdf"
        # with open(temp_pdf_path, "wb") as temp_pdf:
        #     temp_pdf.write(pdf_stream)

        # # Extract text using PyPDF2
        # pdf_text = ""
        # reader = PdfReader("temp.pdf")
        # for page in reader.pages:
        #     pdf_text += page.extract_text()

        # return pdf_text

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
