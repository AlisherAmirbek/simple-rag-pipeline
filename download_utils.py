import os
import requests
from huggingface_hub import hf_hub_download
import shutil

def download_file(repo_id, file_name):

    file_path = hf_hub_download(repo_id=repo_id, filename=file_name)

    print(f"File downloaded to: {file_path}")
    return file_path

