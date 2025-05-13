import boto3
import os
import sys
from collections import defaultdict
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

bucket_name = "fcp-indi"
prefix = "data/Projects/ABIDE2/RawData"
exclude_dirs = ['dwi/']

# Anonymous access config
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Faster download config
transfer_config = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,
    max_concurrency=10,
    multipart_chunksize=5 * 1024 * 1024,
    use_threads=True
)

paginator = s3.get_paginator('list_objects_v2')

class ProgressPercentage:
    def __init__(self, filename, filesize):
        self._filename = filename
        self._filesize = filesize
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._filesize) * 100
        sys.stdout.write(
            f"\rDownloading {self._filename}... {percentage:.2f}%"
        )
        sys.stdout.flush()

# Group keys by subject
subject_files = defaultdict(list)

for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if any(excluded in key.lower() for excluded in exclude_dirs):
            continue
        parts = key.split('/')
        if len(parts) > 4:
            subject_prefix = '/'.join(parts[:5]) + '/'
            subject_files[subject_prefix].append((key, obj['Size']))

# Download task for a single file
def download_file(key, size):
    local_path = os.path.join(".", key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    progress = ProgressPercentage(key, size)
    try:
        s3.download_file(
            bucket_name,
            key,
            local_path,
            Callback=progress,
            Config=transfer_config
        )
        print(" ✔")
    except Exception as e:
        print(f"\n❌ Failed to download {key}: {e}")

# Using ThreadPoolExecutor for parallel downloads
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = []

    for subject_prefix, files in subject_files.items():
        anat_files = [(k, s) for k, s in files if '/anat/' in k]
        if not anat_files:
            print(f"⏭️ Skipping {subject_prefix} (no anat files)")
            continue

        print(f"⬇️ Queueing anat files from {subject_prefix}")
        for key, size in anat_files:
            futures.append(executor.submit(download_file, key, size))

    # Wait for all futures to complete
    for future in as_completed(futures):
        _ = future.result()
