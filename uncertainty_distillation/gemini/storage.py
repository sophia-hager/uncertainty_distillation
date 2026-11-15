"""Reading and writing JSONL datasets in Google Cloud Storage.

GCS is the hand-off point between the pipeline and Vertex AI: datasets, batch
job inputs and batch job outputs all live there as JSONL.
"""

import json
import re


def _storage():
    try:
        from google.cloud import storage
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The Gemini pipeline needs the Cloud Storage client: "
            "pip install -r requirements-gemini.txt"
        ) from error
    return storage


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    match = re.match(r"gs://(.*?)/(.*)", gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    return match.groups()


def get_client(project: str | None = None):
    return _storage().Client(project=project or None)


def write_jsonl(client, gcs_uri: str, dataset: list[dict]) -> str:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    blob = client.get_bucket(bucket_name).blob(blob_name)
    with blob.open("w") as blob_file:
        for item in dataset:
            blob_file.write(json.dumps(item) + "\n")
    return gcs_uri


def read_jsonl(client, gcs_uri: str) -> list[dict]:
    """Read one JSONL object, or every output file under a directory URI."""
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    bucket = client.get_bucket(bucket_name)

    if not gcs_uri.endswith("/"):
        content = bucket.blob(blob_name).download_as_string()
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    items = []
    for blob in bucket.list_blobs(prefix=blob_name):
        # Skip directory placeholders and the inputs we uploaded ourselves.
        if blob.name.endswith("/") or blob.name.endswith("input.jsonl"):
            continue
        content = blob.download_as_string()
        items.extend(json.loads(line) for line in content.splitlines() if line.strip())
    return items
