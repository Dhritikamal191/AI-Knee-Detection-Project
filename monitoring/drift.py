from pathlib import Path
import hashlib


def calculate_sha256(file_path: str | Path) -> str:
    """
    Calculate SHA256 checksum of a file.
    """

    file_path = Path(file_path)

    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()