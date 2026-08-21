from pathlib import Path
import tempfile

from fastapi import UploadFile, HTTPException


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def validate_image_file(file: UploadFile) -> None:
    """
    Validate that the uploaded file is a supported image format.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Only JPG, JPEG and PNG images are allowed."
            )
        )


async def save_upload_file(file: UploadFile) -> Path:
    """
    Save an uploaded image to a temporary file.
    """

    validate_image_file(file)

    suffix = Path(file.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        content = await file.read()
        tmp.write(content)

        return Path(tmp.name)