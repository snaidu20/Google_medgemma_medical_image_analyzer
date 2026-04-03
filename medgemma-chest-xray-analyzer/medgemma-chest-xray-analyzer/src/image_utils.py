"""
Image loading and validation utilities for chest X-ray analysis.
"""

import io
from pathlib import Path
from typing import Union

import requests
from PIL import Image

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

_USER_AGENT = (
    "MedGemma-XRay-Analyzer/1.0 "
    "(https://github.com/your-repo; research use only)"
)


def load_image_from_path(file_path: str) -> Image.Image:
    """Load an image from a local file path.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return Image.open(path).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    """Download and load an image from a URL.

    Args:
        url: HTTP/HTTPS URL pointing to an image.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ConnectionError: If the download fails or the URL is unreachable.
        ValueError: If the response content is not a valid image.
    """
    headers = {"User-Agent": _USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Could not connect to URL: {url}") from exc
    except requests.exceptions.HTTPError as exc:
        raise ConnectionError(
            f"HTTP error {exc.response.status_code} fetching: {url}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ConnectionError(f"Request timed out for URL: {url}") from exc

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image from URL: {url}") from exc


def load_image(source: str) -> Image.Image:
    """Smart loader — auto-detects URL vs local file path.

    Args:
        source: Either an HTTP/HTTPS URL or a local file path.

    Returns:
        PIL Image in RGB mode.
    """
    if source.lower().startswith(("http://", "https://")):
        return load_image_from_url(source)
    return load_image_from_path(source)


def validate_xray_image(image: Image.Image) -> dict:
    """Validate a loaded chest X-ray image.

    Args:
        image: PIL Image object to validate.

    Returns:
        dict with keys:
            - is_valid (bool)
            - width (int)
            - height (int)
            - mode (str)
            - warnings (list[str])
    """
    warnings = []
    width, height = image.size

    if width < 200 or height < 200:
        warnings.append(
            f"Image is very small ({width}x{height}px). "
            "This may reduce analysis accuracy. Recommend at least 512x512px."
        )

    if width > 4096 or height > 4096:
        warnings.append(
            f"Image is very large ({width}x{height}px). "
            "Consider resizing to ≤4096px to improve processing speed."
        )

    return {
        "is_valid": True,
        "width": width,
        "height": height,
        "mode": image.mode,
        "warnings": warnings,
    }
