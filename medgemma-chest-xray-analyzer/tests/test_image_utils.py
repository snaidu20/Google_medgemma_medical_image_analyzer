"""
Tests for src/image_utils.py

Run:
    pytest tests/ -v
    pytest tests/test_image_utils.py -v
"""

import os
import sys

import pytest
from PIL import Image

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.image_utils import (
    load_image,
    load_image_from_url,
    validate_xray_image,
)

SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/"
    "Chest_Xray_PA_3-8-2010.png"
)
BAD_URL = "https://this.domain.does.not.exist.invalid/image.png"
NONEXISTENT_PATH = "/tmp/does_not_exist_xray_12345.png"


# ─── URL Loading ──────────────────────────────────────────────────────────────

def test_load_image_from_url_success():
    """Download a real public-domain chest X-ray."""
    img = load_image_from_url(SAMPLE_URL)
    assert isinstance(img, Image.Image), "Should return a PIL Image"
    assert img.mode == "RGB", "Image should be converted to RGB"
    w, h = img.size
    assert w > 0 and h > 0, "Image should have positive dimensions"


def test_load_image_from_url_invalid_domain():
    """Unreachable URL should raise ConnectionError."""
    with pytest.raises(ConnectionError):
        load_image_from_url(BAD_URL)


# ─── File Loading ─────────────────────────────────────────────────────────────

def test_load_image_from_path_nonexistent():
    """Missing file should raise FileNotFoundError."""
    from src.image_utils import load_image_from_path
    with pytest.raises(FileNotFoundError):
        load_image_from_path(NONEXISTENT_PATH)


def test_load_image_from_path_unsupported_format(tmp_path):
    """Unsupported extension should raise ValueError."""
    from src.image_utils import load_image_from_path
    bad_file = tmp_path / "test.xyz"
    bad_file.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image_from_path(str(bad_file))


# ─── Smart Loader ─────────────────────────────────────────────────────────────

def test_load_image_auto_detects_url():
    """load_image() should handle an http URL and return a PIL Image."""
    img = load_image(SAMPLE_URL)
    assert isinstance(img, Image.Image)


def test_load_image_auto_detects_path():
    """load_image() should detect a non-URL string as a file path."""
    with pytest.raises(FileNotFoundError):
        load_image(NONEXISTENT_PATH)


# ─── Validation ───────────────────────────────────────────────────────────────

def test_validate_xray_image_normal():
    """A standard-size image should be valid with no warnings."""
    img = Image.new("RGB", (512, 512))
    result = validate_xray_image(img)
    assert result["is_valid"] is True
    assert result["width"] == 512
    assert result["height"] == 512
    assert result["mode"] == "RGB"
    assert result["warnings"] == []


def test_validate_xray_image_too_small():
    """An image smaller than 200x200 should trigger a warning."""
    img = Image.new("RGB", (100, 100))
    result = validate_xray_image(img)
    assert result["is_valid"] is True  # Still valid — just a warning
    assert len(result["warnings"]) >= 1
    assert any("small" in w.lower() for w in result["warnings"])


def test_validate_xray_image_too_large():
    """An image larger than 4096px on any side should trigger a warning."""
    img = Image.new("RGB", (5000, 5000))
    result = validate_xray_image(img)
    assert result["is_valid"] is True
    assert len(result["warnings"]) >= 1
    assert any("large" in w.lower() for w in result["warnings"])


def test_validate_xray_image_returns_correct_keys():
    """Validation dict must always include required keys."""
    img = Image.new("RGB", (800, 800))
    result = validate_xray_image(img)
    required_keys = {"is_valid", "width", "height", "mode", "warnings"}
    assert required_keys.issubset(result.keys())
