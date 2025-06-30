"""
Test suite for OpenAI Image MCP Server
"""

import pytest
import asyncio
import tempfile
import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Import server components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from server import (
    validate_image_path,
    is_base64_image,
    save_base64_image,
    load_image_as_base64
)

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_validate_image_path_valid(self):
        """Test validation of valid image paths"""
        # This would need a real image file to test properly
        # For now, test the basic path validation
        assert not validate_image_path("relative/path.jpg")  # Not absolute
        assert not validate_image_path("/nonexistent/path.jpg")  # Doesn't exist
    
    def test_is_base64_image(self):
        """Test base64 image validation"""
        # Valid base64
        valid_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        assert is_base64_image(valid_b64)
        
        # Valid data URL
        data_url = f"data:image/png;base64,{valid_b64}"
        assert is_base64_image(data_url)
        
        # Invalid base64
        assert not is_base64_image("not_base64_data")
        assert not is_base64_image("")

    @pytest.mark.asyncio
    async def test_save_base64_image(self):
        """Test saving base64 image data"""
        # Create a simple 1x1 PNG in base64
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            await save_base64_image(b64_data, tmp_path)
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_load_image_as_base64(self):
        """Test loading image as base64"""
        # Create a temporary image file
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            # Save the image first
            await save_base64_image(b64_data, tmp_path)
            
            # Load it back
            loaded_b64, mime_type = await load_image_as_base64(tmp_path)
            
            assert loaded_b64 == b64_data
            assert mime_type == "image/png"
            
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])