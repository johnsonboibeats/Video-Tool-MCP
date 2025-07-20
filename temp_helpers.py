#!/usr/bin/env python3
"""
Modern OpenAI Image MCP Server
Built with FastMCP and latest OpenAI APIs (Images + Vision)

Features:
- Image generation with gpt-image-1
- Image editing with mask support  
- Image analysis with GPT-4 Vision
- Smart OCR and text extraction
- Batch processing with progress tracking
- Intelligent image workflows
"""

import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union
import json

import aiofiles
from openai import AsyncOpenAI, AsyncAzureOpenAI
from PIL import Image as PILImage, ImageOps
import httpx
from fastmcp import FastMCP, Context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AppContext:
    """Application context with shared resources"""
    openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI]
    temp_dir: Path
    http_client: httpx.AsyncClient

# Global context variable to store the application context
app_context: AppContext = None

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle with proper resource cleanup"""
    # Initialize OpenAI client based on environment
    if os.getenv("AZURE_OPENAI_API_KEY"):
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "")
        )
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Setup temp directory
    temp_dir = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir())) / "openai_image_mcp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize HTTP client
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            # Create context and store it globally
            global app_context
            app_context = AppContext(
                openai_client=client,
                temp_dir=temp_dir,
                http_client=http_client
            )
            yield app_context
        finally:
            # Cleanup temp files older than 1 hour
            import time
            current_time = time.time()
            for file_path in temp_dir.glob("*"):
                if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass

# Create FastMCP server with lifespan management
mcp = FastMCP(
    name="Image Tool MCP",
    lifespan=app_lifespan,
    dependencies=["fastmcp>=0.4.0", "openai>=1.97.0", "pillow>=11.3.0", "httpx>=0.28.1"]
)

# Utility functions
async def save_base64_image(base64_data: str, file_path: Path, format: str = "PNG") -> None:
    """Save base64 image data to file"""
    image_data = base64.b64decode(base64_data)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(image_data)

async def load_image_as_base64(file_path: Union[str, Path]) -> tuple[str, str]:
    """Load image file and return as base64 with mime type"""
    file_path = Path(file_path)
    
    async with aiofiles.open(file_path, "rb") as f:
        image_data = await f.read()
    
    # Determine mime type from extension
    ext = file_path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg", 
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif"
    }.get(ext, "image/png")
    
    base64_data = base64.b64encode(image_data).decode()
    return base64_data, mime_type

def validate_image_path(path: str) -> bool:
    """Validate if path is absolute and points to valid image"""
    if not os.path.isabs(path):
        return False
    
    if not os.path.exists(path):
        return False
        
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def is_base64_image(data: str) -> bool:
    """Check if string is valid base64 image data"""
    try:
        if data.startswith("data:image/"):
            # Data URL format
            return True
        # Try to decode as base64
        base64.b64decode(data)
        return True
    except Exception:
        return False#!/usr/bin/env python3
"""
Modern OpenAI Image MCP Server
Built with FastMCP and latest OpenAI APIs (Images + Vision)

Features:
- Image generation with gpt-image-1
- Image editing with mask support  
- Image analysis with GPT-4 Vision
- Smart OCR and text extraction
- Batch processing with progress tracking
- Intelligent image workflows
"""

import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union
import json

import aiofiles
from openai import AsyncOpenAI, AsyncAzureOpenAI
from PIL import Image as PILImage, ImageOps
import httpx
from fastmcp import FastMCP, Context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AppContext:
    """Application context with shared resources"""
    openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI]
    temp_dir: Path
    http_client: httpx.AsyncClient

# Global context variable to store the application context
app_context: AppContext = None

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle with proper resource cleanup"""
    # Initialize OpenAI client based on environment
    if os.getenv("AZURE_OPENAI_API_KEY"):
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "")
        )
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Setup temp directory
    temp_dir = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir())) / "openai_image_mcp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize HTTP client
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            # Create context and store it globally
            global app_context
            app_context = AppContext(
                openai_client=client,
                temp_dir=temp_dir,
                http_client=http_client
            )
            yield app_context
        finally:
            # Cleanup temp files older than 1 hour
            import time
            current_time = time.time()
            for file_path in temp_dir.glob("*"):
                if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass

# Create FastMCP server with lifespan management
mcp = FastMCP(
    name="Image Tool MCP",
    lifespan=app_lifespan,
    dependencies=["fastmcp>=0.4.0", "openai>=1.97.0", "pillow>=11.3.0", "httpx>=0.28.1"]
)

# Utility functions
async def save_base64_image(base64_data: str, file_path: Path, format: str = "PNG") -> None:
    """Save base64 image data to file"""
    image_data = base64.b64decode(base64_data)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(image_data)

async def load_image_as_base64(file_path: Union[str, Path]) -> tuple[str, str]:
    """Load image file and return as base64 with mime type"""
    file_path = Path(file_path)
    
    async with aiofiles.open(file_path, "rb") as f:
        image_data = await f.read()
    
    # Determine mime type from extension
    ext = file_path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg", 
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif"
    }.get(ext, "image/png")
    
    base64_data = base64.b64encode(image_data).decode()
    return base64_data, mime_type

def validate_image_path(path: str) -> bool:
    """Validate if path is absolute and points to valid image"""
    if not os.path.isabs(path):
        return False
    
    if not os.path.exists(path):
        return False
        
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def is_base64_image(data: str) -> bool:
    """Check if string is valid base64 image data"""
    try:
        if data.startswith("data:image/"):
            # Data URL format
            return True
        # Try to decode as base64
        base64.b64decode(data)
        return True
    except Exception:
        return False


# =============================================================================
# IMAGE GENERATION TOOLS (OpenAI Images API)
# =============================================================================
