#!/usr/bin/env python3
"""
Image Tool MCP Server - Simplified Railway Deploy Version
"""


# Suppress warnings for cleaner deployment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*not JSON serializable.*", category=UserWarning)

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import secrets
import string
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict


# Configure logging early - check for stdio mode to avoid polluting stdout
import sys
if "--transport" in sys.argv and "stdio" in sys.argv:
    logging.basicConfig(level=logging.CRITICAL)
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


# Core imports
try:
    import aiofiles
    from openai import AsyncOpenAI, AsyncAzureOpenAI
    from fastmcp import FastMCP, Context
    from dotenv import load_dotenv
except Exception as e:
    logger.error(f"Core import failed: {e}")
    raise


# Image processing imports with graceful fallbacks
try:
    from PIL import Image as PILImage, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available - image processing will be disabled")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("Httpx not available - some image features may be disabled")

# Load environment variables
load_dotenv()

class AppContext(BaseModel):
    """Application context with shared resources"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI]
    temp_dir: Path
    http_client: Optional[httpx.AsyncClient] = None

# Global app context for FastMCP tools
_global_app_context: Optional[AppContext] = None

def get_app_context() -> AppContext:
    """Get application context from global reference"""
    if _global_app_context is not None:
        return _global_app_context
    raise RuntimeError("Application context not initialized")

def check_openai_client(client) -> None:
    """Check if OpenAI client is available"""
    if client is None:
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")

async def handle_file_input(file_input: str, app_context: AppContext) -> str:
    """Handle file input: base64 data and absolute paths"""
    # Handle base64 data URLs
    if file_input.startswith('data:'):
        try:
            header, data = file_input.split(',', 1)
            file_data = base64.b64decode(data)
            
            # Determine file extension from MIME type
            mime_match = re.search(r'data:([^;]+)', header)
            if mime_match:
                mime_type = mime_match.group(1)
                extension = mimetypes.guess_extension(mime_type) or '.bin'
            else:
                extension = '.bin'
            
            temp_path = app_context.temp_dir / f"temp_file_{int(time.time())}{extension}"
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_data)
            
            return str(temp_path)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")
    
    # Reject relative paths and other potentially unsafe inputs
    raise ValueError("Invalid file path: must be absolute path or base64 data")

def initialize_app_context():
    """Initialize application context synchronously"""
    try:
        # Initialize OpenAI client
        if os.getenv("AZURE_OPENAI_API_KEY"):
            client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "")
            )
            logger.info("Initialized Azure OpenAI client")
        elif os.getenv("OPENAI_API_KEY"):
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Initialized OpenAI client")
        else:
            logger.warning("No OpenAI API key found - image tools will not function")
            client = None
        
        # Setup temp directory
        temp_dir = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir())) / "image_tool_mcp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory: {temp_dir}")
        
        # Create HTTP client
        http_client = httpx.AsyncClient() if HTTPX_AVAILABLE else None
        
        # Create context
        context = AppContext(
            openai_client=client,
            temp_dir=temp_dir,
            http_client=http_client
        )
        
        # Set global context for FastMCP tools
        global _global_app_context
        _global_app_context = context
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to initialize app context: {e}")
        raise

# Initialize application context at startup
try:
    initialize_app_context()
    logger.info("Application context initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize application context: {e}")
    logger.warning("Server will start but image tools may not function properly")
    # Create a minimal context to allow server to start
    _global_app_context = AppContext(
        openai_client=None,
        temp_dir=Path(tempfile.gettempdir()) / "image_tool_mcp",
        http_client=None
    )

# Railway security configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://claude.ai,https://web.claude.ai").split(",")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))

logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")
logger.info(f"Rate limit: {MAX_REQUESTS_PER_MINUTE} requests per minute")

# Create FastMCP server
mcp = FastMCP("Image Tool MCP")

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
# Removed CORS middleware for simplified Railway deployment
from collections import defaultdict

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    def __init__(self, app):
        super().__init__(app)
        self.max_requests = 100
        self.window = 60
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if now - req_time < self.window
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                {"error": "Rate limit exceeded. Please try again later."}, 
                status_code=429
            )
        
        self.requests[client_ip].append(now)
        return await call_next(request)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log requests for monitoring"""
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log all requests except health checks
        if request.url.path != "/health":
            logger.info(f"Request from {client_ip} - {user_agent} - {request.method} {request.url.path}")
        else:
            # Log health checks at debug level
            logger.debug(f"Health check from {client_ip}")
        
        return await call_next(request)

# Add basic middleware for Railway deployment
mcp.add_middleware(RateLimitMiddleware)
mcp.add_middleware(RequestLoggingMiddleware)

logger.info("Security middleware configured")
logger.info("FastMCP server instance created and configured")
logger.info("Custom routes registered: /health, /")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    """Health check endpoint for Railway deployment"""
    logger.info("Health check endpoint called")
    try:
        response_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "server": "Image Tool MCP Server"
        }
        
        # Safely check OpenAI configuration
        try:
            response_data["openai_configured"] = _global_app_context.openai_client is not None if _global_app_context else False
        except:
            response_data["openai_configured"] = False
            
        logger.info(f"Health check response: {response_data}")
        return JSONResponse(response_data)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request):
    """Root endpoint for basic connectivity test"""
    logger.info("Root endpoint called")
    return JSONResponse({
        "message": "Image Tool MCP Server is running",
        "health_check": "/health",
        "mcp_endpoint": "/mcp/"
    })

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

def validate_image_path(path: str) -> str:
    """Validate file path input with security checks"""
    if not path or not path.strip():
        raise ValueError("File path cannot be empty")
    
    path = path.strip()
    
    # Security: Prevent path traversal attacks
    if ".." in path:
        raise ValueError("Invalid file path: potential security risk")
    
    # Allow local absolute paths (starting with /)
    if path.startswith('/'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path
    
        
    # Allow base64 data URLs
    if path.startswith('data:'):
        return path
    
    # Reject relative paths and other potentially unsafe inputs
    raise ValueError("Invalid file path: must be absolute path or base64 data")

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

async def get_file_path(file_input: str) -> str:
    """Handle file input and return local file path"""
    if not file_input:
        raise ValueError("File input cannot be empty")
    
    # Handle base64 data URLs by converting to temp file
    if file_input.startswith('data:'):
        app_context = get_app_context()
        return await handle_file_input(file_input, app_context)
    
    # Validate and return absolute paths
    return validate_image_path(file_input)

# =============================================================================
# IMAGE PROCESSING TOOLS
# =============================================================================

@mcp.tool()
async def create_image(
    prompt: str,
    ctx: Context,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    background: Literal["transparent", "opaque", "auto"] = "auto", 
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_compression: Optional[int] = None,
    moderation: Literal["auto", "low"] = "auto",
    n: int = 1,
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None
) -> Union[str, list[str]]:
    """Generate images from text prompts using OpenAI's latest gpt-image-1 model.
    
    Supports both local files and Google Drive files.
    
    Args:
        prompt: Text description of the image to generate (max 32000 chars)
        model: Image generation model (only gpt-image-1 supported)
        size: Image dimensions or 'auto' for optimal size
        quality: Generation quality level
        background: Background handling for generated image
        output_format: Image format (png/jpeg/webp)
        output_compression: Compression level 0-100 (webp/jpeg only)
        moderation: Content moderation level
        n: Number of images to generate (1-10)
        output_mode: Return as base64 data or save to file
        file_path: Absolute path for file output (required if output_mode='file')
        
    Returns:
        Generated image(s) as base64 data or file paths
    """
    # Get application context
    app_context = get_app_context()
    
    client = app_context.openai_client
    check_openai_client(client)
    temp_dir = app_context.temp_dir
    
    # Validate inputs
    if len(prompt) > 32000:
        raise ValueError("Prompt must be 32000 characters or less")
    
    if n < 1 or n > 10:
        raise ValueError("Number of images must be between 1 and 10")
        
    if background == "transparent" and output_format not in ["png", "webp"]:
        raise ValueError("Transparent background requires PNG or WebP format")
        
    if output_mode == "file" and not file_path:
        raise ValueError("file_path is required when output_mode is 'file'")
        
    if file_path and not os.path.isabs(file_path):
        raise ValueError("file_path must be an absolute path")
    
    # Progress tracking for batch generation
    if n > 1:
        await ctx.report_progress(0, n, f"Starting generation of {n} images...")
    
    # Prepare API parameters
    params = {
        "prompt": prompt,
        "model": model,
        "n": n
    }
    
    # Add optional parameters
    if size != "auto":
        params["size"] = size
    if quality != "auto":
        params["quality"] = quality  
    if background != "auto":
        params["background"] = background
    if output_format:
        params["output_format"] = output_format
    if moderation != "auto":
        params["moderation"] = moderation
        
    # Add compression for supported formats
    if output_compression is not None and output_format in ["webp", "jpeg"]:
        if 0 <= output_compression <= 100:
            params["output_compression"] = output_compression
        else:
            raise ValueError("output_compression must be between 0 and 100")
    
    try:
        # Generate images
        ctx.info(f"Generating {n} image(s) with prompt: {prompt[:100]}...")
        response = await client.images.generate(**params)
        
        # Process results
        images = []
        file_paths = []
        
        for i, image_data in enumerate(response.data):
            if n > 1:
                await ctx.report_progress(i + 1, n, f"Processing image {i + 1}/{n}")
            
            b64_data = image_data.b64_json
            
            if output_mode == "file":
                # Save to file
                if n > 1:
                    # Multiple files: add index
                    path = Path(file_path)
                    save_path = path.parent / f"{path.stem}_{i+1}{path.suffix}"
                else:
                    save_path = Path(file_path)
                
                await save_base64_image(b64_data, save_path, output_format.upper())
                file_paths.append(str(save_path))
                ctx.info(f"Image saved to: {save_path}")
                
            else:
                # Return as base64 string
                images.append(f"data:image/{output_format};base64,{b64_data}")
        
        # Return results
        if output_mode == "file":
            return file_paths if n > 1 else file_paths[0]
        else:
            return images if n > 1 else images[0]
            
    except Exception as e:
        ctx.error(f"Image generation failed: {str(e)}")
        raise ValueError(f"Failed to generate image: {str(e)}")

@mcp.tool()
async def analyze_image(
    image: str,
    ctx: Context,
    prompt: str = "Describe this image in detail, including objects, people, scenery, colors, mood, and any text visible.",
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    detail: Literal["low", "high", "auto"] = "auto"
) -> str:
    """Analyze an image using OpenAI's Vision API to extract detailed information.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Absolute file path, Google Drive URL/ID, or base64 string of image to analyze
        prompt: Analysis prompt (what to look for in the image)
        model: Vision model to use (gpt-4o, gpt-4o-mini, etc.)
        max_tokens: Maximum tokens in response
        detail: Image detail level for processing
        
    Returns:
        Detailed analysis of the image content
    """
    # Get application context
    app_context = get_app_context()
    
    client = app_context.openai_client
    check_openai_client(client)
    
    # Get file path (handles Google Drive files, local files, and base64)
    file_path = await get_file_path(image)
    
    # Prepare image for API
    try:
        # Validate the file path using the new unified approach
        validated_path = validate_image_path(file_path)
        
        # Load image file as base64
        base64_data, mime_type = await load_image_as_base64(validated_path)
        image_url = f"data:{mime_type};base64,{base64_data}"
        
    except (ValueError, FileNotFoundError) as e:
        # If validation fails, try to handle as base64 data
        if is_base64_image(image):
            if image.startswith("data:image/"):
                image_url = image
            else:
                # Add data URL prefix
                image_url = f"data:image/png;base64,{image}"
        else:
            raise ValueError(f"Invalid image input: {str(e)}")
    
    try:
        ctx.info(f"Analyzing image with model {model}...")
        
        # Call Vision API
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": detail
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        
        analysis = response.choices[0].message.content
        ctx.info("Image analysis completed successfully")
        return analysis
        
    except Exception as e:
        ctx.error(f"Image analysis failed: {str(e)}")
        raise ValueError(f"Failed to analyze image: {str(e)}")

# =============================================================================
# MISSING TOOLS RESTORATION
# =============================================================================

@mcp.tool()
async def edit_image(
    image: str,
    prompt: str,
    mask: Optional[str] = None,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None,
    ctx: Context = None
) -> Union[str, list[str]]:
    """Edit existing images using masks and text prompts.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Original image (file path, Google Drive URL/ID, or base64)
        prompt: Text description of desired changes
        mask: Optional mask image for selective editing (same formats as image)
        model: Image generation model
        size: Image dimensions
        quality: Generation quality level
        output_format: Image format
        output_mode: Return as base64 data or save to file
        file_path: Absolute path for file output (required if output_mode='file')
        
    Returns:
        Edited image(s) as base64 data or file paths
    """
    app_context = get_app_context()
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    # Get file paths
    image_path = await get_file_path(image)
    mask_path = await get_file_path(mask) if mask else None
    
    # Prepare images for API
    image_base64, _ = await load_image_as_base64(image_path)
    
    params = {
        "model": model,
        "prompt": prompt,
        "image": image_base64,
        "size": size if size != "auto" else "1024x1024",
        "quality": quality,
        "response_format": "b64_json"
    }
    
    if mask_path:
        mask_base64, _ = await load_image_as_base64(mask_path)
        params["mask"] = mask_base64
    
    try:
        if ctx: ctx.info(f"Editing image with prompt: {prompt[:100]}...")
        response = await client.images.edit(**params)
        
        b64_data = response.data[0].b64_json
        
        if output_mode == "file":
            if not file_path:
                raise ValueError("file_path required for file output")
            save_path = Path(file_path)
            await save_base64_image(b64_data, save_path, output_format.upper())
            return str(save_path)
        else:
            return f"data:image/{output_format};base64,{b64_data}"
            
    except Exception as e:
        raise ValueError(f"Failed to edit image: {str(e)}")

@mcp.tool()
async def generate_variations(
    image: str,
    n: int = 1,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None,
    ctx: Context = None
) -> Union[str, list[str]]:
    """Generate variations of existing images.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Original image (file path, Google Drive URL/ID, or base64)
        n: Number of variations to generate (1-10)
        model: Image generation model
        size: Image dimensions
        quality: Generation quality level
        output_format: Image format
        output_mode: Return as base64 data or save to file
        file_path: Absolute path for file output (required if output_mode='file')
        
    Returns:
        Image variation(s) as base64 data or file paths
    """
    app_context = get_app_context()
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    if n < 1 or n > 10:
        raise ValueError("Number of variations must be between 1 and 10")
    
    image_path = await get_file_path(image)
    image_base64, _ = await load_image_as_base64(image_path)
    
    params = {
        "model": model,
        "image": image_base64,
        "n": n,
        "size": size if size != "auto" else "1024x1024",
        "response_format": "b64_json"
    }
    
    try:
        if ctx: ctx.info(f"Generating {n} variation(s)...")
        response = await client.images.create_variation(**params)
        
        results = []
        for i, img_data in enumerate(response.data):
            b64_data = img_data.b64_json
            
            if output_mode == "file":
                if not file_path:
                    raise ValueError("file_path required for file output")
                path = Path(file_path)
                if n > 1:
                    save_path = path.parent / f"{path.stem}_{i+1}{path.suffix}"
                else:
                    save_path = path
                await save_base64_image(b64_data, save_path, output_format.upper())
                results.append(str(save_path))
            else:
                results.append(f"data:image/{output_format};base64,{b64_data}")
        
        return results if n > 1 else results[0]
        
    except Exception as e:
        raise ValueError(f"Failed to generate variations: {str(e)}")

@mcp.tool()
async def extract_text(
    image: str,
    language: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Extract text from images using OCR.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Image to extract text from (file path, Google Drive URL/ID, or base64)
        language: Language hint for better accuracy (e.g., 'eng', 'spa', 'fra')
        
    Returns:
        Extracted text with confidence scores and bounding boxes
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    prompt = "Extract all text from this image. If the image contains multiple languages, identify each language. Provide the text with confidence scores and approximate locations if possible."
    
    try:
        if ctx: ctx.info("Extracting text from image...")
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        text_content = response.choices[0].message.content
        
        return {
            "success": True,
            "text": text_content,
            "source_file": str(image_path),
            "language_hint": language
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

@mcp.tool()
async def compare_images(
    image1: str,
    image2: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Compare two images and analyze differences.
    
    Supports both local files and Google Drive files.
    
    Args:
        image1: First image (file path, Google Drive URL/ID, or base64)
        image2: Second image (file path, Google Drive URL/ID, or base64)
        
    Returns:
        Detailed comparison analysis including similarities and differences
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image1_path = await get_file_path(image1)
    image2_path = await get_file_path(image2)
    
    image1_base64, _ = await load_image_as_base64(image1_path)
    image2_base64, _ = await load_image_as_base64(image2_path)
    
    prompt = """Compare these two images and provide a detailed analysis of:
1. Overall similarity/difference score (0-100%)
2. Visual elements that are the same
3. Visual elements that are different
4. Color differences
5. Composition differences
6. Content changes
7. Quality differences
8. Style variations

Be specific and quantitative where possible."""
    
    try:
        if ctx: ctx.info("Comparing images...")
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image1_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image2_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        return {
            "success": True,
            "comparison": response.choices[0].message.content,
            "image1": str(image1_path),
            "image2": str(image2_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "image1": str(image1_path),
            "image2": str(image2_path)
        }

@mcp.tool()
async def smart_edit(
    image: str,
    analysis_prompt: str,
    edit_prompt: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Intelligent image editing with analysis and targeted modifications.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Image to edit (file path, Google Drive URL/ID, or base64)
        analysis_prompt: What to analyze in the image
        edit_prompt: How to modify based on the analysis
        
    Returns:
        Edited image with analysis and modification details
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    # First, analyze the image
    analysis_prompt_full = f"{analysis_prompt} Provide specific details that would help with targeted editing."
    
    try:
        if ctx: ctx.info("Analyzing image for smart editing...")
        
        analysis_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt_full},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        analysis = analysis_response.choices[0].message.content
        
        # Now edit based on analysis
        edit_prompt_full = f"Based on this analysis: {analysis}\n\nApply these changes: {edit_prompt}"
        
        if ctx: ctx.info("Performing smart edit...")
        
        response = await client.images.edit(
            model="gpt-image-1",
            image=image_base64,
            prompt=edit_prompt_full,
            response_format="b64_json"
        )
        
        b64_data = response.data[0].b64_json
        
        return {
            "success": True,
            "analysis": analysis,
            "edited_image": f"data:image/png;base64,{b64_data}",
            "source_file": str(image_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

@mcp.tool()
async def transform_image(
    image: str,
    operation: Literal["resize", "rotate", "flip_horizontal", "flip_vertical", "grayscale", "blur", "sharpen", "contrast", "brightness"],
    value: Optional[Union[int, float]] = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_path: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Apply basic image transformations using PIL.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Image to transform (file path, Google Drive URL/ID, or base64)
        operation: Type of transformation to apply
        value: Operation-specific value (degrees for rotate, factor for contrast/brightness)
        output_format: Output image format
        output_path: Optional absolute path to save result
        
    Returns:
        Transformed image as base64 or file path
    """
    if not PIL_AVAILABLE:
        return {"success": False, "error": "PIL/Pillow not available"}
    
    try:
        image_path = await get_file_path(image)
        
        with PILImage.open(image_path) as img:
            if operation == "resize" and value:
                if isinstance(value, tuple):
                    img = img.resize(value)
                else:
                    # Resize by percentage
                    width = int(img.width * (value / 100))
                    height = int(img.height * (value / 100))
                    img = img.resize((width, height))
                    
            elif operation == "rotate" and value:
                img = img.rotate(value, expand=True)
                
            elif operation == "flip_horizontal":
                img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
                
            elif operation == "flip_vertical":
                img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
                
            elif operation == "grayscale":
                img = ImageOps.grayscale(img)
                
            elif operation == "blur":
                from PIL import ImageFilter
                img = img.filter(ImageFilter.BLUR)
                
            elif operation == "sharpen":
                from PIL import ImageFilter
                img = img.filter(ImageFilter.SHARPEN)
                
            elif operation == "contrast" and value:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(value)
                
            elif operation == "brightness" and value:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(value)
            
            # Save result
            if output_path:
                save_path = Path(output_path)
                img.save(save_path, format=output_format.upper())
                return {"success": True, "file_path": str(save_path)}
            else:
                # Return as base64
                buffer = io.BytesIO()
                img.save(buffer, format=output_format.upper())
                buffer.seek(0)
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                return {
                    "success": True,
                    "image": f"data:image/{output_format};base64,{b64_data}",
                    "dimensions": f"{img.width}x{img.height}"
                }
                
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def batch_process(
    images: List[str],
    operation: Literal["analyze", "extract_text", "transform", "resize"],
    operation_params: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    """Process multiple images with the same operation.
    
    Supports both local files and Google Drive files.
    
    Args:
        images: List of images (file paths, Google Drive URLs/IDs, or base64)
        operation: Operation to perform on all images
        operation_params: Parameters for the operation
        
    Returns:
        Batch processing results for all images
    """
    results = []
    total_images = len(images)
    
    for i, image in enumerate(images):
        if ctx:
            await ctx.report_progress(i + 1, total_images, f"Processing image {i + 1}/{total_images}")
        
        try:
            if operation == "analyze":
                result = await analyze_image(image, **operation_params)
            elif operation == "extract_text":
                result = await extract_text(image, **operation_params)
            elif operation == "transform":
                result = await transform_image(image, **operation_params)
            elif operation == "resize":
                result = await transform_image(image, operation="resize", **operation_params)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
            results.append({
                "image": image,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "image": image,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_processed": total_images,
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "results": results
    }

@mcp.tool()
async def image_metadata(
    image: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Extract comprehensive metadata and properties from images.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Image to analyze (file path, Google Drive URL/ID, or base64)
        
    Returns:
        Detailed image metadata including EXIF, dimensions, format, etc.
    """
    if not PIL_AVAILABLE:
        return {"success": False, "error": "PIL/Pillow not available"}
    
    try:
        image_path = await get_file_path(image)
        
        with PILImage.open(image_path) as img:
            metadata = {
                "success": True,
                "file_path": str(image_path),
                "format": img.format,
                "mode": img.mode,
                "size": {
                    "width": img.width,
                    "height": img.height,
                    "total_pixels": img.width * img.height
                },
                "color_info": {
                    "bands": img.getbands(),
                    "palette": img.palette is not None
                }
            }
            
            # Get file system info
            stat = image_path.stat()
            metadata["file_info"] = {
                "size_bytes": stat.st_size,
                "size_kb": stat.st_size / 1024,
                "size_mb": stat.st_size / 1024 / 1024,
                "modified": stat.st_mtime,
                "created": stat.st_ctime
            }
            
            # Try to get EXIF data
            try:
                from PIL import ExifTags
                exif = img._getexif()
                if exif:
                    exif_data = {}
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = str(value)
                    metadata["exif"] = exif_data
            except:
                metadata["exif"] = "No EXIF data found"
            
            return metadata
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def describe_and_recreate(
    image: str,
    style_modification: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Analyze an image and recreate it with style modifications.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Source image (file path, Google Drive URL/ID, or base64)
        style_modification: Description of style changes to apply
        
    Returns:
        Original description and recreated image with style modifications
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    # First, describe the image
    describe_prompt = "Provide a detailed, technical description of this image including: subject matter, composition, lighting, color palette, style, mood, and any specific visual elements. Be precise and comprehensive."
    
    try:
        if ctx: ctx.info("Analyzing image for recreation...")
        
        description_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": describe_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        original_description = description_response.choices[0].message.content
        
        # Now recreate with style modifications
        recreate_prompt = f"Recreate this image: {original_description}\n\nApply these style modifications: {style_modification}"
        
        if ctx: ctx.info("Recreating image with style modifications...")
        
        response = await client.images.generate(
            model="gpt-image-1",
            prompt=recreate_prompt,
            n=1,
            response_format="b64_json"
        )
        
        b64_data = response.data[0].b64_json
        
        return {
            "success": True,
            "original_description": original_description,
            "style_modification": style_modification,
            "recreated_image": f"data:image/png;base64,{b64_data}",
            "source_file": str(image_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

@mcp.tool()
async def prompt_from_image(
    image: str,
    purpose: str = "accurate recreation",
    ctx: Context = None
) -> Dict[str, Any]:
    """Generate optimized prompts from images for AI image generation.
    
    Supports both local files and Google Drive files.
    
    Args:
        image: Source image (file path, Google Drive URL/ID, or base64)
        purpose: Purpose of the generated prompt (recreation, variation, improvement, etc.)
        
    Returns:
        Optimized prompt for AI image generation
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    prompt = f"Create an optimized text prompt for AI image generation that would recreate this image. Purpose: {purpose}. The prompt should be detailed, specific, and include: subject, composition, lighting, color palette, style, mood, and any technical details. Format it for best AI image generation results."
    
    try:
        if ctx: ctx.info("Generating optimized prompt from image...")
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                   ]
                }
            ],
            max_tokens=1000
        )
        
        generated_prompt = response.choices[0].message.content
        
        return {
            "success": True,
            "generated_prompt": generated_prompt,
            "purpose": purpose,
            "source_file": str(image_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Tool MCP Server")
    parser.add_argument("--transport", default="http", choices=["http", "stdio"], 
                       help="Transport method (http or stdio)")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                       help="Host to bind to (http mode only)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)),
                       help="Port to bind to (http mode only)")
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        # In stdio mode, disable all logging to avoid polluting stdout for JSON-RPC
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("__main__").setLevel(logging.CRITICAL)
        logging.getLogger("fastmcp").setLevel(logging.CRITICAL)
        mcp.run(transport="stdio")
    else:
        # HTTP mode - show configuration
        logger.info("=" * 50)
        logger.info("STARTING IMAGE TOOL MCP SERVER")
        logger.info("=" * 50)
        logger.info(f"Server configuration:")
        logger.info(f"  Host: {args.host}")
        logger.info(f"  Port: {args.port}")
        logger.info(f"  OpenAI configured: {_global_app_context.openai_client is not None if _global_app_context else 'Context not initialized'}")
        logger.info(f"  Temp directory: {_global_app_context.temp_dir if _global_app_context else 'Not set'}")
        logger.info("Available endpoints:")
        logger.info(f"  Health check: http://{args.host}:{args.port}/health")
        logger.info(f"  Root: http://{args.host}:{args.port}/")
        logger.info(f"  MCP: http://{args.host}:{args.port}/mcp/")
        logger.info("=" * 50)
        
        try:
            logger.info("Attempting to start FastMCP server...")
            mcp.run(transport="http", host=args.host, port=args.port)
        except Exception as e:
            logger.error(f"CRITICAL ERROR - Failed to start server: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise