#!/usr/bin/env python3
"""
Image Tool MCP Server with OAuth 2.0 for Claude Web Integration

A comprehensive Model Context Protocol server for image processing tasks using:
- OpenAI gpt-image-1 for image generation and editing
- GPT-4 Vision for image analysis and OCR
- PIL for image manipulation and processing
- Google Drive API for cloud file access
- OAuth 2.0 authentication for Claude Web integration

Features:
- High-quality image generation with customizable parameters
- Advanced image editing with mask support and smart editing
- Intelligent image analysis and OCR text extraction
- Batch processing with progress tracking
- Google Drive integration for cloud file access
- Image format conversion and optimization

Tools Available:
1. create-image - Generate images from text prompts
2. edit-image - Edit existing images with prompts and masks
3. analyze-image - Analyze images with AI vision
4. extract-text - OCR text extraction from images
5. compare-images - Compare two images and analyze differences
6. smart-edit - Intelligent image editing with analysis
7. generate-variations - Create variations of existing images
8. transform-image - Basic image transformations
9. batch-process - Process multiple images with same operation
10. image-metadata - Extract image metadata and properties
11. describe-and-recreate - Analyze and recreate images with style modifications
12. prompt-from-image - Generate optimized prompts from images
"""

# Suppress known warnings for cleaner deployment logs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*not JSON serializable.*", category=UserWarning)

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
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Core imports
import aiofiles
from openai import AsyncOpenAI, AsyncAzureOpenAI
from fastmcp import FastMCP, Context
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from dotenv import load_dotenv

# Google Drive integration with graceful fallbacks
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    logger.warning("Google Drive libraries not available - Google Drive integration disabled")

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
    drive_service: Optional[Any] = None

# Global app context for FastMCP tools
_global_app_context: Optional[AppContext] = None

def get_app_context() -> AppContext:
    """Get application context from global reference"""
    if _global_app_context is not None:
        return _global_app_context
    raise RuntimeError("Application context not initialized")

def setup_google_drive_service():
    """Setup Google Drive service with authentication"""
    if not GOOGLE_DRIVE_AVAILABLE:
        return None
        
    try:
        # Try service account credentials first
        if os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
            try:
                service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                service = build('drive', 'v3', credentials=credentials)
                logger.info("Google Drive service initialized with service account")
                return service
            except Exception as e:
                logger.error(f"Failed to setup Google Drive with service account: {e}")
        
        # Try service account file
        if os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"):
            service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_file,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                service = build('drive', 'v3', credentials=credentials)
                logger.info(f"Google Drive service initialized with file: {service_account_file}")
                return service
            except Exception as e:
                logger.error(f"Failed to setup Google Drive with file: {e}")
        
        logger.warning("No Google Drive credentials found. Integration disabled.")
        return None
        
    except Exception as e:
        logger.error(f"Failed to setup Google Drive service: {e}")
        return None

def extract_google_drive_id(file_input: str) -> Optional[str]:
    """Extract Google Drive file ID from various input formats"""
    # Direct file ID
    if len(file_input) == 33 and file_input.isalnum():
        return file_input
    
    # Google Drive URL patterns
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        r'docs\.google\.com/.*?/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_input)
        if match:
            return match.group(1)
    
    return None

async def download_from_google_drive(file_id: str, drive_service) -> str:
    """Download file from Google Drive to temporary directory"""
    if not drive_service:
        raise RuntimeError("Google Drive service not available")
    
    try:
        # Get file metadata
        file_metadata = drive_service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name', f'drive_file_{file_id}')
        
        # Download file content
        file_content = drive_service.files().get_media(fileId=file_id).execute()
        
        # Save to temp directory
        app_context = get_app_context()
        temp_path = app_context.temp_dir / file_name
        
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Downloaded Google Drive file: {file_name}")
        return str(temp_path)
        
    except Exception as e:
        logger.error(f"Failed to download file from Google Drive: {e}")
        raise RuntimeError(f"Failed to download file from Google Drive: {e}")

async def get_file_path(file_input: str) -> str:
    """Universal file handler for local files and Google Drive files"""
    app_context = get_app_context()
    
    # Check if it's a Google Drive file
    drive_id = extract_google_drive_id(file_input)
    if drive_id:
        if not app_context.drive_service:
            raise RuntimeError("Google Drive service not available")
        return await download_from_google_drive(drive_id, app_context.drive_service)
    
    # Handle local file path
    if os.path.isfile(file_input):
        return file_input
    
    # Handle base64 data
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
    
    raise ValueError(f"File not found or invalid input: {file_input}")

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
        else:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Initialized OpenAI client")
        
        # Setup temp directory
        temp_dir = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir())) / "image_tool_mcp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory: {temp_dir}")
        
        # Setup Google Drive service
        drive_service = setup_google_drive_service()
        if drive_service:
            logger.info("Google Drive service initialized successfully")
        else:
            logger.info("Google Drive service not available - continuing without it")
        
        # Create HTTP client
        http_client = httpx.AsyncClient() if HTTPX_AVAILABLE else None
        
        # Create context
        context = AppContext(
            openai_client=client,
            temp_dir=temp_dir,
            http_client=http_client,
            drive_service=drive_service
        )
        
        # Set global context for FastMCP tools
        global _global_app_context
        _global_app_context = context
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to initialize app context: {e}")
        raise

# Initialize application context at startup
initialize_app_context()

# Setup authentication for FastMCP server
def setup_authentication():
    """Setup Bearer authentication for FastMCP server"""
    try:
        # For production, you would use your OAuth provider's JWKS URI
        # For development, we'll generate a key pair
        if os.getenv("FASTMCP_AUTH_JWKS_URI"):
            # Production: Use OAuth provider
            auth = BearerAuthProvider(
                jwks_uri=os.getenv("FASTMCP_AUTH_JWKS_URI"),
                issuer=os.getenv("FASTMCP_AUTH_ISSUER"),
                audience=os.getenv("FASTMCP_AUTH_AUDIENCE", "image-tool-mcp")
            )
            logger.info("Using production OAuth authentication")
        else:
            # Development: Generate key pair
            key_pair = RSAKeyPair.generate()
            
            # Save the key pair for development (in production, this would be managed externally)
            if not os.path.exists(".dev-auth"):
                os.makedirs(".dev-auth")
            
            with open(".dev-auth/public_key.pem", "w") as f:
                f.write(key_pair.public_key)
            
            # Generate a test token
            access_token = key_pair.create_token(
                audience="image-tool-mcp",
                scopes=["image:read", "image:write", "drive:read"]
            )
            
            # Save test token for development
            with open(".dev-auth/test_token.txt", "w") as f:
                f.write(access_token)
            
            auth = BearerAuthProvider(
                public_key=key_pair.public_key,
                audience="image-tool-mcp"
            )
            
            logger.info("Using development authentication")
            logger.info(f"Development token saved to .dev-auth/test_token.txt")
            
        return auth
    except Exception as e:
        logger.warning(f"Authentication setup failed, running without auth: {e}")
        return None

# Create FastMCP server instance with proper configuration
auth_provider = setup_authentication()

mcp = FastMCP(
    name="Image Tool MCP",
    auth=auth_provider
)

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
    
    # Get file path (handles Google Drive files, local files, and base64)
    file_path = await get_file_path(image)
    
    # Prepare image for API
    if validate_image_path(file_path):
        # Load image file as base64
        base64_data, mime_type = await load_image_as_base64(file_path)
        image_url = f"data:{mime_type};base64,{base64_data}"
    elif is_base64_image(image):
        if image.startswith("data:image/"):
            image_url = image
        else:
            # Add data URL prefix
            image_url = f"data:image/png;base64,{image}"
    else:
        raise ValueError("image must be an absolute file path, Google Drive URL/ID, or base64 string")
    
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
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info("Starting Image Tool MCP Server for Railway deployment...")
    logger.info("Available image processing tools:")
    logger.info("- Generate images from text prompts")
    logger.info("- Analyze images with AI vision")
    logger.info(f"Server will run on port {port}")
    
    # Use FastMCP's built-in HTTP server
    mcp.run(transport="http", host="0.0.0.0", port=port)