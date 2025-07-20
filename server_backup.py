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
    """
    Generate images from text prompts using OpenAI's latest gpt-image-1 model.
    
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
    # Get application context from global variable
    global app_context
    if app_context is None:
        raise RuntimeError("Application context not initialized")
    
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
                # Return as Image object for MCP
                mime_type = f"image/{output_format}"
                images.append(Image(data=base64.b64decode(b64_data), format=output_format))
        
        # Return results
        if output_mode == "file":
            return file_paths if n > 1 else file_paths[0]
        else:
            return images if n > 1 else images[0]
            
    except Exception as e:
        ctx.error(f"Image generation failed: {str(e)}")
        raise ValueError(f"Failed to generate image: {str(e)}")


@mcp.tool()
async def edit_image(
    image: str,
    prompt: str,
    ctx: Context,
    mask: Optional[str] = None,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    n: int = 1,
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    output_mode: Literal["base64", "file"] = "base64", 
    file_path: Optional[str] = None
) -> Union[str, list[str]]:
    """
    Edit existing images using OpenAI's image editing capabilities.
    
    Args:
        image: Absolute file path or base64 string of image to edit
        prompt: Description of desired edits (max 32000 chars)
        mask: Optional mask image (absolute path or base64) - transparent areas will be edited
        model: Image model to use (only gpt-image-1 supported)
        n: Number of edited versions to generate (1-10)
        quality: Generation quality level
        size: Output image size
        output_mode: Return format (base64 or file)
        file_path: Output file path (required if output_mode='file')
        
    Returns:
        Edited image(s) as base64 data or file paths
    """
    # Get application context from global variable
    global app_context
    if app_context is None:
        raise RuntimeError("Application context not initialized")
    
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    # Validate inputs
    if len(prompt) > 32000:
        raise ValueError("Prompt must be 32000 characters or less")
        
    if n < 1 or n > 10:
        raise ValueError("Number of images must be between 1 and 10")
        
    if output_mode == "file" and not file_path:
        raise ValueError("file_path is required when output_mode is 'file'")
        
    if file_path and not os.path.isabs(file_path):
        raise ValueError("file_path must be an absolute path")
    
    # Validate and prepare image input
    if validate_image_path(image):
        # File path input
        image_file = open(image, "rb")
    elif is_base64_image(image):
        # Base64 input - convert to temporary file
        if image.startswith("data:image/"):
            # Data URL format
            header, b64_data = image.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        else:
            # Raw base64
            b64_data = image
            mime_type = "image/png"  # Default
            
        # Save to temp file
        temp_image_path = temp_dir / f"edit_input_{asyncio.current_task().get_name()}.png"
        await save_base64_image(b64_data, temp_image_path)
        image_file = open(temp_image_path, "rb")
    else:
        raise ValueError("image must be an absolute file path or base64 string")
    
    # Prepare mask if provided
    mask_file = None
    if mask:
        if validate_image_path(mask):
            mask_file = open(mask, "rb")
        elif is_base64_image(mask):
            if mask.startswith("data:image/"):
                header, b64_data = mask.split(",", 1)
            else:
                b64_data = mask
                
            temp_mask_path = temp_dir / f"edit_mask_{asyncio.current_task().get_name()}.png"
            await save_base64_image(b64_data, temp_mask_path)
            mask_file = open(temp_mask_path, "rb")
        else:
            raise ValueError("mask must be an absolute file path or base64 string")
    
    # Progress tracking
    if n > 1:
        await ctx.report_progress(0, n, f"Starting editing of {n} variations...")
    
    # Prepare API parameters
    params = {
        "image": image_file,
        "prompt": prompt,
        "model": model,
        "n": n
    }
    
    if mask_file:
        params["mask"] = mask_file
    if quality != "auto":
        params["quality"] = quality
    if size != "auto":
        params["size"] = size
    
    try:
        # Edit image
        ctx.info(f"Editing image with prompt: {prompt[:100]}...")
        response = await client.images.edit(**params)
        
        # Process results
        images = []
        file_paths = []
        
        for i, image_data in enumerate(response.data):
            if n > 1:
                await ctx.report_progress(i + 1, n, f"Processing edited image {i + 1}/{n}")
            
            b64_data = image_data.b64_json
            
            if output_mode == "file":
                # Save to file
                if n > 1:
                    path = Path(file_path)
                    save_path = path.parent / f"{path.stem}_edited_{i+1}{path.suffix}"
                else:
                    save_path = Path(file_path)
                
                await save_base64_image(b64_data, save_path)
                file_paths.append(str(save_path))
                ctx.info(f"Edited image saved to: {save_path}")
                
            else:
                # Return as Image object
                images.append(Image(data=base64.b64decode(b64_data), format="png"))
        
        # Return results
        if output_mode == "file":
            return file_paths if n > 1 else file_paths[0]
        else:
            return images if n > 1 else images[0]
            
    except Exception as e:
        ctx.error(f"Image editing failed: {str(e)}")
        raise ValueError(f"Failed to edit image: {str(e)}")
    finally:
        # Clean up file handles
        if 'image_file' in locals():
            image_file.close()
        if mask_file:
            mask_file.close()


@mcp.tool()
async def generate_variations(
    image: str,
    ctx: Context,
    n: int = 1,
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = "1024x1024",
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None
) -> Union[str, list[str]]:
    """
    Generate variations of an existing image.
    
    Args:
        image: Absolute file path or base64 string of source image
        n: Number of variations to generate (1-10)
        size: Output image size
        output_mode: Return format (base64 or file)
        file_path: Output file path (required if output_mode='file')
        
    Returns:
        Image variations as base64 data or file paths
    """
    # Get application context from global variable
    global app_context
    if app_context is None:
        raise RuntimeError("Application context not initialized")
    
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    # Validate inputs
    if n < 1 or n > 10:
        raise ValueError("Number of variations must be between 1 and 10")
        
    if output_mode == "file" and not file_path:
        raise ValueError("file_path is required when output_mode is 'file'")
        
    if file_path and not os.path.isabs(file_path):
        raise ValueError("file_path must be an absolute path")
    
    # Prepare image input
    if validate_image_path(image):
        image_file = open(image, "rb")
    elif is_base64_image(image):
        if image.startswith("data:image/"):
            header, b64_data = image.split(",", 1)
        else:
            b64_data = image
            
        temp_image_path = temp_dir / f"variation_input_{asyncio.current_task().get_name()}.png"
        await save_base64_image(b64_data, temp_image_path)
        image_file = open(temp_image_path, "rb")
    else:
        raise ValueError("image must be an absolute file path or base64 string")
    
    # Progress tracking
    if n > 1:
        await ctx.report_progress(0, n, f"Generating {n} variations...")
    
    try:
        # Generate variations
        ctx.info(f"Generating {n} variation(s) of the image...")
        response = await client.images.create_variation(
            image=image_file,
            n=n,
            size=size
        )
        
        # Process results
        images = []
        file_paths = []
        
        for i, image_data in enumerate(response.data):
            if n > 1:
                await ctx.report_progress(i + 1, n, f"Processing variation {i + 1}/{n}")
            
            b64_data = image_data.b64_json
            
            if output_mode == "file":
                if n > 1:
                    path = Path(file_path)
                    save_path = path.parent / f"{path.stem}_var_{i+1}{path.suffix}"
                else:
                    save_path = Path(file_path)
                
                await save_base64_image(b64_data, save_path)
                file_paths.append(str(save_path))
                ctx.info(f"Variation saved to: {save_path}")
                
            else:
                images.append(Image(data=base64.b64decode(b64_data), format="png"))
        
        # Return results
        if output_mode == "file":
            return file_paths if n > 1 else file_paths[0]
        else:
            return images if n > 1 else images[0]
            
    except Exception as e:
        ctx.error(f"Variation generation failed: {str(e)}")
        raise ValueError(f"Failed to generate variations: {str(e)}")
    finally:
        if 'image_file' in locals():
            image_file.close()
# =============================================================================
# IMAGE ANALYSIS TOOLS (OpenAI Vision API)
# =============================================================================

@mcp.tool()
async def analyze_image(
    image: str,
    ctx: Context,
    prompt: str = "Describe this image in detail, including objects, people, scenery, colors, mood, and any text visible.",
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    detail: Literal["low", "high", "auto"] = "auto"
) -> str:
    """
    Analyze an image using OpenAI's Vision API to extract detailed information.
    
    Args:
        image: Absolute file path or base64 string of image to analyze
        prompt: Analysis prompt (what to look for in the image)
        model: Vision model to use (gpt-4o, gpt-4o-mini, etc.)
        max_tokens: Maximum tokens in response
        detail: Image detail level for processing
        
    Returns:
        Detailed analysis of the image content
    """
    # Get application context
    global app_context
    if app_context is None:
        raise RuntimeError("Application context not initialized")
    
    client = app_context.openai_client
    
    # Prepare image for API
    if validate_image_path(image):
        # Load image file as base64
        base64_data, mime_type = await load_image_as_base64(image)
        image_url = f"data:{mime_type};base64,{base64_data}"
    elif is_base64_image(image):
        if image.startswith("data:image/"):
            image_url = image
        else:
            # Add data URL prefix
            image_url = f"data:image/png;base64,{image}"
    else:
        raise ValueError("image must be an absolute file path or base64 string")
    
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


@mcp.tool()
async def extract_text(
    image: str,
    ctx: Context,
    model: str = "gpt-4o",
    language: str = "auto",
    preserve_formatting: bool = True
) -> str:
    """
    Extract text from images using OpenAI's Vision API (OCR functionality).
    
    Args:
        image: Absolute file path or base64 string of image containing text
        model: Vision model to use
        language: Expected language of text (auto-detect if 'auto')
        preserve_formatting: Whether to maintain original text formatting
        
    Returns:
        Extracted text from the image
    """
    # Prepare OCR prompt
    if preserve_formatting:
        ocr_prompt = f"""Extract all text from this image. 
        Preserve the original formatting, spacing, and structure as much as possible.
        If the text appears to be in a specific language other than English, maintain that language.
        Return only the extracted text without additional commentary."""
    else:
        ocr_prompt = f"""Extract all text from this image and return it as clean, readable text.
        Remove any formatting artifacts but preserve the meaning and content.
        Return only the extracted text without additional commentary."""
    
    if language != "auto":
        ocr_prompt += f"\nThe text is expected to be in {language}."
    
    # Use the analyze_image function with OCR-specific prompt
    ctx.info("Extracting text from image using Vision API...")
    extracted_text = await analyze_image(
        image=image,
        prompt=ocr_prompt,
        model=model,
        max_tokens=2000,
        detail="high",
        ctx=ctx
    )
    
    ctx.info("Text extraction completed")
    return extracted_text


@mcp.tool()
async def compare_images(
    image1: str,
    image2: str,
    ctx: Context,
    comparison_prompt: str = "Compare these two images and describe the key differences, similarities, and any notable changes between them.",
    model: str = "gpt-4o",
    max_tokens: int = 1500
) -> str:
    """
    Compare two images and analyze their differences and similarities.
    
    Args:
        image1: First image (absolute file path or base64 string)
        image2: Second image (absolute file path or base64 string)
        comparison_prompt: What to compare between the images
        model: Vision model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        Detailed comparison of the two images
    """
    # Get application context
    global app_context
    if app_context is None:
        raise RuntimeError("Application context not initialized")
    
    client = app_context.openai_client
    
    # Prepare both images
    images_data = []
    for i, image in enumerate([image1, image2], 1):
        if validate_image_path(image):
            base64_data, mime_type = await load_image_as_base64(image)
            image_url = f"data:{mime_type};base64,{base64_data}"
        elif is_base64_image(image):
            if image.startswith("data:image/"):
                image_url = image
            else:
                image_url = f"data:image/png;base64,{image}"
        else:
            raise ValueError(f"image{i} must be an absolute file path or base64 string")
        
        images_data.append({
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"}
        })
    
    try:
        ctx.info("Comparing images using Vision API...")
        
        # Create message content with both images
        content = [{"type": "text", "text": comparison_prompt}]
        content.extend(images_data)
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens
        )
        
        comparison = response.choices[0].message.content
        ctx.info("Image comparison completed successfully")
        return comparison
        
    except Exception as e:
        ctx.error(f"Image comparison failed: {str(e)}")
        raise ValueError(f"Failed to compare images: {str(e)}")


# =============================================================================
# HYBRID INTELLIGENCE TOOLS (Vision + Images APIs)
# =============================================================================

@mcp.tool()
async def smart_edit(
    image: str,
    edit_request: str,
    ctx: Context,
    analyze_first: bool = True,
    model: str = "gpt-4o",
    edit_model: Literal["gpt-image-1"] = "gpt-image-1",
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None
) -> str:
    """
    Intelligently edit an image by first analyzing it, then applying targeted edits.
    
    Args:
        image: Image to edit (absolute file path or base64 string)
        edit_request: Description of desired edits
        analyze_first: Whether to analyze the image before editing
        model: Vision model for analysis
        edit_model: Image model for editing
        output_mode: Return format (base64 or file)
        file_path: Output file path (required if output_mode='file')
        
    Returns:
        Intelligently edited image
    """
    ctx.info("Starting smart edit process...")
    
    # Step 1: Analyze the image if requested
    if analyze_first:
        ctx.info("Analyzing image to understand content...")
        analysis = await analyze_image(
            image=image,
            prompt=f"""Analyze this image in detail, focusing on elements relevant to this edit request: "{edit_request}"
            
            Describe:
            1. Current state of elements that will be edited
            2. Overall composition and style
            3. Colors, lighting, and mood
            4. Any potential challenges for the requested edit
            
            This analysis will be used to create a better edit prompt.""",
            model=model,
            max_tokens=800,
            ctx=ctx
        )
        
        # Step 2: Create enhanced edit prompt based on analysis
        enhanced_prompt = f"""Based on this image analysis: {analysis}
        
        Please {edit_request}
        
        Maintain the overall style, lighting, and composition while making the requested changes naturally and seamlessly."""
        
        ctx.info("Generated enhanced edit prompt based on image analysis")
    else:
        enhanced_prompt = edit_request
    
    # Step 3: Apply the edit
    ctx.info("Applying intelligent edits to the image...")
    result = await edit_image(
        image=image,
        prompt=enhanced_prompt,
        model=edit_model,
        n=1,
        quality="high",
        output_mode=output_mode,
        file_path=file_path,
        ctx=ctx
    )
    
    ctx.info("Smart edit completed successfully")
    return result


@mcp.tool()
async def describe_and_recreate(
    image: str,
    ctx: Context,
    style_modification: str = "",
    model: str = "gpt-4o",
    generation_model: Literal["gpt-image-1"] = "gpt-image-1",
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None
) -> str:
    """
    Analyze an existing image and recreate it with optional style modifications.
    
    Args:
        image: Source image to analyze and recreate
        style_modification: Optional style changes to apply
        model: Vision model for analysis
        generation_model: Image generation model
        output_mode: Return format (base64 or file)
        file_path: Output file path (required if output_mode='file')
        
    Returns:
        Recreated image based on analysis
    """
    ctx.info("Starting describe and recreate process...")
    
    # Step 1: Analyze the image for recreation
    ctx.info("Analyzing image for detailed recreation...")
    description = await analyze_image(
        image=image,
        prompt=f"""Create a detailed description of this image that would allow someone to recreate it accurately. Include:
        
        1. Subject matter and main elements
        2. Composition and layout
        3. Colors, lighting, and shadows
        4. Style and artistic approach
        5. Mood and atmosphere
        6. Background and setting details
        7. Any text or specific details
        
        Format this as a clear, detailed prompt for image generation.""",
        model=model,
        max_tokens=1200,
        ctx=ctx
    )
    
    # Step 2: Apply style modifications if requested
    if style_modification:
        recreation_prompt = f"{description}\n\nStyle modification: {style_modification}"
        ctx.info(f"Applying style modification: {style_modification}")
    else:
        recreation_prompt = description
    
    # Step 3: Generate the new image
    ctx.info("Generating recreated image...")
    result = await create_image(
        prompt=recreation_prompt,
        model=generation_model,
        quality="high",
        output_mode=output_mode,
        file_path=file_path,
        ctx=ctx
    )
    
    ctx.info("Image recreation completed successfully")
    return result


@mcp.tool()
async def prompt_from_image(
    image: str,
    ctx: Context,
    style_focus: Literal["general", "artistic", "photographic", "technical"] = "general",
    model: str = "gpt-4o"
) -> str:
    """
    Analyze an image and generate an optimized prompt for recreating similar images.
    
    Args:
        image: Image to analyze for prompt generation
        style_focus: Type of prompt to generate
        model: Vision model to use
        
    Returns:
        Optimized prompt for image generation
    """
    # Create style-specific analysis prompts
    style_prompts = {
        "general": """Analyze this image and create a detailed prompt that could be used to generate similar images. 
        Focus on the key visual elements, composition, subject matter, and overall aesthetic.""",
        
        "artistic": """Analyze this image from an artistic perspective and create a prompt focusing on:
        - Art style and technique
        - Color palette and composition
        - Artistic movement or influence
        - Mood and emotional impact
        Create a prompt suitable for artistic image generation.""",
        
        "photographic": """Analyze this image from a photography perspective and create a prompt including:
        - Camera settings and technique
        - Lighting conditions and quality
        - Composition and framing
        - Subject and background details
        Create a prompt for photorealistic image generation.""",
        
        "technical": """Analyze this image and create a highly detailed technical prompt including:
        - Precise color descriptions
        - Exact positioning and proportions
        - Lighting specifications
        - Material and texture details
        Create a comprehensive prompt for accurate reproduction."""
    }
    
    analysis_prompt = style_prompts.get(style_focus, style_prompts["general"])
    
    ctx.info(f"Generating {style_focus} prompt from image analysis...")
    
    # Analyze and generate prompt
    optimized_prompt = await analyze_image(
        image=image,
        prompt=analysis_prompt,
        model=model,
        max_tokens=1000,
        ctx=ctx
    )
    
    ctx.info("Prompt generation completed")
    return optimized_prompt


# =============================================================================
# UTILITY TOOLS (Built-in Processing)
# =============================================================================

@mcp.tool()
async def image_metadata(
    image: str,
    ctx: Context
) -> dict[str, Any]:
    """
    Extract metadata and properties from an image file.
    
    Args:
        image: Absolute file path to image
        
    Returns:
        Dictionary containing image metadata and properties
    """
    if not validate_image_path(image):
        raise ValueError("image must be a valid absolute file path")
    
    try:
        ctx.info("Extracting image metadata...")
        
        # Get file stats
        file_path = Path(image)
        file_stats = file_path.stat()
        
        # Open image with PIL
        with PILImage.open(image) as img:
            # Basic image info
            metadata = {
                "filename": file_path.name,
                "file_size_bytes": file_stats.st_size,
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "dimensions": {
                    "width": img.width,
                    "height": img.height,
                    "aspect_ratio": round(img.width / img.height, 2)
                },
                "format": img.format,
                "mode": img.mode,
                "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                "created": file_stats.st_ctime,
                "modified": file_stats.st_mtime
            }
            
            # EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                exif_data = img._getexif()
                if exif_data:
                    metadata["exif"] = {}
                    # Add common EXIF tags
                    exif_tags = {
                        256: "ImageWidth",
                        257: "ImageLength", 
                        272: "Make",
                        273: "Model",
                        306: "DateTime",
                        34665: "ExifOffset"
                    }
                    for tag_id, tag_name in exif_tags.items():
                        if tag_id in exif_data:
                            metadata["exif"][tag_name] = exif_data[tag_id]
            
            # Color analysis
            if img.mode == "RGB":
                # Get dominant colors (simplified)
                img_small = img.resize((50, 50))
                colors = img_small.getcolors(maxcolors=256)
                if colors:
                    dominant_color = max(colors, key=lambda x: x[0])[1]
                    metadata["dominant_color"] = {
                        "r": dominant_color[0],
                        "g": dominant_color[1], 
                        "b": dominant_color[2]
                    }
        
        ctx.info("Metadata extraction completed")
        return metadata
        
    except Exception as e:
        ctx.error(f"Metadata extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract metadata: {str(e)}")


@mcp.tool()
async def transform_image(
    image: str,
    operation: Literal["resize", "crop", "rotate", "flip", "convert"],
    parameters: dict[str, Any],
    output_path: str,
    ctx: Context
) -> str:
    """
    Apply basic transformations to an image using PIL.
    
    Args:
        image: Absolute file path to source image
        operation: Type of transformation to apply
        parameters: Operation-specific parameters
        output_path: Absolute path for output file
        
    Returns:
        Path to transformed image
    """
    if not validate_image_path(image):
        raise ValueError("image must be a valid absolute file path")
        
    if not os.path.isabs(output_path):
        raise ValueError("output_path must be an absolute path")
    
    try:
        ctx.info(f"Applying {operation} transformation...")
        
        with PILImage.open(image) as img:
            if operation == "resize":
                width = parameters.get("width")
                height = parameters.get("height")
                maintain_aspect = parameters.get("maintain_aspect", True)
                
                if maintain_aspect and (width or height):
                    img.thumbnail((width or img.width, height or img.height), PILImage.Resampling.LANCZOS)
                    result = img.copy()
                else:
                    result = img.resize((width, height), PILImage.Resampling.LANCZOS)
                    
            elif operation == "crop":
                left = parameters.get("left", 0)
                top = parameters.get("top", 0) 
                right = parameters.get("right", img.width)
                bottom = parameters.get("bottom", img.height)
                result = img.crop((left, top, right, bottom))
                
            elif operation == "rotate":
                angle = parameters.get("angle", 0)
                expand = parameters.get("expand", True)
                result = img.rotate(angle, expand=expand)
                
            elif operation == "flip":
                direction = parameters.get("direction", "horizontal")
                if direction == "horizontal":
                    result = img.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
                elif direction == "vertical":
                    result = img.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM)
                else:
                    raise ValueError("direction must be 'horizontal' or 'vertical'")
                    
            elif operation == "convert":
                mode = parameters.get("mode", "RGB")
                result = img.convert(mode)
                
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Save result
            output_format = parameters.get("format", "PNG")
            quality = parameters.get("quality", 95)
            
            if output_format.upper() == "JPEG":
                result.save(output_path, format="JPEG", quality=quality, optimize=True)
            else:
                result.save(output_path, format=output_format.upper())
            
            ctx.info(f"Image transformation completed: {output_path}")
            return output_path
            
    except Exception as e:
        ctx.error(f"Image transformation failed: {str(e)}")
        raise ValueError(f"Failed to transform image: {str(e)}")


@mcp.tool()
async def batch_process(
    images: list[str],
    operation: str,
    operation_params: dict[str, Any],
    output_directory: str,
    ctx: Context
) -> list[str]:
    """
    Process multiple images with the same operation.
    
    Args:
        images: List of absolute file paths to images
        operation: Name of operation to apply (matches other tool names)
        operation_params: Parameters for the operation
        output_directory: Directory to save processed images
        
    Returns:
        List of output file paths
    """
    if not os.path.isabs(output_directory):
        raise ValueError("output_directory must be an absolute path")
    
    # Create output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    results = []
    total_images = len(images)
    
    ctx.info(f"Starting batch processing of {total_images} images with operation '{operation}'")
    
    for i, image_path in enumerate(images):
        try:
            await ctx.report_progress(i, total_images, f"Processing image {i+1}/{total_images}")
            
            # Generate output path
            input_path = Path(image_path)
            output_path = Path(output_directory) / f"{input_path.stem}_processed{input_path.suffix}"
            
            # Apply operation based on type
            if operation == "analyze_image":
                # For analysis, save results as text file
                result = await analyze_image(image=image_path, **operation_params, ctx=ctx)
                text_output = output_path.with_suffix('.txt')
                async with aiofiles.open(text_output, 'w') as f:
                    await f.write(result)
                results.append(str(text_output))
                
            elif operation == "transform_image":
                result = await transform_image(
                    image=image_path,
                    output_path=str(output_path),
                    **operation_params,
                    ctx=ctx
                )
                results.append(result)
                
            elif operation == "create_image":
                # For generation, use image filename in prompt
                prompt = operation_params.get("prompt", "").replace("{filename}", input_path.stem)
                result = await create_image(
                    prompt=prompt,
                    output_mode="file",
                    file_path=str(output_path),
                    **{k: v for k, v in operation_params.items() if k != "prompt"},
                    ctx=ctx
                )
                results.append(result)
                
            else:
                ctx.error(f"Unsupported batch operation: {operation}")
                continue
                
        except Exception as e:
            ctx.error(f"Failed to process {image_path}: {str(e)}")
            continue
    
    await ctx.report_progress(total_images, total_images, "Batch processing completed")
    ctx.info(f"Batch processing completed. Processed {len(results)}/{total_images} images successfully")
    
    return results


# =============================================================================
# SERVER STARTUP
# =============================================================================
# For Railway deployment - expose ASGI app
app = mcp.http_app()

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info("Starting Image Tool MCP Server for Railway deployment...")
    logger.info("Image processing tools available:")
    logger.info("- Image generation with OpenAI gpt-image-1")
    logger.info("- Image editing with mask support")
    logger.info("- Image analysis with GPT-4 Vision")
    logger.info("- OCR and text extraction")
    logger.info("- Batch processing with progress tracking")
    logger.info("- Smart editing workflows")
    logger.info(f"Server will run on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)