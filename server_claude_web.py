#!/usr/bin/env python3
"""
Image Tool MCP Server - Optimized for Claude Web Interface
Supports SSE transport for remote deployment
"""

import os
import sys
import logging
from fastmcp import FastMCP
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Import required packages
try:
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = None
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    openai_client = AsyncOpenAI(api_key=api_key)
    logger.info("✓ OpenAI client initialized")
else:
    logger.warning("⚠ No OPENAI_API_KEY found - tools will have limited functionality")

# Create FastMCP server
mcp = FastMCP(
    name="Image Tool MCP",
    description="Generate and analyze images with OpenAI"
)

# Test tool to verify connection
@mcp.tool()
async def test_connection() -> Dict[str, Any]:
    """Test if the MCP server is working properly."""
    return {
        "status": "connected",
        "server": "Image Tool MCP",
        "version": "1.0.0",
        "openai_configured": openai_client is not None,
        "transport": "sse",
        "message": "Server is ready to generate and analyze images!"
    }

# Image generation tool
@mcp.tool()
async def create_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard"
) -> Dict[str, Any]:
    """Generate an image from text using DALL-E 3.
    
    Args:
        prompt: Text description of the image to generate
        size: Image dimensions (1024x1024, 1792x1024, 1024x1792)
        quality: Image quality (standard or hd)
    
    Returns:
        Generated image URL and metadata
    """
    if not openai_client:
        return {
            "error": "OpenAI API key not configured",
            "message": "Please set OPENAI_API_KEY environment variable in Railway"
        }
    
    try:
        logger.info(f"Generating image: {prompt[:50]}...")
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1
        )
        
        result = {
            "success": True,
            "url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt,
            "size": size,
            "quality": quality
        }
        logger.info("✓ Image generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"✗ Image generation failed: {e}")
        return {
            "error": "Image generation failed",
            "message": str(e)
        }

# Image analysis tool
@mcp.tool()
async def analyze_image(
    image_url: str,
    question: str = "Describe this image in detail"
) -> Dict[str, Any]:
    """Analyze an image using GPT-4 Vision.
    
    Args:
        image_url: URL of the image to analyze
        question: What to analyze about the image
    
    Returns:
        Image analysis results
    """
    if not openai_client:
        return {
            "error": "OpenAI API key not configured",
            "message": "Please set OPENAI_API_KEY environment variable in Railway"
        }
    
    try:
        logger.info(f"Analyzing image: {image_url[:50]}...")
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            max_tokens=1000
        )
        
        result = {
            "success": True,
            "analysis": response.choices[0].message.content,
            "model": "gpt-4o",
            "image_url": image_url
        }
        logger.info("✓ Image analyzed successfully")
        return result
        
    except Exception as e:
        logger.error(f"✗ Image analysis failed: {e}")
        return {
            "error": "Image analysis failed",
            "message": str(e)
        }

# Main entry point
if __name__ == "__main__":
    # Get configuration
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"  # Required for Railway
    
    # Determine transport based on command line or environment
    transport = "sse"  # Default to SSE for Claude web interface
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    
    # Log startup information
    logger.info("="*60)
    logger.info("🚀 Image Tool MCP Server Starting")
    logger.info("="*60)
    logger.info(f"📍 Host: {host}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🚊 Transport: {transport}")
    logger.info(f"🔑 OpenAI: {'✓ Configured' if openai_client else '✗ Not configured'}")
    logger.info(f"🛠️  Tools: test_connection, create_image, analyze_image")
    logger.info("="*60)
    
    try:
        # Run the server with SSE transport
        mcp.run(transport=transport, host=host, port=port)
    except Exception as e:
        logger.error(f"💥 Server failed to start: {e}")
        sys.exit(1)
