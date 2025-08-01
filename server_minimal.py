#!/usr/bin/env python3
"""
Minimal FastMCP Server for Claude Desktop
Following the latest FastMCP 2.x patterns
"""

import os
import logging
from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server instance
mcp = FastMCP("Image Tool MCP Server")

# Add a simple test tool
@mcp.tool()
async def test_connection() -> str:
    """Test if the MCP server is working"""
    return "MCP server is connected and working!"

# Add image generation tool (simplified)
@mcp.tool()
async def generate_image(prompt: str) -> dict:
    """Generate an image from a text prompt
    
    Args:
        prompt: Description of the image to generate
        
    Returns:
        Status message (actual generation requires OpenAI API key)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "message": "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        }
    
    # For testing, just return a success message
    return {
        "status": "success", 
        "message": f"Would generate image with prompt: {prompt}",
        "api_configured": True
    }

# Run the server
if __name__ == "__main__":
    import sys
    
    # Check command line args
    transport = "http"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"
    
    logger.info(f"Starting Image Tool MCP Server on {host}:{port}")
    logger.info(f"Transport: {transport}")
    logger.info("Available tools: test_connection, generate_image")
    
    # Run with specified transport
    mcp.run(transport=transport, host=host, port=port)
