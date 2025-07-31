#!/usr/bin/env python3
"""
Minimal Image Tool MCP Server for Railway Testing
"""

import os
import time
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

print("🔧 Starting minimal MCP server...")

# Initialize FastMCP
mcp = FastMCP("Image Tool MCP Server")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "server": "Image Tool MCP Server",
        "version": "minimal-test"
    })

@mcp.custom_route("/", methods=["GET"])
async def root(request: Request):
    """Root endpoint"""
    return JSONResponse({
        "message": "Image Tool MCP Server is running",
        "status": "ok"
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    mcp.run(transport="http", host=host, port=port)