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