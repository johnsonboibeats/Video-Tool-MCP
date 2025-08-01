# Image Tool MCP Server

A comprehensive Model Context Protocol (MCP) server for image processing tasks, deployed on Railway with Claude Desktop compatibility.

## Features

- **Image Generation**: Create images from text prompts using OpenAI's gpt-image-1 model
- **Image Analysis**: Analyze images with detailed descriptions using GPT-4o
- **Image Editing**: Edit images with text prompts and optional masks
- **Image Variations**: Generate variations of existing images
- **Text Extraction**: Extract text from images using OCR
- **Image Comparison**: Compare two images for similarities and differences
- **Smart Editing**: Advanced editing with analysis and modification prompts
- **Image Transformations**: Resize, rotate, flip, and apply filters
- **Batch Processing**: Process multiple images with various operations
- **Image Metadata**: Extract technical information from images
- **Prompt Generation**: Generate optimized prompts from images
- **Local File Support**: Process files from local file system
- **Base64 Support**: Handle base64 encoded image data

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for image generation and analysis

### Optional
- `PORT`: Server port (default: 8080)
- `ALLOWED_ORIGINS`: CORS allowed origins (default: claude.ai domains)
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting (default: 100)

## Local Development

### Prerequisites
- Python 3.8+
- OpenAI API key with image generation access

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
4. Run the server: `python server.py`

## Railway Deployment

### Environment Variables
```json
{
  "OPENAI_API_KEY": "your-openai-api-key"
}
```

### Deploy
```bash
railway up
```

## Usage Examples

### Image Generation
```json
{
  "prompt": "A serene mountain landscape at sunset",
  "size": "1024x1024",
  "output_format": "png"
}
```

### Image Analysis
```json
{
  "image": "/path/to/image.jpg",
  "prompt": "Describe this image in detail"
}
```

### Image Editing
```json
{
  "image": "/path/to/image.jpg",
  "prompt": "Add a red car to the scene",
  "output_format": "png"
}
```

### File Input Types

The server supports the following input types:

#### Local Files
```json
{
  "image": "/absolute/path/to/image.jpg"
}
```

#### Base64 Data
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with server info
- `POST /mcp/` - MCP protocol endpoint

## Available Tools

1. **create_image** - Generate images from text prompts
2. **analyze_image** - Analyze images with detailed descriptions
3. **edit_image** - Edit images with text prompts
4. **generate_variations** - Create variations of existing images
5. **extract_text** - Extract text from images using OCR
6. **compare_images** - Compare two images
7. **smart_edit** - Advanced editing with analysis
8. **transform_image** - Apply transformations (resize, rotate, etc.)
9. **batch_process** - Process multiple images
10. **image_metadata** - Extract image metadata
11. **describe_and_recreate** - Describe and recreate images
12. **prompt_from_image** - Generate prompts from images

## Error Handling

The server includes comprehensive error handling for:
- Invalid file paths
- Missing API keys
- Rate limiting
- Network timeouts
- Invalid image formats

## Security Features

- Path traversal protection
- Rate limiting
- CORS configuration
- Input validation
- Secure file handling

## License

MIT License
