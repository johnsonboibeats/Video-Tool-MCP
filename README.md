# Image Tool MCP Server

A comprehensive Model Context Protocol (MCP) server for image processing tasks, deployed on Railway with OAuth authentication for Google Drive integration and Claude Desktop compatibility.

## Features

- **Image Generation**: Generate images from text prompts using OpenAI's gpt-image-1
- **Image Analysis**: Analyze images using OpenAI's Vision API for detailed descriptions and OCR
- **Google Drive Integration**: Process files directly from Google Drive
- **Multiple Formats**: Support for various image formats (PNG, JPG, WebP, etc.)

## Environment Variables

### Required
- `OPENAI_API_KEY`: OpenAI API key for image generation and analysis services

### Google Drive (Choose One)
- `GOOGLE_SERVICE_ACCOUNT_JSON`: Base64-encoded service account JSON (recommended for Railway)
- `GOOGLE_SERVICE_ACCOUNT_FILE`: Path to service account JSON file
- `GOOGLE_OAUTH_CREDENTIALS_FILE`: Path to OAuth credentials file

### Optional
- `MCP_TRANSPORT`: Transport type (`http`, `stdio`, `sse`) - defaults to `http`
- `HOST`: Server host - defaults to `0.0.0.0`
- `PORT`: Server port - defaults to `8080`
- `MCP_TEMP_DIR`: Temporary directory for file processing
- `RAILWAY_PUBLIC_DOMAIN`: Auto-detected for Railway deployments
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins - defaults to `https://claude.ai,https://web.claude.ai`
- `MAX_REQUESTS_PER_MINUTE`: Rate limit for API requests - defaults to `100`

## Deployment

### Railway Deployment
1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Server will auto-detect Railway environment
4. Access at: `https://your-app.up.railway.app`

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key-here"
export GOOGLE_SERVICE_ACCOUNT_JSON="base64-encoded-json"

# Run server
python server.py
```

## Claude Desktop Integration

### Configuration
Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "image-tool": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "GOOGLE_SERVICE_ACCOUNT_JSON": "your-base64-json"
      }
    }
  }
}
```

### For Remote Server (Railway)
```json
{
  "mcpServers": {
    "image-tool": {
      "url": "https://your-app.up.railway.app/mcp/"
    }
  }
}
```

**Claude Web Configuration:**
```json
{
  "type": "url",
  "url": "https://your-app.up.railway.app/mcp/"
}
```

## API Endpoints

### Core MCP
- `GET /mcp/` - MCP protocol endpoint
- `GET /health` - Health check and server status

### OAuth 2.0 Discovery
- `GET /.well-known/oauth-authorization-server` - OAuth server metadata
- `GET /.well-known/oauth-protected-resource` - Resource server metadata
- `GET /.well-known/jwks.json` - JSON Web Key Set

### OAuth Flow
- `GET/POST /oauth/authorize` - Authorization endpoint
- `POST /oauth/token` - Token endpoint
- `GET /oauth/userinfo` - User information
- `POST /oauth/introspect` - Token introspection
- `POST /oauth/revoke` - Token revocation
- `POST /register` - Client registration

## Available Tools

### 1. create-image
Generate images from text prompts with full control over parameters.

**Parameters:**
- `prompt` (str): Text description (max 32000 chars)
- `model` (str): "gpt-image-1" (default)
- `size` (str): "1024x1024", "1536x1024", "1024x1536", "auto"
- `quality` (str): "auto", "high", "medium", "low"
- `background` (str): "auto", "transparent", "opaque"
- `output_format` (str): "png", "jpeg", "webp"
- `output_compression` (int): 0-100 (for webp/jpeg)
- `n` (int): Number of images (1-10)
- `output_mode` (str): "base64" or "file"
- `file_path` (str): Output path (required if output_mode="file")

### 2. analyze-image
Comprehensive image analysis using Vision API.

**Parameters:**
- `image` (str): Image path or base64 string
- `prompt` (str): Analysis prompt (default: detailed description)
- `model` (str): Vision model ("gpt-4o", "gpt-4o-mini")
- `max_tokens` (int): Response length limit
- `detail` (str): "low", "high", "auto"

## File Input Support

### Local Files
```
"/path/to/image.png"
"/path/to/image.jpg"
```

### Google Drive Files
```
"drive://1abc123def456"  # File ID format
"https://drive.google.com/file/d/1abc123def456/view"  # Full URL
```

## Supported Image Formats

**Input/Output:** PNG, JPG, JPEG, WebP, GIF

## Security Features

- **Input Validation**: Comprehensive validation for all parameters
- **Path Traversal Protection**: Prevents directory traversal attacks
- **CORS Configuration**: Restricts access to allowed origins (Claude Web by default)
- **Rate Limiting**: Built-in protection against abuse (100 requests/minute by default)
- **Request Logging**: Monitor access patterns and detect unusual activity
- **Railway Domain Security**: Uses Railway's hard-to-guess domain format for security through obscurity

## Error Handling

- **Specific Error Types**: FileNotFoundError, PermissionError, ValidationError
- **Detailed Logging**: Structured logging with correlation IDs
- **Graceful Degradation**: Continues operation with reduced functionality
- **User-Friendly Messages**: Clear error descriptions for common issues

## Monitoring & Health

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "version": "1.0.0",
  "server": "Image Tool MCP Server",
  "features": ["image_generation", "image_analysis", ...],
  "file_support": {
    "local_files": true,
    "google_drive": true
  },
  "tools_count": 2
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and feature requests, please use the GitHub issue tracker.
