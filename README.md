# Image Tool MCP Server

A comprehensive Model Context Protocol (MCP) server for image processing tasks, deployed on Railway with Claude Desktop compatibility and full OAuth 2.0 support.

## Features

- **Image Generation**: Create images from text prompts using OpenAI's gpt-image-1 or Google Vertex AI Imagen 4.0 Ultra
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
- **Google Drive Integration**: Search, upload, and process images directly from Google Drive
- **OAuth 2.0 Support**: Full OAuth implementation for Claude integration

## OAuth 2.0 Implementation

This server includes a complete OAuth 2.0 implementation that is fully compliant with Anthropic's requirements for remote MCP servers.

### OAuth Features
- ✅ Authorization Code Grant flow
- ✅ Refresh Token support
- ✅ Token revocation
- ✅ OAuth discovery endpoints
- ✅ User info endpoint
- ✅ Claude-compatible error responses
- ✅ JWT token handling
- ✅ Secure token storage

### OAuth Endpoints
- `/.well-known/oauth-authorization-server` - OAuth server discovery
- `/.well-known/oauth-protected-resource/mcp` - Protected resource info
- `/oauth/authorize` - Authorization endpoint
- `/oauth/token` - Token exchange endpoint
- `/oauth/userinfo` - User information endpoint
- `/oauth/revoke` - Token revocation endpoint
- `/oauth/test` - OAuth testing endpoint
- `/oauth/status` - OAuth server status

### OAuth Configuration
- **Supported Grant Types**: Authorization Code, Refresh Token
- **Supported Scopes**: `read`, `write`, `image_processing`
- **Token Expiry**: Access tokens (1 hour), Refresh tokens (30 days)
- **Algorithm**: HS256 (HMAC with SHA-256)

For detailed OAuth documentation, see [OAUTH_IMPLEMENTATION.md](OAUTH_IMPLEMENTATION.md).

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for image generation and analysis (used by most tools; create_image can also use Vertex)

### Optional
#### Vertex AI (Imagen) for create_image (Optional)
- `CREATE_IMAGE_MODEL`: Default model for the `create_image` tool only. Examples:
  - `openai:gpt-image-1` (default)
  - `vertex:imagen-4.0-ultra-generate-preview-06-06` (see: https://cloud.google.com/vertex-ai/generative-ai/docs/models/imagen/4-0-ultra-generate-preview-06-06)
- `GOOGLE_CLOUD_PROJECT`: GCP project for Vertex AI
- `VERTEX_LOCATION`: Vertex region (e.g., `us-central1`)
- `GOOGLE_CLOUD_REGION`: Same as above (for consistency)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to ADC JSON file (in Railway, provide JSON via env and write it to a temp file at startup)

Note: Vertex configuration is used only by `create_image` when `CREATE_IMAGE_MODEL` selects a Vertex model. Other tools continue using OpenAI.

#### Vertex AI (Gemini) for analyze_image (Optional)
- `ANALYZE_IMAGE_MODEL`: Default model for the `analyze_image` tool only. Examples:
  - `gpt-4o` (default)
  - `vertex:gemini-2.5-pro`
  - Provide `GOOGLE_CLOUD_PROJECT`, `VERTEX_LOCATION`, and ADC as above.
- `PORT`: Server port (default: 8080)
- `ALLOWED_ORIGINS`: CORS allowed origins (default: claude.ai domains)
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting (default: 100)
- `JWT_SECRET`: Secret key for JWT signing (auto-generated if not provided)
- `JWT_EXPIRY_HOURS`: Access token expiry in hours (default: 1)
- `REFRESH_TOKEN_EXPIRY_DAYS`: Refresh token expiry in days (default: 30)

### Google Drive Integration (Optional)
- `GOOGLE_OAUTH_TOKEN`: OAuth token JSON for Google Drive access
- `GOOGLE_SERVICE_ACCOUNT_JSON`: Service account credentials JSON for Google Drive access

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

### Testing OAuth
Run the OAuth test script to verify the implementation:
```bash
python test_oauth.py
```

## Railway Deployment

### Environment Variables
```json
{
  "OPENAI_API_KEY": "your-openai-api-key",
  "JWT_SECRET": "your-jwt-secret-key"
}
```

### Deploy
```bash
railway up
```

## Claude Integration

### Adding to Claude
1. In Claude Desktop, go to Settings > Connectors
2. Add your server URL (e.g., `https://your-server.railway.app`)
3. Claude will automatically discover OAuth endpoints
4. Complete the OAuth authorization flow
5. Start using image processing tools

### OAuth Flow
1. Claude requests authorization via `/oauth/authorize`
2. Server generates authorization code
3. Claude exchanges code for access token
4. Claude uses Bearer token for API requests
5. Tokens are automatically refreshed when needed

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

### Health and Status
- `GET /health` - Health check with OAuth status
- `GET /` - Root endpoint with server info
- `GET /oauth/status` - OAuth server status

### OAuth Endpoints
- `GET /.well-known/oauth-authorization-server` - OAuth discovery
- `GET /.well-known/oauth-protected-resource/mcp` - Protected resource info
- `GET /oauth/authorize` - Authorization endpoint
- `POST /oauth/token` - Token exchange
- `GET /oauth/userinfo` - User information
- `POST /oauth/revoke` - Token revocation
- `GET /oauth/test` - OAuth testing
- `GET /.well-known/jwks.json` - JWKS endpoint

### MCP Protocol
- `POST /mcp/` - MCP protocol endpoint

## Available Tools

### Image Processing
1. **create_image** - Generate images from text prompts
2. **analyze_image** - Analyze images with detailed descriptions
3. **edit_image** - Edit images with text prompts
4. **generate_variations** - Create variations of existing images
5. **extract_text** - Extract text from images using OCR
6. **batch_process** - Process multiple images
7. **image_metadata** - Extract image metadata

### Google Drive Integration
8. **search_images** - Search for image files in Google Drive
9. **upload_image** - Upload images to Google Drive with metadata
10. **get_image_from_drive** - Get direct download URLs for Drive images

## Error Handling

The server includes comprehensive error handling for:
- Invalid file paths
- Missing API keys
- Rate limiting
- Network timeouts
- Invalid image formats
- OAuth authentication errors
- Token validation failures

## Security Features

- Path traversal protection
- Rate limiting
- CORS configuration
- Input validation
- Secure file handling
- OAuth 2.0 authentication
- JWT token security
- Token revocation support

## Testing

### OAuth Testing
```bash
python test_oauth.py
```

### Manual Testing
```bash
# Test OAuth discovery
curl https://your-server.railway.app/.well-known/oauth-authorization-server

# Test health endpoint
curl https://your-server.railway.app/health

# Test OAuth status
curl https://your-server.railway.app/oauth/status
```

## Production Considerations

### OAuth Security
- Use HTTPS in production
- Set a strong JWT_SECRET
- Consider using Redis for token storage
- Implement rate limiting on OAuth endpoints
- Monitor OAuth usage patterns

### Token Storage
For production deployments, consider replacing in-memory token storage with:
- Redis for distributed deployments
- Database for persistent storage
- Secure key management for JWT secrets

## License

MIT License
