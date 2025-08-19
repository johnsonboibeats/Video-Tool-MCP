# Video Tool MCP Server

A comprehensive Model Context Protocol (MCP) server for video generation and processing tasks, deployed on Railway with Claude Desktop compatibility and full OAuth 2.0 support.

## Features

- **Video Generation**: Create videos from text prompts using Google Veo3 API
- **Video Analysis**: Analyze videos with detailed descriptions using GPT-4o
- **Video Editing**: Edit videos with text prompts and optional masks
- **Video Variations**: Generate variations of existing videos
- **Text Extraction**: Extract text from video frames using OCR
- **Video Comparison**: Compare two videos for similarities and differences
- **Smart Editing**: Advanced editing with analysis and modification prompts
- **Video Transformations**: Resize, rotate, flip, and apply filters to video frames
- **Batch Processing**: Process multiple videos with various operations
- **Video Metadata**: Extract technical information from videos
- **Prompt Generation**: Generate optimized prompts from video content
- **Local File Support**: Process files from local file system
- **Base64 Support**: Handle base64 encoded video data
- **Google Drive Integration**: Search, upload, and process videos directly from Google Drive
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
- `GEMINI_API_KEY`: Your Gemini API key for Veo3 video generation
- `OPENAI_API_KEY`: Your OpenAI API key for video analysis (used by most tools)
- `GOOGLE_OAUTH_TOKEN`: OAuth token JSON for Google Drive access (videos are auto-uploaded to Drive)

### Optional
#### Gemini API (Veo3) for create_video (Required for video generation)
- Models available: `veo-3.0-generate-preview`, `veo-3.0-fast-generate-preview`
- Uses Gemini API for streamlined video generation

Note: Gemini API configuration is used by `create_video` for video generation. Other tools continue using OpenAI.

#### Server Configuration (Optional)
- `PORT`: Server port (default: 8080)
- `ALLOWED_ORIGINS`: CORS allowed origins (default: claude.ai domains)
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting (default: 100)
- `JWT_SECRET`: Secret key for JWT signing (auto-generated if not provided)
- `JWT_EXPIRY_HOURS`: Access token expiry in hours (default: 1)
- `REFRESH_TOKEN_EXPIRY_DAYS`: Refresh token expiry in days (default: 30)

## Local Development

### Prerequisites
- Python 3.8+
- Google API key with Veo3 access
- OpenAI API key for video analysis

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   export GOOGLE_OAUTH_TOKEN="your-oauth-token-json"
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
  "GEMINI_API_KEY": "your-gemini-api-key",
  "OPENAI_API_KEY": "your-openai-api-key",
  "GOOGLE_OAUTH_TOKEN": "your-oauth-token-json",
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
5. Start using video processing tools

### OAuth Flow
1. Claude requests authorization via `/oauth/authorize`
2. Server generates authorization code
3. Claude exchanges code for access token
4. Claude uses Bearer token for API requests
5. Tokens are automatically refreshed when needed

## Usage Examples

### Video Generation
```json
{
  "prompt": "A serene mountain landscape at sunset with flowing water",
  "duration": 5,
  "resolution": "1080p",
  "output_format": "mp4"
}
```

### Video Analysis
```json
{
  "video": "/path/to/video.mp4",
  "prompt": "Describe this video in detail"
}
```

### Video Editing
```json
{
  "video": "/path/to/video.mp4",
  "prompt": "Add a red car to the scene",
  "output_format": "mp4"
}
```

### File Input Types

The server supports the following input types:

#### Local Files
```json
{
  "video": "/absolute/path/to/video.mp4"
}
```

#### Base64 Data
```json
{
  "video": "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28y..."
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

### Video Processing
1. **create_video** - Generate videos from text prompts using Google Veo3
2. **analyze_video** - Analyze videos with detailed descriptions
3. **edit_video** - Edit videos with text prompts
4. **generate_variations** - Create variations of existing videos
5. **extract_text** - Extract text from video frames using OCR
6. **batch_process** - Process multiple videos
7. **video_metadata** - Extract video metadata

### Google Drive Integration
8. **search_videos** - Search for video files in Google Drive
9. **upload_video** - Upload videos to Google Drive with metadata
10. **get_video_from_drive** - Get direct download URLs for Drive videos

## Error Handling

The server includes comprehensive error handling for:
- Invalid file paths
- Missing API keys
- Rate limiting
- Network timeouts
- Invalid video formats
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
