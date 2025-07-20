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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Core imports
import aiofiles
from openai import AsyncOpenAI, AsyncAzureOpenAI
from PIL import Image as PILImage, ImageOps
import httpx
from fastmcp import FastMCP, Context
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

# Starlette for OAuth endpoints
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

@dataclass
class AppContext:
    """Application context with shared resources"""
    openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI]
    temp_dir: Path
    http_client: httpx.AsyncClient
    drive_service: Optional[Any] = None

# Global context variables
app_context: AppContext = None

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
        global app_context
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
    global app_context
    
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

class SimpleOAuth2Server:
    """Simple OAuth 2.0 server for Claude Web"""
    
    def __init__(self):
        self.authorization_codes = {}
        self.access_tokens = {}
        self.token_expiry = 3600
        self.server_url = "https://image-tool-mcp-production.up.railway.app"
        logger.info("OAuth2 server initialized")
        
    def generate_authorization_code(self, client_id: str, state: str = None, 
                                  code_challenge: str = None, code_challenge_method: str = None) -> str:
        """Generate authorization code"""
        auth_code = secrets.token_urlsafe(32)
        self.authorization_codes[auth_code] = {
            "client_id": client_id,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "created_at": time.time(),
            "expires_at": time.time() + 600
        }
        logger.info(f"Generated auth code for client: {client_id}")
        return auth_code
    
    def exchange_code_for_token(self, code: str, client_id: str, code_verifier: str = None) -> Optional[str]:
        """Exchange code for token"""
        if code not in self.authorization_codes:
            logger.error(f"Invalid code: {code}")
            return None
            
        code_data = self.authorization_codes[code]
        
        # Check expiration
        if time.time() > code_data["expires_at"]:
            del self.authorization_codes[code]
            logger.error("Code expired")
            return None
        
        # Verify PKCE if needed
        if code_data.get("code_challenge") and code_verifier:
            if code_data.get("code_challenge_method") == "S256":
                verifier_hash = hashlib.sha256(code_verifier.encode()).digest()
                verifier_challenge = base64.urlsafe_b64encode(verifier_hash).decode().rstrip('=')
                if verifier_challenge != code_data["code_challenge"]:
                    del self.authorization_codes[code]
                    logger.error("PKCE failed")
                    return None
        
        # Generate token
        access_token = secrets.token_urlsafe(32)
        self.access_tokens[access_token] = {
            "client_id": client_id,
            "created_at": time.time(),
            "expires_at": time.time() + self.token_expiry
        }
        
        del self.authorization_codes[code]
        logger.info(f"Token issued for client: {client_id}")
        return access_token

# Initialize OAuth server
oauth_server = SimpleOAuth2Server()

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
        http_client = httpx.AsyncClient()
        
        # Create context
        app_context = AppContext(
            openai_client=client,
            temp_dir=temp_dir,
            http_client=http_client,
            drive_service=drive_service
        )
        
        # Set global context for FastMCP tools
        global app_context as global_app_context
        global_app_context = app_context
        
        return app_context
        
    except Exception as e:
        logger.error(f"Failed to initialize app context: {e}")
        raise

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle with proper resource cleanup"""
    logger.info("Image Tool MCP Server starting...")
    
    try:
        # Initialize app context
        app_context = initialize_app_context()
        yield app_context
    finally:
        # Cleanup temp files older than 1 hour
        import time
        current_time = time.time()
        for file_path in app_context.temp_dir.glob("*"):
            if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:
                try:
                    file_path.unlink()
                except OSError:
                    pass
        logger.info("Image Tool MCP Server shutdown complete")

# Create FastMCP server
mcp = FastMCP(
    name="Image Tool MCP",
    lifespan=app_lifespan,
    dependencies=["fastmcp>=0.4.0", "openai>=1.97.0", "pillow>=11.3.0", "httpx>=0.28.1"]
)

# [Rest of the tools will be copied from original server.py]
# This is a framework - the actual tools need to be added here

# OAuth endpoints
async def oauth_discovery(request: Request) -> JSONResponse:
    """OAuth discovery endpoint"""
    base_url = oauth_server.server_url
    metadata = {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["claudeai"]
    }
    logger.info("OAuth discovery requested")
    return JSONResponse(metadata)

async def oauth_protected_resource(request: Request) -> JSONResponse:
    """OAuth Protected Resource Metadata endpoint"""
    base_url = oauth_server.server_url
    metadata = {
        "resource": base_url,
        "authorization_servers": [base_url],
        "scopes_supported": ["claudeai"],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{base_url}/docs"
    }
    logger.info("OAuth protected resource metadata requested")
    return JSONResponse(metadata)

async def authorize_endpoint(request: Request) -> JSONResponse:
    """Authorization endpoint"""
    params = dict(request.query_params)
    logger.info(f"Authorization request: {params}")
    
    response_type = params.get("response_type")
    client_id = params.get("client_id")
    redirect_uri = params.get("redirect_uri")
    state = params.get("state")
    code_challenge = params.get("code_challenge")
    code_challenge_method = params.get("code_challenge_method", "plain")
    resource = params.get("resource")  # MCP requires this parameter
    
    if not all([response_type, client_id, redirect_uri]):
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    
    if response_type != "code":
        return JSONResponse({"error": "unsupported_response_type"}, status_code=400)
    
    # Validate resource parameter if provided
    if resource and resource != oauth_server.server_url:
        return JSONResponse({"error": "invalid_resource"}, status_code=400)
    
    # Generate authorization code
    auth_code = oauth_server.generate_authorization_code(
        client_id=client_id,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method
    )
    
    # Redirect with code
    redirect_url = f"{redirect_uri}?code={auth_code}"
    if state:
        redirect_url += f"&state={state}"
    
    logger.info(f"Redirecting to: {redirect_url}")
    return RedirectResponse(url=redirect_url)

async def token_endpoint(request: Request) -> JSONResponse:
    """Token endpoint"""
    try:
        if request.method == "POST":
            form_data = await request.form()
            params = dict(form_data)
        else:
            params = dict(request.query_params)
        
        logger.info(f"Token request: {params}")
        
        grant_type = params.get("grant_type")
        code = params.get("code")
        client_id = params.get("client_id")
        code_verifier = params.get("code_verifier")
        
        if grant_type != "authorization_code":
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
        
        if not all([code, client_id]):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        
        # Exchange code for token
        access_token = oauth_server.exchange_code_for_token(
            code=code,
            client_id=client_id,
            code_verifier=code_verifier
        )
        
        if not access_token:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        
        response_data = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": oauth_server.token_expiry,
            "scope": "claudeai"
        }
        
        logger.info(f"Token issued for client: {client_id}")
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Token error: {e}")
        return JSONResponse({"error": "server_error"}, status_code=500)

def create_app():
    """Create the ASGI application following FastMCP + Starlette documentation"""
    # Get FastMCP app first
    mcp_app = mcp.http_app()
    
    # OAuth routes
    routes = [
        Route("/.well-known/oauth-authorization-server", oauth_discovery, methods=["GET"]),
        Route("/.well-known/oauth-protected-resource", oauth_protected_resource, methods=["GET"]),
        Route("/authorize", authorize_endpoint, methods=["GET", "POST"]),
        Route("/token", token_endpoint, methods=["POST", "GET"]),
        Mount("/mcp", mcp_app),
    ]
    
    # CORS middleware for the main Starlette app
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["https://claude.ai", "https://*.claude.ai"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    ]
    
    # Create app using FastMCP's built-in lifespan (as per documentation)
    app = Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=mcp_app.lifespan  # Use FastMCP's lifespan as documented
    )
    
    logger.info("ASGI app created successfully following FastMCP + Starlette documentation")
    return app

# Create and export app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info("Starting Image Tool MCP Server for Railway deployment...")
    logger.info("OAuth endpoints available:")
    logger.info("- /.well-known/oauth-authorization-server")
    logger.info("- /.well-known/oauth-protected-resource") 
    logger.info("- /authorize")
    logger.info("- /token")
    logger.info("- /mcp (FastMCP server)")
    logger.info(f"Server will run on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)