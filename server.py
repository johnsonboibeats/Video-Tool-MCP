#!/usr/bin/env python3
"""
Image Tool MCP Server - Simplified Railway Deploy Version
"""


# Suppress warnings for cleaner deployment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*not JSON serializable.*", category=UserWarning)

import argparse
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
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict


# Configure logging early - in stdio mode, log to stderr to avoid polluting stdout
if "--transport" in sys.argv and "stdio" in sys.argv:
    # In stdio mode, log to stderr so it doesn't interfere with JSON-RPC on stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


# Core imports
try:
    import aiofiles
    from openai import AsyncOpenAI, AsyncAzureOpenAI
    from fastmcp import FastMCP, Context
    from dotenv import load_dotenv
except Exception as e:
    logger.error(f"Core import failed: {e}")
    raise


# Image processing imports with graceful fallbacks
try:
    from PIL import Image as PILImage, ImageOps, ImageFilter, ImageEnhance, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available - image processing will be disabled")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("Httpx not available - some image features may be disabled")

# Google Drive imports with graceful fallbacks
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    logger.warning("Google Drive libraries not available - Drive integration will be disabled")

# Load environment variables
load_dotenv()

class AppContext(BaseModel):
    """Application context with shared resources"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    openai_client: Optional[Union[AsyncOpenAI, AsyncAzureOpenAI]] = None
    drive_service: Any = None
    temp_dir: Path
    http_client: Optional[httpx.AsyncClient] = None

# Global app context for FastMCP tools
_global_app_context: Optional[AppContext] = None

# Global transport mode detection
_transport_mode: str = "http"  # Default to http (remote)

def get_default_output_mode() -> str:
    """Determine default output mode based on transport"""
    global _transport_mode
    # Remote: URLs for better UX, Local: Files for direct filesystem access
    return "url" if _transport_mode != "stdio" else "file"

def get_app_context() -> AppContext:
    """Get application context from global reference"""
    if _global_app_context is not None:
        return _global_app_context
    raise RuntimeError("Application context not initialized")

def check_openai_client(client) -> None:
    """Check if OpenAI client is available"""
    if client is None:
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")

# =============================================================================
# GOOGLE DRIVE INTEGRATION
# =============================================================================

# Google Drive constants
GOOGLE_DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']

def extract_file_id_from_url(url: str) -> Optional[str]:
    """Extract Google Drive file ID from various URL formats"""
    if not url:
        return None
        
    if url.startswith('drive://'):
        return url[8:]
    
    # Handle direct file IDs
    if len(url) == 33 and url.replace('-', '').replace('_', '').isalnum():
        return url
    
    # Parse Google Drive URLs
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        r'docs\.google\.com/.*?/d/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

async def authenticate_google_drive() -> Any:
    """Authenticate with Google Drive using OAuth 2.0"""
    if not GOOGLE_DRIVE_AVAILABLE:
        return None
    
    try:
        # Check for OAuth token from environment (Railway-friendly)
        oauth_token = os.getenv('GOOGLE_OAUTH_TOKEN')
        if oauth_token:
            try:
                token_data = json.loads(oauth_token)
                creds = Credentials.from_authorized_user_info(token_data, GOOGLE_DRIVE_SCOPES)
                logger.info("Using OAuth credentials from GOOGLE_OAUTH_TOKEN")
                
                # Build and return the service
                service = build('drive', 'v3', credentials=creds)
                return service
            except Exception as e:
                logger.error(f"Failed to load OAuth token: {e}")
        
        logger.warning("No Google Drive OAuth token found - Drive features will be disabled. Please set GOOGLE_OAUTH_TOKEN environment variable.")
        return None
        
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Drive: {e}")
        return None

async def handle_file_input(file_input: str, app_context: AppContext) -> str:
    """Handle file input from base64 data or HTTP URLs."""
    # Handle base64 data URLs
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
    
    # Handle HTTP URLs by downloading them
    if file_input.startswith('http://') or file_input.startswith('https://'):
        try:
            if not app_context.http_client:
                raise ValueError("HTTP client not available for downloading URLs")
            
            response = await app_context.http_client.get(file_input)
            response.raise_for_status()
            
            # Determine file extension from Content-Type or URL
            content_type = response.headers.get('content-type', '')
            if content_type:
                extension = mimetypes.guess_extension(content_type.split(';')[0]) or '.bin'
            else:
                # Extract extension from URL
                extension = Path(file_input).suffix or '.bin'
            
            temp_path = app_context.temp_dir / f"temp_download_{int(time.time())}{extension}"
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(response.content)
            
            return str(temp_path)
        except Exception as e:
            raise ValueError(f"Failed to download file from URL: {e}")
    
    # This function should not be called with other input types.
    # The get_file_path function is responsible for routing.
    raise ValueError("Internal error: handle_file_input called with invalid input type.")

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
        elif os.getenv("OPENAI_API_KEY"):
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Initialized OpenAI client")
        else:
            logger.warning("No OpenAI API key found - image tools will not function")
            client = None
        
        # Setup temp directory
        temp_dir = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir())) / "image_tool_mcp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory: {temp_dir}")
        
        # Create HTTP client
        http_client = httpx.AsyncClient() if HTTPX_AVAILABLE else None
        
        # Initialize Google Drive service (synchronous version)
        drive_service = None
        if GOOGLE_DRIVE_AVAILABLE:
            try:
                # Check for OAuth token from environment (Railway-friendly)
                oauth_token = os.getenv('GOOGLE_OAUTH_TOKEN')
                if oauth_token:
                    try:
                        token_data = json.loads(oauth_token)
                        creds = Credentials.from_authorized_user_info(token_data, GOOGLE_DRIVE_SCOPES)
                        logger.info("Using OAuth credentials from GOOGLE_OAUTH_TOKEN")
                        drive_service = build('drive', 'v3', credentials=creds)
                    except Exception as e:
                        logger.error(f"Failed to load OAuth token: {e}")
                
                if drive_service:
                    logger.info("Google Drive service initialized successfully")
                else:
                    logger.warning("No Google Drive OAuth token found - Drive features will be disabled. Please set GOOGLE_OAUTH_TOKEN environment variable.")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Google Drive service: {e}")
                drive_service = None
        
        # Create context
        context = AppContext(
            openai_client=client,
            drive_service=drive_service,
            temp_dir=temp_dir,
            http_client=http_client
        )
        
        # Set global context for FastMCP tools
        global _global_app_context
        _global_app_context = context
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to initialize app context: {e}")
        raise

# Initialize application context at startup
try:
    initialize_app_context()
    logger.info("Application context initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize application context: {e}")
    logger.warning("Server will start but image tools may not function properly")
    # Create a minimal context to allow server to start
    _global_app_context = AppContext(
        openai_client=None,
        temp_dir=Path(tempfile.gettempdir()) / "image_tool_mcp",
        http_client=None
    )

# Railway security configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://claude.ai,https://web.claude.ai,https://*.claude.ai").split(",")
# Clean up any extra whitespace or semicolons
ALLOWED_ORIGINS = [origin.strip().rstrip(';') for origin in ALLOWED_ORIGINS]
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))

logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")
logger.info(f"Rate limit: {MAX_REQUESTS_PER_MINUTE} requests per minute")

# Create FastMCP server
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict

# Initialize FastMCP server
mcp = FastMCP("Image Tool MCP")

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    def __init__(self, app):
        super().__init__(app)
        self.max_requests = 100
        self.window = 60
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if now - req_time < self.window
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                {"error": "Rate limit exceeded. Please try again later."}, 
                status_code=429
            )
        
        self.requests[client_ip].append(now)
        return await call_next(request)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log requests for monitoring"""
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log all requests except health checks
        if request.url.path != "/health":
            logger.info(f"Request from {client_ip} - {user_agent} - {request.method} {request.url.path}")
        else:
            # Log health checks at debug level
            logger.debug(f"Health check from {client_ip}")
        
        return await call_next(request)

# Simplified initialization - no middleware for better Claude Web compatibility

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    """Health check endpoint for Railway deployment"""
    logger.info("Health check endpoint called")
    try:
        response_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "server": "Image Tool MCP Server",
            "oauth_enabled": True,
            "oauth_issuer": OAUTH_CONFIG["issuer"],
            "oauth_endpoints": {
                "authorization": "/oauth/authorize",
                "token": "/oauth/token",
                "userinfo": "/oauth/userinfo",
                "revoke": "/oauth/revoke"
            }
        }
        
        # Safely check OpenAI configuration
        try:
            response_data["openai_configured"] = _global_app_context.openai_client is not None if _global_app_context else False
        except Exception:
            response_data["openai_configured"] = False
            
        # Safely check Google Drive configuration
        try:
            response_data["google_drive_configured"] = _global_app_context.drive_service is not None if _global_app_context else False
        except Exception:
            response_data["google_drive_configured"] = False
            
        logger.info(f"Health check response: {response_data}")
        return JSONResponse(response_data)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request):
    """Root endpoint for basic connectivity test"""
    logger.info("Root endpoint called")
    return JSONResponse({
        "message": "Image Tool MCP Server is running",
        "health_check": "/health",
        "mcp_endpoint": "/mcp/"
    })

@mcp.custom_route("/download/{filename}", methods=["GET"])
async def download_image(request: Request):
    """Serve generated images for download"""
    filename = request.path_params["filename"]
    logger.info(f"Download request for: {filename}")
    
    # Get application context for temp directory
    app_context = get_app_context()
    download_path = app_context.temp_dir / "downloads" / filename
    
    if download_path.exists() and download_path.is_file():
        logger.info(f"Serving file: {download_path}")
        return FileResponse(download_path)
    else:
        logger.warning(f"File not found: {download_path}")
        return JSONResponse({"error": "File not found"}, status_code=404)

@mcp.custom_route("/mcp", methods=["GET", "HEAD", "POST"])
async def mcp_redirect(request: Request):
    """Handle MCP endpoint redirects"""
    from starlette.responses import RedirectResponse
    logger.info(f"MCP redirect called: {request.method} {request.url.path}")
    
    if request.method == "HEAD":
        # Return 200 for HEAD requests to /mcp
        return JSONResponse({"status": "ok"}, status_code=200)
    elif request.method == "POST":
        # Redirect POST requests to /mcp/
        return RedirectResponse(url="/mcp/", status_code=307)
    else:
        # Redirect GET requests to /mcp/
        return RedirectResponse(url="/mcp/", status_code=307)

# =============================================================================
# OAUTH IMPLEMENTATION
# =============================================================================

import jwt
from datetime import datetime, timedelta


# OAuth configuration
OAUTH_CONFIG = {
    "issuer": "https://claude.ai",
    "authorization_endpoint": "https://claude.ai/oauth/authorize",
    "token_endpoint": "https://claude.ai/oauth/token",
    "jwks_uri": "https://claude.ai/.well-known/jwks.json",
    "response_types_supported": ["code"],
    "subject_types_supported": ["public"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "scopes_supported": ["read", "write", "image_processing"],
    "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
    "grant_types_supported": ["authorization_code", "refresh_token"]
}

# JWT configuration for token handling
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 1
REFRESH_TOKEN_EXPIRY_DAYS = 30

# In-memory token storage (use Redis/database in production)
token_store: Dict[str, Dict[str, Any]] = {}

def generate_access_token(user_id: str, scopes: list = None, audience: str = None) -> str:
    """Generate JWT access token with audience validation"""
    payload = {
        "sub": user_id,
        "iss": OAUTH_CONFIG["issuer"],
        "aud": audience or "claude",  # RFC 8707: Include specific audience
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
        "scope": " ".join(scopes or ["read", "write"])
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def generate_refresh_token(user_id: str) -> str:
    """Generate refresh token"""
    payload = {
        "sub": user_id,
        "iss": OAUTH_CONFIG["issuer"],
        "aud": "claude",
        "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS),
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str, expected_audience: str = None) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token with audience validation"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Validate audience (RFC 8707 requirement)
        if expected_audience:
            token_audience = payload.get("aud")
            if not token_audience or expected_audience not in (token_audience if isinstance(token_audience, list) else [token_audience]):
                return None
        
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_token_from_header(request: Request) -> Optional[str]:
    """Extract token from Authorization header"""
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    return None

async def require_auth(request: Request) -> Optional[Dict[str, Any]]:
    """Middleware to require authentication"""
    token = get_token_from_header(request)
    if not token:
        return None
    
    # Validate token with audience (RFC 8707)
    server_uri = f"{request.base_url.scheme}://{request.base_url.netloc}"
    payload = verify_token(token, expected_audience=server_uri)
    if not payload:
        return None
    
    return payload

# =============================================================================
# OAUTH ENDPOINTS
# =============================================================================

@mcp.custom_route("/oauth/authorize", methods=["GET"])
async def oauth_authorize(request: Request):
    """OAuth authorization endpoint"""
    logger.info("OAuth authorization endpoint called")
    
    # Extract OAuth parameters
    client_id = request.query_params.get("client_id")
    redirect_uri = request.query_params.get("redirect_uri")
    response_type = request.query_params.get("response_type")
    scope = request.query_params.get("scope", "read write")
    state = request.query_params.get("state")
    resource = request.query_params.get("resource")  # RFC 8707 requirement
    
    # Validate resource parameter (RFC 8707)
    if not resource:
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "resource parameter is required per RFC 8707"
        }, status_code=400)
    
    # Validate resource matches this server
    server_uri = f"{request.base_url.scheme}://{request.base_url.netloc}"
    if not resource.startswith(server_uri):
        return JSONResponse({
            "error": "invalid_request",
            "error_description": "resource parameter must match server URI"
        }, status_code=400)
    
    # Validate parameters
    if not client_id or client_id != "Claude":
        return JSONResponse({
            "error": "invalid_client",
            "error_description": "Invalid client ID"
        }, status_code=400)
    
    if response_type != "code":
        return JSONResponse({
            "error": "unsupported_response_type",
            "error_description": "Only 'code' response type is supported"
        }, status_code=400)
    
    # Generate authorization code
    auth_code = secrets.token_urlsafe(32)
    user_id = "claude_user"  # In production, get from session
    
    # Store authorization code (use Redis/database in production)
    token_store[auth_code] = {
        "user_id": user_id,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "resource": resource,  # RFC 8707: Store resource for audience validation
        "expires": datetime.utcnow() + timedelta(minutes=10)
    }
    
    # Build redirect URL
    redirect_url = f"{redirect_uri}?code={auth_code}"
    if state:
        redirect_url += f"&state={state}"
    
    logger.info(f"Authorization code generated for client {client_id}")
    return JSONResponse({
        "authorization_code": auth_code,
        "redirect_uri": redirect_url,
        "expires_in": 600
    })

@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token(request: Request):
    """OAuth token endpoint"""
    logger.info("OAuth token endpoint called")
    
    try:
        form_data = await request.form()
        grant_type = form_data.get("grant_type")
        
        if grant_type == "authorization_code":
            return await handle_authorization_code_grant(form_data)
        elif grant_type == "refresh_token":
            return await handle_refresh_token_grant(form_data)
        else:
            return JSONResponse({
                "error": "unsupported_grant_type",
                "error_description": f"Grant type '{grant_type}' not supported"
            }, status_code=400)
            
    except Exception as e:
        logger.error(f"Token endpoint error: {e}")
        return JSONResponse({
            "error": "server_error",
            "error_description": "Internal server error"
        }, status_code=500)

async def handle_authorization_code_grant(form_data) -> JSONResponse:
    """Handle authorization code grant"""
    code = form_data.get("code")
    client_id = form_data.get("client_id")
    client_secret = form_data.get("client_secret")
    redirect_uri = form_data.get("redirect_uri")
    
    # Validate authorization code
    if code not in token_store:
        return JSONResponse({
            "error": "invalid_grant",
            "error_description": "Invalid authorization code"
        }, status_code=400)
    
    auth_data = token_store[code]
    
    # Check if code is expired
    if datetime.utcnow() > auth_data["expires"]:
        del token_store[code]
        return JSONResponse({
            "error": "invalid_grant",
            "error_description": "Authorization code expired"
        }, status_code=400)
    
    # Validate client
    if client_id != auth_data["client_id"]:
        return JSONResponse({
            "error": "invalid_client",
            "error_description": "Client ID mismatch"
        }, status_code=400)
    
    # Generate tokens
    user_id = auth_data["user_id"]
    scopes = auth_data["scope"].split()
    
    # Include resource as audience (RFC 8707)
    resource = auth_data.get("resource", "claude")
    access_token = generate_access_token(user_id, scopes, audience=resource)
    refresh_token = generate_refresh_token(user_id)
    
    # Store refresh token
    token_store[refresh_token] = {
        "user_id": user_id,
        "client_id": client_id,
        "scopes": scopes,
        "resource": resource,  # RFC 8707: Store resource for audience validation
        "expires": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    }
    
    # Clean up authorization code
    del token_store[code]
    
    logger.info(f"Access token generated for user {user_id}")
    
    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": JWT_EXPIRY_HOURS * 3600,
        "refresh_token": refresh_token,
        "scope": " ".join(scopes)
    })

async def handle_refresh_token_grant(form_data) -> JSONResponse:
    """Handle refresh token grant"""
    refresh_token = form_data.get("refresh_token")
    client_id = form_data.get("client_id")
    client_secret = form_data.get("client_secret")
    
    # Validate refresh token
    if refresh_token not in token_store:
        return JSONResponse({
            "error": "invalid_grant",
            "error_description": "Invalid refresh token"
        }, status_code=400)
    
    token_data = token_store[refresh_token]
    
    # Check if refresh token is expired
    if datetime.utcnow() > token_data["expires"]:
        del token_store[refresh_token]
        return JSONResponse({
            "error": "invalid_grant",
            "error_description": "Refresh token expired"
        }, status_code=400)
    
    # Validate client
    if client_id != token_data["client_id"]:
        return JSONResponse({
            "error": "invalid_client",
            "error_description": "Client ID mismatch"
        }, status_code=400)
    
    # Generate new tokens
    user_id = token_data["user_id"]
    scopes = token_data["scopes"]
    
    # Include resource as audience (RFC 8707)
    resource = token_data.get("resource", "claude")
    access_token = generate_access_token(user_id, scopes, audience=resource)
    new_refresh_token = generate_refresh_token(user_id)
    
    # Store new refresh token
    token_store[new_refresh_token] = {
        "user_id": user_id,
        "client_id": client_id,
        "scopes": scopes,
        "resource": resource,  # RFC 8707: Store resource for audience validation
        "expires": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)
    }
    
    # Clean up old refresh token
    del token_store[refresh_token]
    
    logger.info(f"Token refreshed for user {user_id}")
    
    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": JWT_EXPIRY_HOURS * 3600,
        "refresh_token": new_refresh_token,
        "scope": " ".join(scopes)
    })

@mcp.custom_route("/oauth/revoke", methods=["POST"])
async def oauth_revoke(request: Request):
    """OAuth token revocation endpoint"""
    logger.info("OAuth revocation endpoint called")
    
    try:
        form_data = await request.form()
        token = form_data.get("token")
        token_type_hint = form_data.get("token_type_hint", "access_token")
        
        if not token:
            return JSONResponse({
                "error": "invalid_request",
                "error_description": "Token parameter required"
            }, status_code=400)
        
        # Revoke token (remove from store)
        if token in token_store:
            del token_store[token]
            logger.info(f"Token revoked: {token[:10]}...")
        
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        logger.error(f"Revocation endpoint error: {e}")
        return JSONResponse({
            "error": "server_error",
            "error_description": "Internal server error"
        }, status_code=500)

@mcp.custom_route("/oauth/userinfo", methods=["GET"])
async def oauth_userinfo(request: Request):
    """OAuth userinfo endpoint"""
    logger.info("OAuth userinfo endpoint called")
    
    # Verify access token
    payload = await require_auth(request)
    if not payload:
        return JSONResponse({
            "error": "invalid_token",
            "error_description": "Invalid or expired access token"
        }, status_code=401)
    
    user_id = payload.get("sub")
    
    return JSONResponse({
        "sub": user_id,
        "iss": OAUTH_CONFIG["issuer"],
        "name": "Claude User",
        "email": f"{user_id}@claude.ai",
        "scope": payload.get("scope", "read write")
    })

# =============================================================================
# ENHANCED OAUTH DISCOVERY ENDPOINTS
# =============================================================================

@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_discovery(request: Request):
    """Enhanced OAuth discovery endpoint for Claude compatibility"""
    logger.info("OAuth discovery endpoint called")
    return JSONResponse(OAUTH_CONFIG)

@mcp.custom_route("/.well-known/oauth-authorization-server/mcp", methods=["GET"])
async def oauth_discovery_mcp(request: Request):
    """OAuth discovery endpoint for MCP path"""
    return await oauth_discovery(request)

@mcp.custom_route("/.well-known/oauth-protected-resource/mcp", methods=["GET"])
async def oauth_protected_resource(request: Request):
    """OAuth protected resource endpoint for Claude compatibility"""
    logger.info("OAuth protected resource endpoint called")
    return JSONResponse({
        "resource": "mcp",
        "scopes": ["read", "write", "image_processing"],
        "token_endpoint": f"{request.base_url}oauth/token",
        "userinfo_endpoint": f"{request.base_url}oauth/userinfo"
    })

# =============================================================================
# OAUTH MIDDLEWARE
# =============================================================================

class OAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle OAuth authentication for MCP endpoints"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip OAuth for discovery and token endpoints
        if any(path in request.url.path for path in [
            "/.well-known/",
            "/oauth/",
            "/health",
            "/",
            "/mcp"
        ]):
            return await call_next(request)
        
        # Check for valid OAuth token
        payload = await require_auth(request)
        if not payload:
            # RFC9728 Section 5.1: Include WWW-Authenticate header
            response = JSONResponse({
                "error": "invalid_token",
                "error_description": "Valid OAuth token required"
            }, status_code=401)
            response.headers["WWW-Authenticate"] = f'Bearer realm="{request.base_url}", error="invalid_token", error_description="Valid OAuth token required"'
            return response
        
        # Add user info to request state
        request.state.user = payload
        
        return await call_next(request)

# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# FastMCP handles CORS automatically, no need to add middleware manually
# The CORS configuration is handled internally by FastMCP

# Note: OAuth protection is implemented directly in routes that need it
# rather than using global middleware to avoid conflicts with MCP protocol

def require_oauth_auth(func):
    """Decorator to require OAuth authentication for specific routes"""
    async def wrapper(request: Request, *args, **kwargs):
        # Skip OAuth for MCP protocol endpoints
        if request.url.path.startswith("/mcp/"):
            return await func(request, *args, **kwargs)
        
        # Check for valid OAuth token
        payload = await require_auth(request)
        if not payload:
            return JSONResponse({
                "error": "invalid_token",
                "error_description": "Valid OAuth token required"
            }, status_code=401)
        
        # Add user info to request state
        request.state.user = payload
        
        return await func(request, *args, **kwargs)
    
    return wrapper

# =============================================================================
# ENHANCED ERROR HANDLING FOR CLAUDE COMPATIBILITY
# =============================================================================

def create_claude_compatible_error(error_type: str, message: str, status_code: int = 400):
    """Create error responses compatible with Claude's expectations"""
    return JSONResponse({
        "error": error_type,
        "error_description": message,
        "claude_compatible": True,
        "timestamp": datetime.utcnow().isoformat()
    }, status_code=status_code)

# =============================================================================
# OAUTH TESTING ENDPOINTS
# =============================================================================

@mcp.custom_route("/oauth/test", methods=["GET"])
async def oauth_test(request: Request):
    """Test endpoint for OAuth functionality"""
    logger.info("OAuth test endpoint called")
    
    # Check if user is authenticated
    payload = await require_auth(request)
    if payload:
        return JSONResponse({
            "authenticated": True,
            "user_id": payload.get("sub"),
            "scopes": payload.get("scope", "").split(),
            "expires": payload.get("exp")
        })
    else:
        return JSONResponse({
            "authenticated": False,
            "message": "No valid OAuth token provided"
        })

@mcp.custom_route("/oauth/status", methods=["GET"])
async def oauth_status(request: Request):
    """OAuth server status endpoint"""
    logger.info("OAuth status endpoint called")
    
    return JSONResponse({
        "oauth_enabled": True,
        "issuer": OAUTH_CONFIG["issuer"],
        "supported_grant_types": OAUTH_CONFIG["grant_types_supported"],
        "supported_scopes": OAUTH_CONFIG["scopes_supported"],
        "active_tokens": len(token_store),
        "server_time": datetime.utcnow().isoformat()
    })

@mcp.custom_route("/register", methods=["POST"])
async def register_endpoint(request: Request):
    """Registration endpoint for Claude compatibility"""
    logger.info("Registration endpoint called")
    return JSONResponse({
        "status": "success",
        "message": "Registration not required for this MCP server",
        "oauth_required": True
    })

@mcp.custom_route("/auth", methods=["GET", "POST"])
async def auth_endpoint(request: Request):
    """Simple auth endpoint for Claude compatibility"""
    logger.info("Auth endpoint called")
    
    # Check if user is authenticated
    payload = await require_auth(request)
    if payload:
        return JSONResponse({
            "status": "authenticated",
            "user_id": payload.get("sub"),
            "scopes": payload.get("scope", "").split(),
            "message": "User is authenticated"
        })
    else:
        return JSONResponse({
            "status": "unauthenticated",
            "message": "No valid OAuth token provided",
            "oauth_required": True
        })

@mcp.custom_route("/.well-known/jwks.json", methods=["GET"])
async def jwks_endpoint(request: Request):
    """JWKS endpoint for OAuth compliance"""
    logger.info("JWKS endpoint called")
    
    # In production, this would return actual public keys
    # For now, return a minimal JWKS for compliance
    return JSONResponse({
        "keys": [
            {
                "kty": "oct",
                "use": "sig",
                "kid": "image-tool-mcp-key",
                "alg": "HS256"
            }
        ]
    })

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def handle_image_output(
    b64_data: str, 
    output_format: str, 
    output_mode: Optional[str], 
    file_path: Optional[str],
    temp_dir: Path,
    ctx: Context = None
) -> str:
    """Standardized image output handling for all tools"""
    
    # Auto-detect output mode based on transport if not specified
    if output_mode is None:
        output_mode = get_default_output_mode()
    
    # FORCE appropriate mode based on transport
    if _transport_mode != "stdio":
        # Remote: Always use URLs for best UX
        if output_mode != "url":
            if ctx: await ctx.info(f"Using URL mode for remote usage (better UX)")
            output_mode = "url"
    else:
        # Local: Always use files for direct filesystem access
        if output_mode != "file":
            if ctx: await ctx.info(f"Using file mode for local usage (direct filesystem access)")
            output_mode = "file"
    
    if output_mode == "file":
        # Save to specified file path
        if not file_path:
            raise ValueError("file_path required for file output")
        save_path = Path(file_path)
        await save_base64_image(b64_data, save_path, output_format.upper())
        if ctx: await ctx.info(f"Image saved to: {save_path}")
        return str(save_path)
        
    else:  # url mode
        # Create download URL
        if ctx: await ctx.info("Creating download URL")
        
        # Save to publicly accessible download directory
        filename = f"generated_{uuid.uuid4().hex[:8]}.{output_format}"
        download_path = temp_dir / "downloads" / filename
        download_path.parent.mkdir(exist_ok=True)
        
        # Clean up old files before saving new one
        await cleanup_old_downloads(temp_dir, max_age_hours=24, max_total_size_mb=100)
        
        await save_base64_image(b64_data, download_path, output_format.upper())
        
        # Return download URL
        download_url = f"https://web-production-472cb.up.railway.app/download/{filename}"
        if ctx: await ctx.info(f"Full-size image available at: {download_url}")
        return f"Image generated successfully! Download URL: {download_url}"

async def cleanup_old_downloads(temp_dir: Path, max_age_hours: int = 24, max_total_size_mb: int = 100) -> None:
    """Clean up old download files to prevent disk space issues"""
    downloads_dir = temp_dir / "downloads"
    if not downloads_dir.exists():
        return
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    max_total_bytes = max_total_size_mb * 1024 * 1024
    
    # Get all files with their stats
    files_info = []
    total_size = 0
    
    for file_path in downloads_dir.glob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            files_info.append({
                'path': file_path,
                'mtime': stat.st_mtime,
                'size': stat.st_size,
                'age_hours': (current_time - stat.st_mtime) / 3600
            })
            total_size += stat.st_size
    
    cleaned_count = 0
    
    # Remove files older than max_age_hours
    for file_info in files_info[:]:
        if file_info['age_hours'] > max_age_hours:
            try:
                file_info['path'].unlink()
                files_info.remove(file_info)
                total_size -= file_info['size']
                cleaned_count += 1
                logger.info(f"Cleaned up old file: {file_info['path'].name} (age: {file_info['age_hours']:.1f}h)")
            except Exception as e:
                logger.warning(f"Failed to delete {file_info['path']}: {e}")
    
    # If still over size limit, remove oldest files
    if total_size > max_total_bytes:
        files_info.sort(key=lambda x: x['mtime'])  # Oldest first
        for file_info in files_info:
            if total_size <= max_total_bytes:
                break
            try:
                file_info['path'].unlink()
                total_size -= file_info['size']
                cleaned_count += 1
                logger.info(f"Cleaned up for space: {file_info['path'].name} (size: {file_info['size']/1024/1024:.1f}MB)")
            except Exception as e:
                logger.warning(f"Failed to delete {file_info['path']}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleanup complete: removed {cleaned_count} files, {total_size/1024/1024:.1f}MB remaining")

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

async def get_file_path(file_input: str) -> str:
    """Handle file input and return local file path"""
    if not file_input or not file_input.strip():
        raise ValueError("File input cannot be empty")

    file_input = file_input.strip()

    # Security: Prevent path traversal attacks
    if ".." in file_input:
        raise ValueError("Invalid file path: potential security risk")

    app_context = get_app_context()

    # Handle base64 data URLs by converting to temp file
    if file_input.startswith('data:'):
        return await handle_file_input(file_input, app_context)
    
    # Handle Google Drive URLs by converting to direct download URLs
    if file_input.startswith('drive://') or 'drive.google.com' in file_input or 'docs.google.com' in file_input:
        file_id = extract_file_id_from_url(file_input)
        if file_id:
            # Convert to direct download URL
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return await handle_file_input(direct_url, app_context)
        else:
            raise ValueError(f"Invalid Google Drive URL: {file_input}")
    
    # Handle HTTP URLs by downloading to temp file
    if file_input.startswith('http://') or file_input.startswith('https://'):
        return await handle_file_input(file_input, app_context)
    
    # Handle absolute paths
    if os.path.isabs(file_input):
        if _transport_mode == "stdio":
            if os.path.exists(file_input):
                return file_input
            else:
                raise FileNotFoundError(f"File not found: {file_input}")
        else:
            # In remote mode, we cannot access local file paths.
            raise ValueError(f"Cannot access local file path '{file_input}' in remote mode. Please provide a public URL or base64-encoded image.")

    # If we get here, it's an invalid input
    raise ValueError("Invalid file path: must be an absolute path, base64 data, HTTP URL, or Google Drive URL (drive://, drive.google.com, docs.google.com)")

# =============================================================================
# IMAGE PROCESSING TOOLS
# =============================================================================

@mcp.tool()
async def create_image(
    prompt: str,
    ctx: Context = None,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    background: Literal["transparent", "opaque", "auto"] = "auto", 
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_compression: Optional[int] = None,
    moderation: Literal["auto", "low"] = "auto",
    n: int = 1,
    output_mode: Optional[Literal["file", "url"]] = None,
    file_path: Optional[str] = None
) -> Union[str, list[str]]:
    """Generate images from text prompts using OpenAI's latest gpt-image-1 model.
    
    REMOTE USAGE: Always returns download URLs (best UX, no connection issues).
    LOCAL USAGE: Always returns file paths (direct filesystem access).
    
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
        output_mode: OPTIONAL - Auto-detected ('url' for remote, 'file' for local)
        file_path: OPTIONAL - Only needed for local file mode
        
    Returns:
        Download URLs (remote) or file paths (local)
    """
    # Get application context
    app_context = get_app_context()
    
    client = app_context.openai_client
    check_openai_client(client)
    temp_dir = app_context.temp_dir
    
    # Auto-detect output mode based on transport if not specified
    if output_mode is None:
        output_mode = get_default_output_mode()
    
    # FORCE appropriate mode based on transport
    if _transport_mode != "stdio":
        # Remote: Always use URLs for best UX
        if output_mode != "url":
            if ctx: await ctx.info(f"Using URL mode for remote usage (better UX)")
            output_mode = "url"
    else:
        # Local: Always use files for direct filesystem access
        if output_mode != "file":
            if ctx: await ctx.info(f"Using file mode for local usage (direct filesystem access)")
            output_mode = "file"
    
    # No size limitations needed for URL mode (files stored on server)
    
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
    if n > 1 and ctx:
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
        if ctx: await ctx.info(f"Generating {n} image(s) with prompt: {prompt[:100]}...")
        response = await client.images.generate(**params)
        if ctx: await ctx.info(f"OpenAI API call completed successfully, processing {len(response.data)} images")
        
        # Process results
        images = []
        file_paths = []
        
        for i, image_data in enumerate(response.data):
            if n > 1 and ctx:
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
                if ctx: await ctx.info(f"Image saved to: {save_path}")
                
            elif output_mode == "url":
                # Create download URL (for remote usage or when explicitly requested)
                if ctx: await ctx.info("Creating download URL")
                
                # Save to publicly accessible download directory
                filename = f"generated_{uuid.uuid4().hex[:8]}.{output_format}"
                download_path = temp_dir / "downloads" / filename
                download_path.parent.mkdir(exist_ok=True)
                
                # Clean up old files before saving new one
                await cleanup_old_downloads(temp_dir, max_age_hours=24, max_total_size_mb=100)
                
                await save_base64_image(b64_data, download_path, output_format.upper())
                
                # Return download URL
                download_url = f"https://web-production-472cb.up.railway.app/download/{filename}"
                images.append(f"Image generated successfully! Download URL: {download_url}")
                
                if ctx: await ctx.info(f"Full-size image available at: {download_url}")
                
            # Note: base64 mode removed - not needed for either remote or local usage
        
        # Log response preparation
        if ctx: 
            if output_mode == "file":
                await ctx.info(f"Returning file paths: {file_paths}")
            else:  # url mode
                await ctx.info(f"Returning {len(images)} download URL(s)")
        
        # Return results
        if output_mode == "file":
            return file_paths if n > 1 else file_paths[0]
        else:
            return images if n > 1 else images[0]
            
    except Exception as e:
        if ctx: await ctx.error(f"Image generation failed: {str(e)}")
        raise ValueError(f"Failed to generate image: {str(e)}")

# =============================================================================
# MISSING TOOLS RESTORATION
# =============================================================================

@mcp.tool()
async def edit_image(
    image: str,
    prompt: str,
    mask: Optional[str] = None,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_mode: Optional[Literal["file", "url"]] = None,
    file_path: Optional[str] = None,
    ctx: Context = None
) -> Union[str, list[str]]:
    """Edit existing images using masks and text prompts.
    
    REMOTE USAGE: Always returns download URLs (best UX, no connection issues).
    LOCAL USAGE: Always returns file paths (direct filesystem access).
    
    Args:
        image: Original image (file path or base64)
        prompt: Text description of desired changes
        mask: Optional mask image for selective editing (same formats as image)
        model: Image generation model
        size: Image dimensions
        quality: Generation quality level
        output_format: Image format
        output_mode: OPTIONAL - Auto-detected ('url' for remote, 'file' for local)
        file_path: OPTIONAL - Only needed for local file mode
        
    Returns:
        Download URLs (remote) or file paths (local)
    """
    app_context = get_app_context()
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    # Auto-detect output mode based on transport if not specified
    if output_mode is None:
        output_mode = get_default_output_mode()
    
    # Get file paths
    image_path = await get_file_path(image)
    mask_path = await get_file_path(mask) if mask else None
    
    # Load images as bytes
    image_base64, _ = await load_image_as_base64(image_path)
    image_bytes = base64.b64decode(image_base64)
    
    try:
        if ctx: await ctx.info(f"Editing image with prompt: {prompt[:100]}...")
        
        # Use httpx for multipart/form-data request
        import httpx
        import time
        start_time = time.time()
        
        # Get API key from client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and hasattr(client, 'api_key'):
            api_key = client.api_key
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare files for multipart upload
        files = {
            "image": ("image.png", image_bytes, "image/png"),
        }
        
        if mask_path:
            mask_base64, _ = await load_image_as_base64(mask_path)
            mask_bytes = base64.b64decode(mask_base64)
            files["mask"] = ("mask.png", mask_bytes, "image/png")
        
        # Prepare data
        data = {
            "model": model,
            "prompt": prompt,
            "n": "1",
            "size": size if size != "auto" else "1024x1024",
        }
        
        # Add quality parameter if using gpt-image-1
        if model == "gpt-image-1" and quality != "auto":
            data["quality"] = quality
        
        # Add response format for base64
        data["response_format"] = "b64_json"
        
        # Make the request using httpx
        async with httpx.AsyncClient(timeout=120.0) as http_client:
            response = await http_client.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                files=files,
                data=data
            )
            
            api_time = time.time() - start_time
            if ctx: await ctx.info(f"OpenAI API call completed in {api_time:.2f}s")
            
            if response.status_code != 200:
                error_msg = f"API error {response.status_code}: {response.text}"
                if ctx: await ctx.error(error_msg)
                raise ValueError(error_msg)
            
            response_data = response.json()
            b64_data = response_data["data"][0]["b64_json"]
        
        # Use standardized output handling
        result = await handle_image_output(
            b64_data, output_format, output_mode, file_path, temp_dir, ctx
        )
        
        total_time = time.time() - start_time
        if ctx: await ctx.info(f"Total edit_image processing time: {total_time:.2f}s")
        
        return result
            
    except Exception as e:
        if ctx: await ctx.error(f"Edit image failed: {str(e)}")
        raise ValueError(f"Failed to edit image: {str(e)}")

@mcp.tool()
async def generate_variations(
    image: str,
    n: int = 1,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_mode: Optional[Literal["file", "url"]] = None,
    file_path: Optional[str] = None,
    ctx: Context = None
) -> Union[str, list[str]]:
    """Generate variations of existing images.
    
    REMOTE USAGE: Always returns download URLs (best UX, no connection issues).
    LOCAL USAGE: Always returns file paths (direct filesystem access).
    
    Args:
        image: Original image (file path or base64)
        n: Number of variations to generate (1-10)
        model: Image generation model
        size: Image dimensions
        quality: Generation quality level
        output_format: Image format
        output_mode: OPTIONAL - Auto-detected ('url' for remote, 'file' for local)
        file_path: OPTIONAL - Only needed for local file mode
        
    Returns:
        Download URLs (remote) or file paths (local)
    """
    app_context = get_app_context()
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    # Auto-detect output mode based on transport if not specified
    if output_mode is None:
        output_mode = get_default_output_mode()
    
    if n < 1 or n > 10:
        raise ValueError("Number of variations must be between 1 and 10")
    
    image_path = await get_file_path(image)
    image_base64, _ = await load_image_as_base64(image_path)
    
    # Prepare image for API (OpenAI client expects file objects, not base64 strings)
    image_bytes = base64.b64decode(image_base64)
    image_file = io.BytesIO(image_bytes)
    image_file.name = "image.png"  # Required for OpenAI client
    
    params = {
        "model": model,
        "image": image_file,
        "n": n,
        "size": size if size != "auto" else "1024x1024"
    }
    
    try:
        if ctx: await ctx.info(f"Generating {n} variation(s)...")
        response = await client.images.create_variation(**params)
        
        # Process results using standardized output handling
        results = []
        for i, img_data in enumerate(response.data):
            b64_data = img_data.b64_json
            
            # For multiple variations, modify file path with index
            variation_file_path = file_path
            if file_path and n > 1:
                path = Path(file_path)
                variation_file_path = str(path.parent / f"{path.stem}_{i+1}{path.suffix}")
            
            result = await handle_image_output(
                b64_data, output_format, output_mode, variation_file_path, temp_dir, ctx
            )
            results.append(result)
        
        return results if n > 1 else results[0]
        
    except Exception as e:
        raise ValueError(f"Failed to generate variations: {str(e)}")

@mcp.tool()
async def extract_text(
    image: str,
    language: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Extract text from images using OCR.
    
    Supports local files and base64 data.
    
    Args:
        image: Image to extract text from (file path or base64)
        language: Language hint for better accuracy (e.g., 'eng', 'spa', 'fra')
        
    Returns:
        Extracted text with confidence scores and bounding boxes
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    prompt = "Extract all text from this image. If the image contains multiple languages, identify each language. Provide the text with confidence scores and approximate locations if possible."
    
    try:
        if ctx: await ctx.info("Extracting text from image...")
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        text_content = response.choices[0].message.content
        
        return {
            "success": True,
            "text": text_content,
            "source_file": str(image_path),
            "language_hint": language
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }



@mcp.tool()
async def batch_process(
    images: List[str],
    operation: Literal["extract_text", "analyze_image", "image_metadata"],
    operation_params: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    """Process multiple images with the same operation.
    
    Supports local files, base64 data, and Google Drive URLs.
    
    Args:
        images: List of images (file paths, base64 data, or Google Drive URLs)
        operation: Operation to perform on all images
        operation_params: Parameters for the operation
        
    Returns:
        Batch processing results for all images
    """
    results = []
    total_images = len(images)
    
    for i, image in enumerate(images):
        if ctx:
            await ctx.report_progress(i + 1, total_images, f"Processing image {i + 1}/{total_images}")
        
        try:
            if operation == "extract_text":
                result = await extract_text(image, **operation_params)
            elif operation == "analyze_image":
                result = await analyze_image(image, **operation_params)
            elif operation == "image_metadata":
                result = await image_metadata(image, **operation_params)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
            results.append({
                "image": image,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "image": image,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_processed": total_images,
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "results": results
    }

@mcp.tool()
async def image_metadata(
    image: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Extract comprehensive metadata and properties from images.
    
    Supports local files and base64 data.
    
    Args:
        image: Image to analyze (file path or base64)
        
    Returns:
        Detailed image metadata including EXIF, dimensions, format, etc.
    """
    if not PIL_AVAILABLE:
        return {"success": False, "error": "PIL/Pillow not available"}
    
    try:
        image_path = await get_file_path(image)
        
        with PILImage.open(image_path) as img:
            metadata = {
                "success": True,
                "file_path": str(image_path),
                "format": img.format,
                "mode": img.mode,
                "size": {
                    "width": img.width,
                    "height": img.height,
                    "total_pixels": img.width * img.height
                },
                "color_info": {
                    "bands": img.getbands(),
                    "palette": img.palette is not None
                }
            }
            
            # Get file system info
            stat = image_path.stat()
            metadata["file_info"] = {
                "size_bytes": stat.st_size,
                "size_kb": stat.st_size / 1024,
                "size_mb": stat.st_size / 1024 / 1024,
                "modified": stat.st_mtime,
                "created": stat.st_ctime
            }
            
            # Try to get EXIF data
            try:
                exif = img._getexif()
                if exif:
                    exif_data = {}
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = str(value)
                    metadata["exif"] = exif_data
            except Exception:
                metadata["exif"] = "No EXIF data found"
            
            return metadata
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def analyze_image(
    image: str,
    prompt: str = "Describe this image in detail, including objects, people, scenery, colors, mood, and any text visible.",
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    detail: Literal["low", "high", "auto"] = "auto",
    ctx: Context = None
) -> str:
    """Analyze an image using OpenAI's Vision API to extract detailed information.
    
    Supports local files, base64 data, and HTTP URLs.
    
    Args:
        image: Image to analyze (file path, base64 data, or HTTP URL)
        prompt: Analysis prompt (what to look for in the image)
        model: Vision model to use (gpt-4o, gpt-4o-mini, etc.)
        max_tokens: Maximum tokens in response
        detail: Image detail level for processing
        
    Returns:
        Detailed analysis of the image content
    """
    app_context = get_app_context()
    client = app_context.openai_client
    check_openai_client(client)
    
    try:
        if ctx: await ctx.info(f"Analyzing image with model {model}...")
        
        # Get file path (handles local files, URLs, and base64)
        file_path = await get_file_path(image)
        
        # Load image file as base64
        base64_data, mime_type = await load_image_as_base64(file_path)
        image_url = f"data:{mime_type};base64,{base64_data}"
        
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
        if ctx: await ctx.info("Image analysis completed successfully")
        return analysis
        
    except Exception as e:
        error_msg = f"Failed to analyze image: {str(e)}"
        if ctx: await ctx.error(error_msg)
        raise ValueError(error_msg)


# =============================================================================
# GOOGLE DRIVE TOOLS FOR IMAGES
# =============================================================================

@mcp.tool()
async def search_images(
    query: str = "",
    folder_id: Optional[str] = None,
    max_results: int = 50,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search for image files in Google Drive.
    
    Args:
        query: Search query (e.g., "vacation photos", "name contains 'screenshot'")
        folder_id: Optional folder to search within
        max_results: Maximum number of results (1-1000)
        
    Returns:
        List of image files with download URLs and metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN or GOOGLE_SERVICE_ACCOUNT_JSON", "success": False}
        
        if ctx:
            await ctx.info(f"Searching for images: {query}")
        
        # Build search query for images
        q_parts = ["trashed=false", "mimeType contains 'image/'"]
        
        if query:
            q_parts.append(f"name contains '{query}'")
        
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")
        
        q = " and ".join(q_parts)
        
        results = drive_service.files().list(
            q=q,
            pageSize=min(max_results, 1000),
            fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, thumbnailLink)"
        ).execute()
        
        items = results.get('files', [])
        
        # Convert to direct download URLs
        processed_items = []
        for item in items:
            file_id = item['id']
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            processed_items.append({
                "id": file_id,
                "name": item['name'],
                "mime_type": item['mimeType'],
                "size": int(item.get('size', 0)),
                "created_time": item['createdTime'],
                "modified_time": item['modifiedTime'],
                "web_view_link": item.get('webViewLink', ''),
                "thumbnail_link": item.get('thumbnailLink', ''),
                "direct_download_url": direct_url,
                "drive_url": f"drive://{file_id}"  # For use with other tools
            })
        
        return {
            "success": True,
            "query": query,
            "total_results": len(processed_items),
            "images": processed_items
        }
        
    except Exception as e:
        error_msg = f"Error searching images: {str(e)}"
        if ctx: await ctx.error(error_msg)
        logger.error(error_msg)
        return {"error": error_msg, "success": False}

@mcp.tool()
async def upload_image(
    image_path: Optional[str] = None,
    image_data: Optional[str] = None,
    image_url: Optional[str] = None,
    filename: Optional[str] = None,
    folder_id: Optional[str] = None,
    description: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Upload an image to Google Drive.
    
    Args:
        image_path: Local image path (for local files)
        image_data: Base64 encoded image data (for generated images)
        image_url: HTTP URL to image (for remote images)
        filename: Name for the uploaded file
        folder_id: Google Drive folder ID (default: root)
        description: Optional description for the image
        
    Returns:
        Upload result with file ID and metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN or GOOGLE_SERVICE_ACCOUNT_JSON", "success": False}
        
        # Validate input - exactly one source must be provided
        input_count = sum(bool(x) for x in [image_path, image_data, image_url])
        if input_count != 1:
            return {"error": "Exactly one of image_path, image_data, or image_url must be provided", "success": False}
        
        # Handle different input types
        temp_file_path = None
        actual_filename = None
        
        if image_path:
            # Local file path mode
            file_path = await get_file_path(image_path)
            actual_filename = filename or Path(file_path).name
            temp_file_path = file_path
            
            if ctx:
                await ctx.info(f"Uploading local image: {image_path}")
                
        elif image_data:
            # Base64 data mode
            if not filename:
                return {"error": "filename is required when using image_data", "success": False}
            
            try:
                # Handle data URLs
                if image_data.startswith('data:'):
                    header, data = image_data.split(',', 1)
                    file_bytes = base64.b64decode(data)
                    
                    # Extract MIME type for extension
                    mime_match = re.search(r'data:([^;]+)', header)
                    if mime_match and not filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                        mime_type = mime_match.group(1)
                        extension = mimetypes.guess_extension(mime_type) or '.png'
                        if not filename.endswith(extension):
                            filename = f"{filename}{extension}"
                else:
                    # Raw base64
                    file_bytes = base64.b64decode(image_data)
                    if not filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                        filename = f"{filename}.png"
                
                # Create temporary file
                actual_filename = filename
                safe_filename = re.sub(r'[^\w\-_.]', '_', actual_filename)
                temp_file_path = app_context.temp_dir / f"upload_{secrets.token_urlsafe(8)}_{safe_filename}"
                
                # Write decoded data to temp file
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(file_bytes)
                
                if ctx:
                    await ctx.info(f"Created temp file from base64 data: {temp_file_path}")
                    
            except Exception as e:
                return {"error": f"Failed to decode base64 data: {str(e)}", "success": False}
                
        elif image_url:
            # URL fetch mode
            if not filename:
                return {"error": "filename is required when using image_url", "success": False}
            
            if not app_context.http_client:
                return {"error": "HTTP client not available for URL fetching", "success": False}
            
            try:
                # Fetch image from URL
                response = await app_context.http_client.get(image_url)
                response.raise_for_status()
                
                # Create temporary file
                actual_filename = filename
                safe_filename = re.sub(r'[^\w\-_.]', '_', actual_filename)
                temp_file_path = app_context.temp_dir / f"url_{secrets.token_urlsafe(8)}_{safe_filename}"
                
                # Write fetched data to temp file
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(response.content)
                
                if ctx:
                    await ctx.info(f"Downloaded image from URL: {image_url}")
                    
            except Exception as e:
                return {"error": f"Failed to fetch image from URL: {str(e)}", "success": False}
        
        # Determine mime type
        mime_type, _ = mimetypes.guess_type(actual_filename)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/png'  # Default for images
        
        # Prepare file metadata
        file_metadata = {
            'name': actual_filename,
            'description': description or f"Image uploaded via Image-Tool-MCP Server"
        }
        
        # Set parent folder if specified
        if folder_id and folder_id != "root":
            file_metadata['parents'] = [folder_id]
        
        # Create media upload object
        media = MediaFileUpload(
            str(temp_file_path),
            mimetype=mime_type,
            resumable=True
        )
        
        # Upload file
        request = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, mimeType, size, createdTime, webViewLink'
        )
        
        # Execute upload with progress tracking
        response = None
        while response is None:
            status, response = request.next_chunk()
            if ctx and status:
                await ctx.report_progress(int(status.progress() * 100), 100)
        
        if ctx:
            await ctx.info(f"Upload completed: {response['name']}")
        
        # Generate direct download URL
        file_id = response['id']
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return {
            "success": True,
            "file_id": file_id,
            "name": response['name'],
            "mime_type": response['mimeType'],
            "size": int(response.get('size', 0)),
            "created_time": response['createdTime'],
            "web_view_link": response['webViewLink'],
            "direct_download_url": direct_url,
            "drive_url": f"drive://{file_id}",
            "folder_id": folder_id or "root"
        }
        
    except Exception as e:
        error_msg = f"Error uploading image: {str(e)}"
        if ctx: await ctx.error(error_msg)
        logger.error(error_msg)
        return {"error": error_msg, "success": False}

@mcp.tool()
async def get_image_from_drive(
    file_id_or_url: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get direct download URL for an image in Google Drive.
    
    Args:
        file_id_or_url: Google Drive file ID, share URL, or drive:// URL
        
    Returns:
        Direct download URL and image metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN or GOOGLE_SERVICE_ACCOUNT_JSON", "success": False}
        
        # Extract file ID from various URL formats
        file_id = extract_file_id_from_url(file_id_or_url)
        if not file_id:
            file_id = file_id_or_url  # Assume it's a direct file ID
        
        if ctx:
            await ctx.info(f"Getting image info for: {file_id}")
        
        # Get file metadata
        file_metadata = drive_service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink, thumbnailLink"
        ).execute()
        
        # Verify it's an image
        mime_type = file_metadata['mimeType']
        if not mime_type.startswith('image/'):
            return {"error": f"File is not an image (MIME type: {mime_type})", "success": False}
        
        # Generate direct download URL
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return {
            "success": True,
            "file_id": file_id,
            "name": file_metadata['name'],
            "mime_type": mime_type,
            "size": int(file_metadata.get('size', 0)),
            "created_time": file_metadata['createdTime'],
            "modified_time": file_metadata['modifiedTime'],
            "web_view_link": file_metadata.get('webViewLink', ''),
            "thumbnail_link": file_metadata.get('thumbnailLink', ''),
            "direct_download_url": direct_url,
            "drive_url": f"drive://{file_id}"
        }
        
    except Exception as e:
        error_msg = f"Error getting image from Drive: {str(e)}"
        if ctx: await ctx.error(error_msg)
        logger.error(error_msg)
        return {"error": error_msg, "success": False}

# =============================================================================
# GOOGLE DRIVE TOOLS FOR IMAGES
# =============================================================================

@mcp.tool()
async def search_images(
    query: str = "",
    folder_id: Optional[str] = None,
    max_results: int = 50,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search for image files in Google Drive.
    
    Args:
        query: Search query (e.g., "vacation photos", "name contains 'screenshot'")
        folder_id: Optional folder to search within
        max_results: Maximum number of results (1-1000)
        
    Returns:
        List of image files with download URLs and metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN environment variable", "success": False}
        
        if ctx:
            await ctx.info(f"Searching for images: {query}")
        
        # Build search query for images
        q_parts = [
            "trashed=false",
            "(mimeType contains 'image/')"
        ]
        
        if query:
            q_parts.append(f"name contains '{query}'")
        
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")
        
        q = " and ".join(q_parts)
        
        results = drive_service.files().list(
            q=q,
            pageSize=min(max_results, 1000),
            fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink)"
        ).execute()
        
        items = results.get('files', [])
        
        # Convert to direct download URLs
        processed_items = []
        for item in items:
            file_id = item['id']
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            processed_items.append({
                "id": file_id,
                "name": item['name'],
                "mime_type": item['mimeType'],
                "size": int(item.get('size', 0)),
                "created_time": item['createdTime'],
                "modified_time": item['modifiedTime'],
                "web_view_link": item.get('webViewLink', ''),
                "direct_download_url": direct_url,
                "drive_url": f"drive://{file_id}"  # For use with other tools
            })
        
        return {
            "success": True,
            "query": query,
            "total_results": len(processed_items),
            "images": processed_items
        }
        
    except Exception as e:
        error_msg = f"Error searching images: {str(e)}"
        if ctx: await ctx.error(error_msg)
        logger.error(error_msg)
        return {"error": error_msg, "success": False}

@mcp.tool()
async def upload_image(
    image_path: Optional[str] = None,
    image_data: Optional[str] = None,
    image_url: Optional[str] = None,
    filename: Optional[str] = None,
    folder_id: Optional[str] = None,
    description: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Upload an image to Google Drive.
    
    Args:
        image_path: Local image path (for local files)
        image_data: Base64 encoded image data (for generated images)
        image_url: HTTP URL to image (for remote images)
        filename: Name for the uploaded file
        folder_id: Google Drive folder ID (default: root)
        description: Optional description for the image
        
    Returns:
        Upload result with file ID and metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN environment variable", "success": False}
        
        # Validate input - exactly one source must be provided
        input_count = sum(bool(x) for x in [image_path, image_data, image_url])
        if input_count != 1:
            return {"error": "Exactly one of image_path, image_data, or image_url must be provided", "success": False}
        
        # Handle different input types
        temp_file_path = None
        actual_filename = None
        
        if image_path:
            # Local file path mode
            file_path = await get_file_path(image_path, app_context)
            actual_filename = filename or Path(file_path).name
            temp_file_path = file_path
            
            if ctx:
                await ctx.info(f"Uploading local image: {image_path}")
                
        elif image_data:
            # Base64 data mode
            if not filename:
                return {"error": "filename is required when using image_data", "success": False}
            
            try:
                # Handle data URLs
                if image_data.startswith('data:'):
                    header, data = image_data.split(',', 1)
                    file_bytes = base64.b64decode(data)
                    
                    # Extract MIME type for extension
                    mime_match = re.search(r'data:([^;]+)', header)
                    if mime_match and not filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        mime_type = mime_match.group(1)
                        extension = mimetypes.guess_extension(mime_type) or '.png'
                        if not filename.endswith(extension):
                            filename = f"{filename}{extension}"
                else:
                    # Raw base64
                    file_bytes = base64.b64decode(image_data)
                    if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        filename = f"{filename}.png"
                
                # Create temporary file
                actual_filename = filename
                safe_filename = re.sub(r'[^\w\-_.]', '_', actual_filename)
                temp_file_path = app_context.temp_dir / f"upload_{secrets.token_urlsafe(8)}_{safe_filename}"
                
                # Write decoded data to temp file
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(file_bytes)
                
                if ctx:
                    await ctx.info(f"Created temp file from base64 data: {temp_file_path}")
                    
            except Exception as e:
                return {"error": f"Failed to decode base64 data: {str(e)}", "success": False}
                
        elif image_url:
            # URL fetch mode
            if not filename:
                return {"error": "filename is required when using image_url", "success": False}
            
            if not app_context.http_client:
                return {"error": "HTTP client not available for URL fetching", "success": False}
            
            try:
                # Fetch image from URL
                response = await app_context.http_client.get(image_url)
                response.raise_for_status()
                
                # Create temporary file
                actual_filename = filename
                safe_filename = re.sub(r'[^\w\-_.]', '_', actual_filename)
                temp_file_path = app_context.temp_dir / f"url_{secrets.token_urlsafe(8)}_{safe_filename}"
                
                # Write fetched data to temp file
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(response.content)
                
                if ctx:
                    await ctx.info(f"Downloaded image from URL: {image_url}")
                    
            except Exception as e:
                return {"error": f"Failed to fetch image from URL: {str(e)}", "success": False}
        
        # Determine mime type
        mime_type, _ = mimetypes.guess_type(actual_filename)
        if not mime_type or not mime_type.startswith('image/'):
            # Default based on extension
            ext = Path(actual_filename).suffix.lower()
            mime_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            mime_type = mime_map.get(ext, 'image/png')  # Default to PNG
        
        # Prepare file metadata
        file_metadata = {
            'name': actual_filename,
            'description': description or f"Image uploaded via Image-Tool-MCP Server"
        }
        
        # Set parent folder if specified
        if folder_id and folder_id != "root":
            file_metadata['parents'] = [folder_id]
        
        # Create media upload object
        media = MediaFileUpload(
            str(temp_file_path),
            mimetype=mime_type,
            resumable=True
        )
        
        # Upload file
        request = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, mimeType, size, createdTime, webViewLink'
        )
        
        # Execute upload with progress tracking
        response = None
        while response is None:
            status, response = request.next_chunk()
            if ctx and status:
                await ctx.report_progress(int(status.progress() * 100), 100)
        
        if ctx:
            await ctx.info(f"Upload completed: {response['name']}")
        
        # Generate direct download URL
        file_id = response['id']
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return {
            "success": True,
            "file_id": file_id,
            "name": response['name'],
            "mime_type": response['mimeType'],
            "size": int(response.get('size', 0)),
            "created_time": response['createdTime'],
            "web_view_link": response['webViewLink'],
            "direct_download_url": direct_url,
            "drive_url": f"drive://{file_id}",
            "folder_id": folder_id or "root"
        }
        
    except Exception as e:
        error_msg = f"Error uploading image: {str(e)}"
        if ctx: await ctx.error(error_msg)
        logger.error(error_msg)
        return {"error": error_msg, "success": False}

@mcp.tool()
async def get_image_from_drive(
    file_id_or_url: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get direct download URL for an image in Google Drive.
    
    Args:
        file_id_or_url: Google Drive file ID, share URL, or drive:// URL
        
    Returns:
        Direct download URL and image metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN environment variable", "success": False}
        
        # Extract file ID from various URL formats
        file_id = extract_file_id_from_url(file_id_or_url)
        if not file_id:
            file_id = file_id_or_url  # Assume it's a direct file ID
        
        if ctx:
            await ctx.info(f"Getting image info for: {file_id}")
        
        # Get file metadata
        file_metadata = drive_service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink"
        ).execute()
        
        # Verify it's an image
        mime_type = file_metadata['mimeType']
        if not mime_type.startswith('image/'):
            return {"error": f"File is not an image (MIME type: {mime_type})", "success": False}
        
        # Generate direct download URL
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return {
            "success": True,
            "file_id": file_id,
            "name": file_metadata['name'],
            "mime_type": mime_type,
            "size": int(file_metadata.get('size', 0)),
            "created_time": file_metadata['createdTime'],
            "modified_time": file_metadata['modifiedTime'],
            "web_view_link": file_metadata.get('webViewLink', ''),
            "direct_download_url": direct_url,
            "drive_url": f"drive://{file_id}"
        }
        
    except Exception as e:
        error_msg = f"Error getting image from Drive: {str(e)}"
        if ctx: await ctx.error(error_msg)
        logger.error(error_msg)
        return {"error": error_msg, "success": False}


# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Tool MCP Server")
    parser.add_argument("--transport", default="streamable-http", choices=["http", "stdio", "streamable-http"], 
                       help="Transport method (http or stdio)")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                       help="Host to bind to (http mode only)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)),
                       help="Port to bind to (http mode only)")
    
    args = parser.parse_args()
    
    # Set global transport mode for automatic output mode detection
    _transport_mode = args.transport
    
    if args.transport == "stdio":
        # In stdio mode, logging goes to stderr (configured above)
        mcp.run(transport="stdio")
    else:
        # HTTP mode - show configuration
        logger.info("=" * 50)
        logger.info("STARTING IMAGE TOOL MCP SERVER")
        logger.info("=" * 50)
        logger.info(f"Server configuration:")
        logger.info(f"  Transport: {args.transport}")
        logger.info(f"  Host: {args.host}")
        logger.info(f"  Port: {args.port}")
        logger.info(f"  OpenAI configured: {_global_app_context.openai_client is not None if _global_app_context else 'Context not initialized'}")
        logger.info(f"  Temp directory: {_global_app_context.temp_dir if _global_app_context else 'Not set'}")
        logger.info("Available endpoints:")
        logger.info(f"  Health check: http://{args.host}:{args.port}/health")
        logger.info(f"  Root: http://{args.host}:{args.port}/")
        logger.info(f"  MCP: http://{args.host}:{args.port}/mcp/")
        logger.info("=" * 50)
        
        try:
            logger.info("Attempting to start FastMCP server...")
            mcp.run(transport=args.transport, host=args.host, port=args.port, path="/mcp")
        except Exception as e:
            logger.error(f"CRITICAL ERROR - Failed to start server: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise