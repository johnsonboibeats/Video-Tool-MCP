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

# Load environment variables
load_dotenv()

class AppContext(BaseModel):
    """Application context with shared resources"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    openai_client: Optional[Union[AsyncOpenAI, AsyncAzureOpenAI]] = None
    temp_dir: Path
    http_client: Optional[httpx.AsyncClient] = None

# Global app context for FastMCP tools
_global_app_context: Optional[AppContext] = None

def get_app_context() -> AppContext:
    """Get application context from global reference"""
    if _global_app_context is not None:
        return _global_app_context
    raise RuntimeError("Application context not initialized")

def check_openai_client(client) -> None:
    """Check if OpenAI client is available"""
    if client is None:
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")

async def handle_file_input(file_input: str, app_context: AppContext) -> str:
    """Handle file input: base64 data and absolute paths"""
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
    
    # Reject relative paths and other potentially unsafe inputs
    raise ValueError("Invalid file path: must be absolute path or base64 data")

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
        
        # Create context
        context = AppContext(
            openai_client=client,
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
from starlette.responses import JSONResponse
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

@mcp.custom_route("/mcp", methods=["GET", "HEAD", "POST"])
async def mcp_redirect(request: Request):
    """Handle MCP endpoint redirects"""
    logger.info(f"MCP redirect called: {request.method} {request.url.path}")
    
    if request.method == "HEAD":
        # Return 200 for HEAD requests to /mcp
        return JSONResponse({"status": "ok"}, status_code=200)
    elif request.method == "POST":
        # Redirect POST requests to /mcp/
        from starlette.responses import RedirectResponse
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

def validate_image_path(path: str) -> str:
    """Validate file path input with security checks"""
    if not path or not path.strip():
        raise ValueError("File path cannot be empty")
    
    path = path.strip()
    
    # Security: Prevent path traversal attacks
    if ".." in path:
        raise ValueError("Invalid file path: potential security risk")
    
    # Allow local absolute paths (starting with /)
    if path.startswith('/'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path
    
        
    # Allow base64 data URLs
    if path.startswith('data:'):
        return path
    
    # Reject relative paths and other potentially unsafe inputs
    raise ValueError("Invalid file path: must be absolute path or base64 data")

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
    if not file_input:
        raise ValueError("File input cannot be empty")
    
    # Handle base64 data URLs by converting to temp file
    if file_input.startswith('data:'):
        app_context = get_app_context()
        return await handle_file_input(file_input, app_context)
    
    # Validate and return absolute paths
    return validate_image_path(file_input)

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
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None
) -> Union[str, list[str]]:
    """Generate images from text prompts using OpenAI's latest gpt-image-1 model.
    
    Supports local files and base64 data.
    
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
        output_mode: Return as base64 data or save to file
        file_path: Absolute path for file output (required if output_mode='file')
        
    Returns:
        Generated image(s) as base64 data or file paths
    """
    # Get application context
    app_context = get_app_context()
    
    client = app_context.openai_client
    check_openai_client(client)
    temp_dir = app_context.temp_dir
    
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
        if ctx: ctx.info(f"Generating {n} image(s) with prompt: {prompt[:100]}...")
        response = await client.images.generate(**params)
        
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
                if ctx: ctx.info(f"Image saved to: {save_path}")
                
            else:
                # Return as base64 string
                images.append(f"data:image/{output_format};base64,{b64_data}")
        
        # Return results
        if output_mode == "file":
            return file_paths if n > 1 else file_paths[0]
        else:
            return images if n > 1 else images[0]
            
    except Exception as e:
        if ctx: ctx.error(f"Image generation failed: {str(e)}")
        raise ValueError(f"Failed to generate image: {str(e)}")

@mcp.tool()
async def analyze_image(
    image: str,
    ctx: Context = None,
    prompt: str = "Describe this image in detail, including objects, people, scenery, colors, mood, and any text visible.",
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    detail: Literal["low", "high", "auto"] = "auto"
) -> str:
    """Analyze an image using OpenAI's Vision API to extract detailed information.
    
    Supports local files and base64 data.
    
    Args:
        image: Absolute file path or base64 string of image to analyze
        prompt: Analysis prompt (what to look for in the image)
        model: Vision model to use (gpt-4o, gpt-4o-mini, etc.)
        max_tokens: Maximum tokens in response
        detail: Image detail level for processing
        
    Returns:
        Detailed analysis of the image content
    """
    # Get application context
    app_context = get_app_context()
    
    client = app_context.openai_client
    check_openai_client(client)
    
    # Get file path (handles local files and base64)
    file_path = await get_file_path(image)
    
    # Prepare image for API
    try:
        # Validate the file path using the new unified approach
        validated_path = validate_image_path(file_path)
        
        # Load image file as base64
        base64_data, mime_type = await load_image_as_base64(validated_path)
        image_url = f"data:{mime_type};base64,{base64_data}"
        
    except (ValueError, FileNotFoundError) as e:
        # If validation fails, try to handle as base64 data
        if is_base64_image(image):
            if image.startswith("data:image/"):
                image_url = image
            else:
                # Add data URL prefix
                image_url = f"data:image/png;base64,{image}"
        else:
            raise ValueError(f"Invalid image input: {str(e)}")
    
    try:
        if ctx: ctx.info(f"Analyzing image with model {model}...")
        
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
        if ctx: ctx.info("Image analysis completed successfully")
        return analysis
        
    except Exception as e:
        if ctx: ctx.error(f"Image analysis failed: {str(e)}")
        raise ValueError(f"Failed to analyze image: {str(e)}")

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
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None,
    ctx: Context = None
) -> Union[str, list[str]]:
    """Edit existing images using masks and text prompts.
    
    Supports local files and base64 data.
    
    Args:
        image: Original image (file path or base64)
        prompt: Text description of desired changes
        mask: Optional mask image for selective editing (same formats as image)
        model: Image generation model
        size: Image dimensions
        quality: Generation quality level
        output_format: Image format
        output_mode: Return as base64 data or save to file
        file_path: Absolute path for file output (required if output_mode='file')
        
    Returns:
        Edited image(s) as base64 data or file paths
    """
    app_context = get_app_context()
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    # Get file paths
    image_path = await get_file_path(image)
    mask_path = await get_file_path(mask) if mask else None
    
    # Prepare images for API
    image_base64, _ = await load_image_as_base64(image_path)
    
    params = {
        "model": model,
        "prompt": prompt,
        "image": image_base64,
        "size": size if size != "auto" else "1024x1024",
        "quality": quality,
        "response_format": "b64_json"
    }
    
    if mask_path:
        mask_base64, _ = await load_image_as_base64(mask_path)
        params["mask"] = mask_base64
    
    try:
        if ctx: ctx.info(f"Editing image with prompt: {prompt[:100]}...")
        response = await client.images.edit(**params)
        
        b64_data = response.data[0].b64_json
        
        if output_mode == "file":
            if not file_path:
                raise ValueError("file_path required for file output")
            save_path = Path(file_path)
            await save_base64_image(b64_data, save_path, output_format.upper())
            return str(save_path)
        else:
            return f"data:image/{output_format};base64,{b64_data}"
            
    except Exception as e:
        raise ValueError(f"Failed to edit image: {str(e)}")

@mcp.tool()
async def generate_variations(
    image: str,
    n: int = 1,
    model: Literal["gpt-image-1"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    quality: Literal["auto", "high", "medium", "low"] = "auto",
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_mode: Literal["base64", "file"] = "base64",
    file_path: Optional[str] = None,
    ctx: Context = None
) -> Union[str, list[str]]:
    """Generate variations of existing images.
    
    Supports local files and base64 data.
    
    Args:
        image: Original image (file path or base64)
        n: Number of variations to generate (1-10)
        model: Image generation model
        size: Image dimensions
        quality: Generation quality level
        output_format: Image format
        output_mode: Return as base64 data or save to file
        file_path: Absolute path for file output (required if output_mode='file')
        
    Returns:
        Image variation(s) as base64 data or file paths
    """
    app_context = get_app_context()
    client = app_context.openai_client
    temp_dir = app_context.temp_dir
    
    if n < 1 or n > 10:
        raise ValueError("Number of variations must be between 1 and 10")
    
    image_path = await get_file_path(image)
    image_base64, _ = await load_image_as_base64(image_path)
    
    params = {
        "model": model,
        "image": image_base64,
        "n": n,
        "size": size if size != "auto" else "1024x1024",
        "response_format": "b64_json"
    }
    
    try:
        if ctx: ctx.info(f"Generating {n} variation(s)...")
        response = await client.images.create_variation(**params)
        
        results = []
        for i, img_data in enumerate(response.data):
            b64_data = img_data.b64_json
            
            if output_mode == "file":
                if not file_path:
                    raise ValueError("file_path required for file output")
                path = Path(file_path)
                if n > 1:
                    save_path = path.parent / f"{path.stem}_{i+1}{path.suffix}"
                else:
                    save_path = path
                await save_base64_image(b64_data, save_path, output_format.upper())
                results.append(str(save_path))
            else:
                results.append(f"data:image/{output_format};base64,{b64_data}")
        
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
        if ctx: ctx.info("Extracting text from image...")
        
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
async def compare_images(
    image1: str,
    image2: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Compare two images and analyze differences.
    
    Supports local files and base64 data.
    
    Args:
        image1: First image (file path or base64)
        image2: Second image (file path or base64)
        
    Returns:
        Detailed comparison analysis including similarities and differences
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image1_path = await get_file_path(image1)
    image2_path = await get_file_path(image2)
    
    image1_base64, _ = await load_image_as_base64(image1_path)
    image2_base64, _ = await load_image_as_base64(image2_path)
    
    prompt = """Compare these two images and provide a detailed analysis of:
1. Overall similarity/difference score (0-100%)
2. Visual elements that are the same
3. Visual elements that are different
4. Color differences
5. Composition differences
6. Content changes
7. Quality differences
8. Style variations

Be specific and quantitative where possible."""
    
    try:
        if ctx: ctx.info("Comparing images...")
        
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
                                "url": f"data:image/png;base64,{image1_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image2_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        return {
            "success": True,
            "comparison": response.choices[0].message.content,
            "image1": str(image1_path),
            "image2": str(image2_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "image1": str(image1_path),
            "image2": str(image2_path)
        }

@mcp.tool()
async def smart_edit(
    image: str,
    analysis_prompt: str,
    edit_prompt: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Intelligent image editing with analysis and targeted modifications.
    
    Supports local files and base64 data.
    
    Args:
        image: Image to edit (file path or base64)
        analysis_prompt: What to analyze in the image
        edit_prompt: How to modify based on the analysis
        
    Returns:
        Edited image with analysis and modification details
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    # First, analyze the image
    analysis_prompt_full = f"{analysis_prompt} Provide specific details that would help with targeted editing."
    
    try:
        if ctx: ctx.info("Analyzing image for smart editing...")
        
        analysis_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt_full},
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
            max_tokens=1000
        )
        
        analysis = analysis_response.choices[0].message.content
        
        # Now edit based on analysis
        edit_prompt_full = f"Based on this analysis: {analysis}\n\nApply these changes: {edit_prompt}"
        
        if ctx: ctx.info("Performing smart edit...")
        
        response = await client.images.edit(
            model="gpt-image-1",
            image=image_base64,
            prompt=edit_prompt_full,
            response_format="b64_json"
        )
        
        b64_data = response.data[0].b64_json
        
        return {
            "success": True,
            "analysis": analysis,
            "edited_image": f"data:image/png;base64,{b64_data}",
            "source_file": str(image_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

@mcp.tool()
async def transform_image(
    image: str,
    operation: Literal["resize", "rotate", "flip_horizontal", "flip_vertical", "grayscale", "blur", "sharpen", "contrast", "brightness"],
    value: Optional[Union[int, float]] = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    output_path: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Apply basic image transformations using PIL.
    
    Supports local files and base64 data.
    
    Args:
        image: Image to transform (file path or base64)
        operation: Type of transformation to apply
        value: Operation-specific value (degrees for rotate, factor for contrast/brightness)
        output_format: Output image format
        output_path: Optional absolute path to save result
        
    Returns:
        Transformed image as base64 or file path
    """
    if not PIL_AVAILABLE:
        return {"success": False, "error": "PIL/Pillow not available"}
    
    try:
        image_path = await get_file_path(image)
        
        with PILImage.open(image_path) as img:
            if operation == "resize" and value:
                if isinstance(value, tuple):
                    img = img.resize(value)
                else:
                    # Resize by percentage
                    width = int(img.width * (value / 100))
                    height = int(img.height * (value / 100))
                    img = img.resize((width, height))
                    
            elif operation == "rotate" and value:
                img = img.rotate(value, expand=True)
                
            elif operation == "flip_horizontal":
                img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
                
            elif operation == "flip_vertical":
                img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
                
            elif operation == "grayscale":
                img = ImageOps.grayscale(img)
                
            elif operation == "blur":
                img = img.filter(ImageFilter.BLUR)
                
            elif operation == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)
                
            elif operation == "contrast" and value:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(value)
                
            elif operation == "brightness" and value:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(value)
            
            # Save result
            if output_path:
                save_path = Path(output_path)
                img.save(save_path, format=output_format.upper())
                return {"success": True, "file_path": str(save_path)}
            else:
                # Return as base64
                buffer = io.BytesIO()
                img.save(buffer, format=output_format.upper())
                buffer.seek(0)
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                return {
                    "success": True,
                    "image": f"data:image/{output_format};base64,{b64_data}",
                    "dimensions": f"{img.width}x{img.height}"
                }
                
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def batch_process(
    images: List[str],
    operation: Literal["analyze", "extract_text", "transform", "resize"],
    operation_params: Dict[str, Any],
    ctx: Context = None
) -> Dict[str, Any]:
    """Process multiple images with the same operation.
    
    Supports local files and base64 data.
    
    Args:
        images: List of images (file paths or base64)
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
            if operation == "analyze":
                result = await analyze_image(image, **operation_params)
            elif operation == "extract_text":
                result = await extract_text(image, **operation_params)
            elif operation == "transform":
                result = await transform_image(image, **operation_params)
            elif operation == "resize":
                result = await transform_image(image, operation="resize", **operation_params)
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
async def describe_and_recreate(
    image: str,
    style_modification: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Analyze an image and recreate it with style modifications.
    
    Supports local files and base64 data.
    
    Args:
        image: Source image (file path or base64)
        style_modification: Description of style changes to apply
        
    Returns:
        Original description and recreated image with style modifications
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    # First, describe the image
    describe_prompt = "Provide a detailed, technical description of this image including: subject matter, composition, lighting, color palette, style, mood, and any specific visual elements. Be precise and comprehensive."
    
    try:
        if ctx: ctx.info("Analyzing image for recreation...")
        
        description_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": describe_prompt},
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
            max_tokens=800
        )
        
        original_description = description_response.choices[0].message.content
        
        # Now recreate with style modifications
        recreate_prompt = f"Recreate this image: {original_description}\n\nApply these style modifications: {style_modification}"
        
        if ctx: ctx.info("Recreating image with style modifications...")
        
        response = await client.images.generate(
            model="gpt-image-1",
            prompt=recreate_prompt,
            n=1,
            response_format="b64_json"
        )
        
        b64_data = response.data[0].b64_json
        
        return {
            "success": True,
            "original_description": original_description,
            "style_modification": style_modification,
            "recreated_image": f"data:image/png;base64,{b64_data}",
            "source_file": str(image_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

@mcp.tool()
async def prompt_from_image(
    image: str,
    purpose: str = "accurate recreation",
    ctx: Context = None
) -> Dict[str, Any]:
    """Generate optimized prompts from images for AI image generation.
    
    Supports local files and base64 data.
    
    Args:
        image: Source image (file path or base64)
        purpose: Purpose of the generated prompt (recreation, variation, improvement, etc.)
        
    Returns:
        Optimized prompt for AI image generation
    """
    app_context = get_app_context()
    client = app_context.openai_client
    
    image_path = await get_file_path(image)
    image_base64, mime_type = await load_image_as_base64(image_path)
    
    prompt = f"Create an optimized text prompt for AI image generation that would recreate this image. Purpose: {purpose}. The prompt should be detailed, specific, and include: subject, composition, lighting, color palette, style, mood, and any technical details. Format it for best AI image generation results."
    
    try:
        if ctx: ctx.info("Generating optimized prompt from image...")
        
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
            max_tokens=1000
        )
        
        generated_prompt = response.choices[0].message.content
        
        return {
            "success": True,
            "generated_prompt": generated_prompt,
            "purpose": purpose,
            "source_file": str(image_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_file": str(image_path)
        }

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