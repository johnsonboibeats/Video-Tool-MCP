#!/usr/bin/env python3
"""
Video Tool MCP Server - Google Veo3 Video Generation
"""

# Debug logging for Railway deployment
import os
import sys
print(f"🚀 Starting Video-Tool-MCP Server...")
print(f"🔧 Python version: {sys.version}")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🌍 Environment check:")
print(f"   GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
print(f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
print(f"   PORT: {os.getenv('PORT', '8080')}")

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


try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("Httpx not available - some features may be disabled")

# Google Gen AI SDK imports (for Veo3 video generation)
try:
    print("📦 Importing Google Gen AI SDK...")
    from google import genai as genai_sdk  # alias to avoid name clash
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
    print("✅ Google Gen AI SDK imported successfully")
except Exception as e:
    GENAI_AVAILABLE = False
    print(f"❌ Google Gen AI SDK import failed: {e}")
    print("⚠️ Video generation via Veo3 will be disabled")

# Vertex AI (Gemini) generative models (for analysis) with graceful fallback
try:
    # Prefer GA module path
    from vertexai.generative_models import GenerativeModel, Part  # type: ignore
    VERTEX_GEN_AVAILABLE = True
except Exception:
    try:
        from vertexai.preview.generative_models import GenerativeModel, Part  # type: ignore
        VERTEX_GEN_AVAILABLE = True
    except Exception:
        VERTEX_GEN_AVAILABLE = False
        logger.warning("Vertex generative models not available - Gemini analysis will be disabled")

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
    download_semaphore: Optional[asyncio.Semaphore] = None
    download_cache: Dict[str, Dict[str, Any]] = {}
    # Gemini API configuration
    gemini_api_key: Optional[str] = None
    gemini_configured: bool = False

# Global app context for FastMCP tools
_global_app_context: Optional[AppContext] = None

# Global transport mode detection
_transport_mode: str = "http"  # Default to http (remote)

# Download policies
MAX_DOWNLOAD_SIZE_MB: int = int(os.getenv("MAX_DOWNLOAD_SIZE_MB", "50"))
DOWNLOAD_CACHE_TTL_SECONDS: int = int(os.getenv("DOWNLOAD_CACHE_TTL_SECONDS", "1800"))
CLEANUP_INTERVAL_SECONDS: int = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "1800"))

# Allowed MIME types for videos
ALLOWED_VIDEO_MIME_PREFIXES = {
    "video/",
}

# Simple metrics
METRICS: Dict[str, Any] = {
    "http_download_requests": 0,
    "http_download_bytes_total": 0,
    "http_download_errors": 0,
    "http_cache_hits": 0,
    "temp_cleanup_runs": 0
}

_cleanup_task_started: bool = False

async def _periodic_cleanup_loop():
    global METRICS
    try:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            ctx = get_app_context()
            try:
                cleanup_temp_dir(ctx.temp_dir, max_age_hours=24, max_total_size_mb=200)
                METRICS["temp_cleanup_runs"] += 1
            except Exception:
                # best-effort
                pass
    except asyncio.CancelledError:
        return

def cleanup_temp_dir(temp_dir: Path, max_age_hours: int = 24, max_total_size_mb: int = 200) -> None:
    """Best-effort cleanup of temp directory by age and total size."""
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    if not temp_dir.exists():
        return
    # Remove old files anywhere under temp_dir
    for p in temp_dir.glob("**/*"):
        if p.is_file():
            try:
                if now - p.stat().st_mtime > max_age_seconds:
                    p.unlink(missing_ok=True)
            except Exception:
                continue
    # Enforce total size
    files = [p for p in temp_dir.glob("**/*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)  # oldest first
    total = sum(p.stat().st_size for p in files)
    limit = max_total_size_mb * 1024 * 1024
    for p in files:
        if total <= limit:
            break
        try:
            sz = p.stat().st_size
            p.unlink(missing_ok=True)
            total -= sz
        except Exception:
            continue

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
        r'drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)',
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
    
    # Handle HTTP URLs with preflight, whitelist, caching
    if file_input.startswith('http://') or file_input.startswith('https://'):
        try:
            if not app_context.http_client:
                raise ValueError("HTTP client not available for downloading URLs")

            # Cache hit?
            cached = None
            try:
                cached = app_context.download_cache.get(file_input)
            except Exception:
                cached = None
            now_ts = time.time()
            if cached and (now_ts - cached.get('ts', 0)) < DOWNLOAD_CACHE_TTL_SECONDS:
                p = Path(cached['path'])
                if p.exists():
                    return cached['path']

            # Limit concurrency
            sem = app_context.download_semaphore or asyncio.Semaphore(5)
            async with sem:
                # HEAD preflight
                content_type = None
                content_length = None
                try:
                    head = await app_context.http_client.head(file_input)
                    if head.status_code < 400:
                        content_type = head.headers.get('content-type')
                        cl = head.headers.get('content-length')
                        if cl and cl.isdigit():
                            content_length = int(cl)
                except Exception:
                    pass

                max_bytes = MAX_DOWNLOAD_SIZE_MB * 1024 * 1024
                if content_length and content_length > max_bytes:
                    raise ValueError(f"File too large: {content_length/1024/1024:.1f} MB > {MAX_DOWNLOAD_SIZE_MB} MB")

                # Download with streaming + simple retries
                last_err = None
                for attempt in range(3):
                    try:
                        async with app_context.http_client.stream("GET", file_input) as response:
                            response.raise_for_status()
                            ct = response.headers.get('content-type') or content_type or ''
                            if not ct:
                                raise ValueError("Missing Content-Type for URL download")
                            if not any(ct.startswith(pfx) for pfx in ALLOWED_IMAGE_MIME_PREFIXES):
                                raise ValueError(f"Unsupported MIME type: {ct}")
                            extension = mimetypes.guess_extension(ct.split(';')[0]) or '.bin'
                            if extension == '.bin':
                                from urllib.parse import urlparse
                                url_ext = Path(urlparse(file_input).path).suffix
                                if url_ext:
                                    extension = url_ext
                            if ct.startswith('image/') and extension == '.bin':
                                if 'png' in ct:
                                    extension = '.png'
                                elif 'jpeg' in ct or 'jpg' in ct:
                                    extension = '.jpg'
                                elif 'webp' in ct:
                                    extension = '.webp'
                                elif 'gif' in ct:
                                    extension = '.gif'

                            temp_path = app_context.temp_dir / f"temp_download_{int(time.time())}{extension}"
                            downloaded = 0
                            async with aiofiles.open(temp_path, 'wb') as f:
                                async for chunk in response.aiter_bytes(64 * 1024):
                                    if not chunk:
                                        continue
                                    downloaded += len(chunk)
                                    if downloaded > max_bytes:
                                        raise ValueError(f"File too large after download: {downloaded/1024/1024:.1f} MB")
                                    await f.write(chunk)


                        # Cache path
                        try:
                            app_context.download_cache[file_input] = {"path": str(temp_path), "ts": now_ts}
                        except Exception:
                            pass

                        logger.info(f"Successfully downloaded file to: {temp_path}, size: {downloaded} bytes")
                        return str(temp_path)
                    except Exception as e:
                        last_err = e
                        await asyncio.sleep(0.5 * (attempt + 1))
                raise ValueError(f"Failed to download file from URL: {last_err}")
        except Exception as e:
            logger.error(f"Download error for {file_input}: {e}")
            raise
    
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
            logger.warning("No OpenAI API key found - some tools may not function")
            client = None
        
        # Setup temp directory
        temp_dir = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir())) / "video_tool_mcp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory: {temp_dir}")
        
        # Create tuned HTTP client with connection limits and keep-alive
        http_client = None
        if HTTPX_AVAILABLE:
            limits = httpx.Limits(
                max_connections=int(os.getenv("HTTP_MAX_CONNECTIONS", "50")),
                max_keepalive_connections=int(os.getenv("HTTP_MAX_KEEPALIVE", "20"))
            )
            http_client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=limits
            )
        
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
        
        # Initialize Vertex AI (optional)
        # Support credentials via GOOGLE_APPLICATION_CREDENTIALS_JSON on platforms like Railway
        try:
            adc_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            adc_path_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if adc_json and not adc_path_env:
                try:
                    # Determine content: raw JSON or base64-encoded
                    content = adc_json
                    if not adc_json.strip().startswith("{"):
                        try:
                            decoded = base64.b64decode(adc_json).decode("utf-8", errors="ignore")
                            if decoded.strip().startswith("{"):
                                content = decoded
                        except Exception:
                            pass
                    # Write to temp file
                    adc_path = Path(tempfile.gettempdir()) / "vertex_adc.json"
                    adc_path.write_text(content)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc_path)
                    logger.info(f"Wrote GOOGLE_APPLICATION_CREDENTIALS to {adc_path}")
                except Exception as e:
                    logger.warning(f"Failed to materialize GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        except Exception:
            pass

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_configured = bool(gemini_api_key)
        if gemini_configured:
            logger.info("Gemini API configuration detected")
        else:
            logger.warning("Gemini API key not set - video generation will fail (default is veo-3.0-generate-001)")

        # Create context
        context = AppContext(
            openai_client=client,
            drive_service=drive_service,
            temp_dir=temp_dir,
            http_client=http_client,
            download_semaphore=asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))),
            download_cache={},
            gemini_api_key=gemini_api_key,
            gemini_configured=gemini_configured
        )
        
        # Set global context for FastMCP tools
        global _global_app_context
        _global_app_context = context
        
        # Cleanup temp dir (synchronous, safe at startup)
        try:
            cleanup_temp_dir(temp_dir, max_age_hours=24, max_total_size_mb=200)
        except Exception as e:
            logger.warning(f"Temp cleanup failed: {e}")

        # Periodic cleanup task will be started on first health or metrics request

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
    logger.warning("Server will start with limited functionality")
    # Create a minimal context to allow server to start
    _global_app_context = AppContext(
        openai_client=None,
        temp_dir=Path(tempfile.gettempdir()) / "video_tool_mcp",
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
mcp = FastMCP("Video Tool MCP")

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
        base_url = get_base_url(request)
        response_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "server": "Video Tool MCP Server",
            "oauth_enabled": True,
            "oauth_issuer": get_oauth_config(base_url)["issuer"],
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

        # Expose Gemini API status
        try:
            response_data["gemini_configured"] = _global_app_context.gemini_configured if _global_app_context else False
            response_data["gemini_api_key_present"] = bool(_global_app_context.gemini_api_key) if _global_app_context else False
        except Exception:
            response_data["gemini_configured"] = False
            response_data["gemini_api_key_present"] = False
            
        logger.info(f"Health check response: {response_data}")
        # Lazy-start periodic cleanup loop after server has an event loop
        global _cleanup_task_started
        if not _cleanup_task_started:
            try:
                asyncio.create_task(_periodic_cleanup_loop())
                _cleanup_task_started = True
            except Exception:
                pass
        return JSONResponse(response_data)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@mcp.custom_route("/metrics", methods=["GET"])
async def metrics_endpoint(request: Request):
    """Return simple internal metrics counters."""
    # Lazy-start periodic cleanup loop
    global _cleanup_task_started
    if not _cleanup_task_started:
        try:
            asyncio.create_task(_periodic_cleanup_loop())
            _cleanup_task_started = True
        except Exception:
            pass
    return JSONResponse(METRICS)

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request):
    """Root endpoint for basic connectivity test"""
    logger.info("Root endpoint called")
    return JSONResponse({
        "message": "Video Tool MCP Server is running",
        "health_check": "/health",
        "mcp_endpoint": "/mcp/"
    })


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
def get_base_url(request: Request) -> str:
    """Get the correct base URL with HTTPS for Railway/proxy deployments"""
    if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https":
        return f"https://{request.url.netloc}/"
    return f"{request.base_url}"

def get_oauth_config(base_url: str) -> Dict[str, Any]:
    """Generate OAuth configuration with correct server URLs"""
    return {
        "issuer": base_url.rstrip('/'),
        "authorization_endpoint": f"{base_url}oauth/authorize",
        "token_endpoint": f"{base_url}oauth/token",
        "userinfo_endpoint": f"{base_url}oauth/userinfo",
        "revocation_endpoint": f"{base_url}oauth/revoke",
        "jwks_uri": f"{base_url}.well-known/jwks.json",
        "response_types_supported": ["code"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["HS256"],
        "scopes_supported": ["read", "write", "video_processing"],
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

def generate_access_token(user_id: str, scopes: list = None, audience: str = None, issuer: Optional[str] = None) -> str:
    """Generate JWT access token with audience validation"""
    payload = {
        "sub": user_id,
        "iss": issuer or "https://video-tool-mcp",
        "aud": audience or "claude",  # RFC 8707: Include specific audience
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
        "scope": " ".join(scopes or ["read", "write"])
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def generate_refresh_token(user_id: str, issuer: Optional[str] = None) -> str:
    """Generate refresh token"""
    payload = {
        "sub": user_id,
        "iss": issuer or "https://video-tool-mcp",
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
    server_uri = get_base_url(request).rstrip('/')
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
    # Derive issuer from stored resource base if possible
    issuer = resource.rstrip('/') if resource else None
    access_token = generate_access_token(user_id, scopes, audience=resource, issuer=issuer)
    refresh_token = generate_refresh_token(user_id, issuer=issuer)
    
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
    issuer = resource.rstrip('/') if resource else None
    access_token = generate_access_token(user_id, scopes, audience=resource, issuer=issuer)
    new_refresh_token = generate_refresh_token(user_id, issuer=issuer)
    
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
    
    base_url = get_base_url(request)
    return JSONResponse({
        "sub": user_id,
        "iss": get_oauth_config(base_url)["issuer"],
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
    base_url = get_base_url(request)
    return JSONResponse(get_oauth_config(base_url))

@mcp.custom_route("/.well-known/oauth-authorization-server/mcp", methods=["GET"])
async def oauth_discovery_mcp(request: Request):
    """OAuth discovery endpoint for MCP path"""
    return await oauth_discovery(request)

@mcp.custom_route("/.well-known/oauth-protected-resource/mcp", methods=["GET"])
async def oauth_protected_resource(request: Request):
    """OAuth protected resource endpoint for Claude compatibility"""
    logger.info("OAuth protected resource endpoint called")
    base_url = get_base_url(request)
    return JSONResponse({
        "resource": "mcp",
        "scopes": ["read", "write", "video_processing"],
        "token_endpoint": f"{base_url}oauth/token",
        "userinfo_endpoint": f"{base_url}oauth/userinfo"
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
    
    base_url = get_base_url(request)
    cfg = get_oauth_config(base_url)
    return JSONResponse({
        "oauth_enabled": True,
        "issuer": cfg["issuer"],
        "supported_grant_types": cfg["grant_types_supported"],
        "supported_scopes": cfg["scopes_supported"],
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
                "kid": "video-tool-mcp-key",
                "alg": "HS256"
            }
        ]
    })

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def _upload_to_drive(
    file_data: bytes,
    filename: str, 
    description: str,
    folder_id: str = "1y8eWyr68gPTiFTS2GuNODZp9zx4kg4FC",  # Downloads folder default
    ctx: Context = None
) -> str:
    """Unified helper to upload any file type to Google Drive"""
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            raise ValueError("Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN")
        
        if ctx:
            await ctx.info(f"Uploading {filename} to Google Drive")
        
        # Create temporary file from bytes
        temp_path = app_context.temp_dir / f"upload_{uuid.uuid4().hex[:8]}_{filename}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write bytes to temp file
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(file_data)
        
        # Upload to Google Drive
        file_metadata = {
            'name': filename,
            'description': description,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(str(temp_path), resumable=True)
        
        request = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name, mimeType, size, createdTime, webViewLink"
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
        
        web_view_link = response['webViewLink']
        if ctx:
            await ctx.info(f"Uploaded successfully: {web_view_link}")
            
        return web_view_link
        
    except Exception as e:
        # Clean up temp file on error
        try:
            if 'temp_path' in locals():
                temp_path.unlink()
        except:
            pass
        
        error_msg = f"Failed to upload {filename}: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

def cleanup_downloads_dir(temp_dir: Path, max_age_hours: int = 24, max_total_size_mb: int = 100) -> None:
    """Specialized cleanup for the downloads subdir; kept for parity with logs."""
    downloads_dir = temp_dir / "downloads"
    if not downloads_dir.exists():
        return
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    max_total_bytes = max_total_size_mb * 1024 * 1024
    files_with_stats = []
    total_size = 0
    for p in downloads_dir.glob("*"):
        if p.is_file():
            st = p.stat()
            files_with_stats.append((p, st.st_mtime, st.st_size))
            total_size += st.st_size
    # Remove by age
    for (p, mtime, size) in list(files_with_stats):
        if now - mtime > max_age_seconds:
            try:
                p.unlink(missing_ok=True)
                total_size -= size
            except Exception:
                continue
    # Enforce total size
    if total_size > max_total_bytes:
        files_with_stats.sort(key=lambda t: t[1])
        for (p, _mtime, size) in files_with_stats:
            if total_size <= max_total_bytes:
                break
            try:
                p.unlink(missing_ok=True)
                total_size -= size
            except Exception:
                continue


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
    
    # Handle Google Drive URLs - use direct API access if possible
    if file_input.startswith('drive://') or 'drive.google.com' in file_input or 'docs.google.com' in file_input:
        file_id = extract_file_id_from_url(file_input)
        if file_id:
            # Fallback to HTTP download
            download_urls = [
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
                f"https://drive.google.com/uc?export=download&id={file_id}",
                f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            ]
            
            # Try each URL format until one works
            last_error = None
            for url in download_urls:
                try:
                    return await handle_file_input(url, app_context)
                except Exception as e:
                    last_error = e
                    continue
            
            # If all URLs failed, raise the last error
            raise ValueError(f"Failed to download from Google Drive (ID: {file_id}). Last error: {last_error}")
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
            raise ValueError(f"Cannot access local file path '{file_input}' in remote mode. Please provide a public URL or base64-encoded data.")

    # If we get here, it's an invalid input
    raise ValueError("Invalid file path: must be an absolute path, base64 data, HTTP URL, or Google Drive URL (drive://, drive.google.com, docs.google.com)")

# =============================================================================
# BACKGROUND OPERATION MONITORING
# =============================================================================

# Global storage for operation results (in production, use Redis or database)
_operation_results = {}

async def _monitor_and_upload_operation(
    operation_name: str,
    prompt: str,
    selected_model: str, 
    folder_id: str,
    expected_count: int
):
    """Background task to monitor operation completion and auto-upload videos"""
    try:
        logger.info(f"🎬 Background monitoring started for operation: {operation_name}")
        
        # Store initial status
        logger.info(f"📝 Storing operation {operation_name} in background monitoring")
        _operation_results[operation_name] = {
            "status": "in_progress",
            "prompt": prompt,
            "model": selected_model,
            "expected_videos": expected_count,
            "videos": [],
            "errors": [],
            "started_at": time.time()
        }
        
        # Initialize client for background task
        app_context = get_app_context()
        if not GENAI_AVAILABLE or not app_context.gemini_configured:
            raise ValueError("Gemini API not configured for background processing")
        
        client = genai_sdk.Client(api_key=app_context.gemini_api_key)
        
        # Poll operation status until completion
        operation = genai_types.GenerateVideosOperation(name=operation_name)
        
        while not operation.done:
            await asyncio.sleep(20)  # Check every 20 seconds
            try:
                operation = client.operations.get(operation)
                logger.info(f"🔄 Operation {operation_name} status: {'completed' if operation.done else 'in_progress'}")
            except Exception as e:
                logger.error(f"Failed to poll operation {operation_name}: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                continue
        
        logger.info(f"✅ Operation {operation_name} completed! Processing videos...")
        
        # Extract and upload videos
        if not hasattr(operation, 'response') or not hasattr(operation.response, 'generated_videos'):
            raise ValueError("No videos found in operation response")
        
        generated_videos = operation.response.generated_videos
        uploaded_videos = []
        
        for i, generated_video in enumerate(generated_videos):
            try:
                # Generate filename
                if len(generated_videos) > 1:
                    filename = f"generated_video_{uuid.uuid4().hex[:8]}_{i+1}.mp4"
                else:
                    filename = f"generated_video_{uuid.uuid4().hex[:8]}.mp4"
                
                logger.info(f"📥 Downloading video {i+1}/{len(generated_videos)}: {filename}")
                
                # Download video data using correct API method
                client.files.download(file=generated_video.video)
                
                # Save video to temporary file, then read the bytes
                temp_video_path = app_context.temp_dir / f"temp_{filename}"
                generated_video.video.save(str(temp_video_path))
                
                # Read the video file as bytes
                async with aiofiles.open(temp_video_path, 'rb') as f:
                    video_data = await f.read()
                
                # Clean up temp file
                try:
                    temp_video_path.unlink()
                except Exception:
                    pass
                
                logger.info(f"📤 Uploading {filename} to Google Drive ({len(video_data)} bytes)")
                
                # Upload to Google Drive
                web_view_link = await _upload_to_drive(
                    file_data=video_data,
                    filename=filename,
                    description=f"Generated video with {selected_model}: {prompt[:100]}...",
                    folder_id=folder_id,
                    ctx=None  # No context in background task
                )
                
                uploaded_videos.append({
                    "filename": filename,
                    "url": web_view_link,
                    "size_bytes": len(video_data)
                })
                
                logger.info(f"✅ Video {i+1} uploaded successfully: {web_view_link}")
                
            except Exception as e:
                error_msg = f"Failed to process video {i+1}: {str(e)}"
                logger.error(error_msg)
                _operation_results[operation_name]["errors"].append(error_msg)
        
        # Update final status
        _operation_results[operation_name].update({
            "status": "completed",
            "videos": uploaded_videos,
            "completed_at": time.time(),
            "total_videos": len(uploaded_videos)
        })
        
        logger.info(f"🎉 Background processing completed for {operation_name}: {len(uploaded_videos)} videos uploaded")
        
    except Exception as e:
        error_msg = f"Background processing failed for {operation_name}: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback for background processing failure:")
        
        # Update error status - ensure we can still store it even if background task fails
        try:
            _operation_results[operation_name] = {
                "status": "failed", 
                "error": error_msg,
                "completed_at": time.time()
            }
            logger.info(f"❌ Stored failure status for operation {operation_name}")
        except Exception as store_error:
            logger.error(f"Failed to store error status for {operation_name}: {store_error}")

async def _monitor_multiple_operations(
    operations: List[str],
    prompt: str,
    selected_model: str,
    folder_id: str,
    expected_count: int
):
    """Background task to monitor multiple operations and combine results"""
    primary_operation = operations[0]
    
    try:
        logger.info(f"🎬 Background monitoring started for {len(operations)} operations")
        
        # Store initial status under primary operation ID
        _operation_results[primary_operation] = {
            "status": "in_progress",
            "prompt": prompt,
            "model": selected_model,
            "expected_videos": expected_count,
            "videos": [],
            "errors": [],
            "started_at": time.time(),
            "all_operations": operations,
            "completed_operations": 0
        }
        
        # Monitor each operation individually
        completed_videos = []
        all_errors = []
        
        # Initialize client for background task
        app_context = get_app_context()
        if not GENAI_AVAILABLE or not app_context.gemini_configured:
            raise ValueError("Gemini API not configured for background processing")
        
        client = genai_sdk.Client(api_key=app_context.gemini_api_key)
        
        for i, operation_name in enumerate(operations):
            try:
                logger.info(f"🔄 Monitoring operation {i+1}/{len(operations)}: {operation_name}")
                
                # Poll this operation until completion
                operation = genai_types.GenerateVideosOperation(name=operation_name)
                
                while not operation.done:
                    await asyncio.sleep(20)  # Check every 20 seconds
                    try:
                        operation = client.operations.get(operation)
                        logger.info(f"🔄 Operation {i+1} status: {'completed' if operation.done else 'in_progress'}")
                    except Exception as e:
                        logger.error(f"Failed to poll operation {operation_name}: {e}")
                        await asyncio.sleep(60)  # Wait longer on error
                        continue
                
                logger.info(f"✅ Operation {i+1} completed! Processing video...")
                
                # Extract and upload video from this operation
                if hasattr(operation, 'response') and hasattr(operation.response, 'generated_videos'):
                    generated_videos = operation.response.generated_videos
                    
                    for j, generated_video in enumerate(generated_videos):
                        try:
                            # Generate filename
                            if len(operations) > 1:
                                filename = f"generated_video_{uuid.uuid4().hex[:8]}_{i+1}_{j+1}.mp4"
                            else:
                                filename = f"generated_video_{uuid.uuid4().hex[:8]}.mp4"
                            
                            logger.info(f"📥 Downloading video from operation {i+1}: {filename}")
                            
                            # Download and save video
                            client.files.download(file=generated_video.video)
                            temp_video_path = app_context.temp_dir / f"temp_{filename}"
                            generated_video.video.save(str(temp_video_path))
                            
                            # Read video bytes
                            async with aiofiles.open(temp_video_path, 'rb') as f:
                                video_data = await f.read()
                            
                            # Clean up temp file
                            try:
                                temp_video_path.unlink()
                            except Exception:
                                pass
                            
                            logger.info(f"📤 Uploading {filename} to Google Drive ({len(video_data)} bytes)")
                            
                            # Upload to Google Drive
                            web_view_link = await _upload_to_drive(
                                file_data=video_data,
                                filename=filename,
                                description=f"Generated video {i+1} with {selected_model}: {prompt[:100]}...",
                                folder_id=folder_id,
                                ctx=None
                            )
                            
                            completed_videos.append({
                                "filename": filename,
                                "url": web_view_link,
                                "size_bytes": len(video_data),
                                "operation_index": i + 1
                            })
                            
                            logger.info(f"✅ Video {i+1} uploaded successfully: {web_view_link}")
                            
                        except Exception as e:
                            error_msg = f"Failed to process video from operation {i+1}: {str(e)}"
                            logger.error(error_msg)
                            all_errors.append(error_msg)
                
                # Update progress
                _operation_results[primary_operation]["completed_operations"] = i + 1
                
            except Exception as e:
                error_msg = f"Failed to monitor operation {i+1}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
        
        # Update final status
        _operation_results[primary_operation].update({
            "status": "completed",
            "videos": completed_videos,
            "errors": all_errors,
            "completed_at": time.time(),
            "total_videos": len(completed_videos)
        })
        
        logger.info(f"🎉 Multi-operation processing completed: {len(completed_videos)} videos uploaded")
        
    except Exception as e:
        error_msg = f"Multi-operation background processing failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback for multi-operation failure:")
        
        # Update error status
        try:
            _operation_results[primary_operation] = {
                "status": "failed", 
                "error": error_msg,
                "completed_at": time.time(),
                "all_operations": operations
            }
            logger.info(f"❌ Stored failure status for multi-operation {primary_operation}")
        except Exception as store_error:
            logger.error(f"Failed to store error status for multi-operation: {store_error}")

# =============================================================================
# VIDEO PROCESSING TOOLS
# =============================================================================

@mcp.tool()
async def create_video(
    prompt: str,
    ctx: Context = None,
    aspect_ratio: Literal["16:9", "9:16", "auto"] = "auto",
    negative_prompt: Optional[str] = None,
    n: int = 1,
    folder_id: Optional[str] = None,
    resolution: Literal["720p", "1080p", "auto"] = "auto",
    generate_audio: bool = True,
    person_generation: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate videos from text prompts using Google Veo3.
    
    Returns operation ID immediately and automatically uploads videos to Google Drive when ready.
    
    Args:
        prompt: Text description of the video to generate (max 32000 chars)
        model: Video generation model (veo-2.0-generate-001, veo-3.0-generate-001, veo-3.0-fast-generate-001)
        aspect_ratio: Video aspect ratio (16:9 or 9:16)
        negative_prompt: Elements to exclude from generation
        n: Number of videos to generate (1-4 for stable models, 1-2 for preview models)
        folder_id: Google Drive folder ID (defaults to Downloads folder)
        resolution: Video resolution (720p or 1080p)
        generate_audio: Whether to generate native audio (music, sound effects, dialogue)
        person_generation: Whether to allow human figure generation
        seed: Optional seed for reproducible generation
        
    Returns:
        Operation information with ID to check status. Videos auto-upload to Google Drive when ready.
    """
    # Get application context
    app_context = get_app_context()

    # Determine model selection (scoped to create_video only)
    env_model = os.getenv("CREATE_VIDEO_MODEL")
    # Use environment variable if set, otherwise use stable Veo3 model
    selected_model = env_model or "gemini:veo-3.0-generate-001"
    
    # Parse provider and model from standardized format
    if ":" in selected_model:
        provider, model_name = selected_model.split(":", 1)
    else:
        # Backwards compatibility for old format
        if selected_model.startswith("veo-"):
            provider, model_name = "gemini", selected_model
        else:
            provider, model_name = "gemini", selected_model  # Default to gemini
    
    # Always log model selection for debugging
    logger.info(f"🎬 VIDEO MODEL DEBUG: env='{env_model}', selected='{selected_model}', provider='{provider}', model='{model_name}'")
    if ctx: 
        await ctx.info(f"Using {provider} provider with model: {model_name}")

    # Validate inputs according to model limits
    if len(prompt) > 32000:
        raise ValueError("Prompt must be 32000 characters or less")
    
    # Validate video count limits based on model capabilities
    if model_name == "veo-2.0-generate-001":
        if n < 1 or n > 4:
            raise ValueError("Number of videos must be between 1 and 4 for Veo2")
    elif model_name in ["veo-3.0-generate-001", "veo-3.0-fast-generate-001"]:
        if n < 1 or n > 4:
            raise ValueError("Number of videos must be between 1 and 4 for stable Veo3 models")
    else:  # Preview models
        if n < 1 or n > 2:
            raise ValueError("Number of videos must be between 1 and 2 for preview models")
    
    # Progress tracking for batch generation
    if n > 1 and ctx:
        await ctx.report_progress(0, n, f"Starting generation of {n} videos...")
    
    try:
        if not GENAI_AVAILABLE or not app_context.gemini_configured:
            raise ValueError("Google Gen AI SDK not available or Gemini API key not configured; set GEMINI_API_KEY and install google-genai")
        
        # Use Google Gen AI SDK (Gemini API mode) for Veo3
        client = genai_sdk.Client(api_key=app_context.gemini_api_key)

        # Prepare video generation config
        cfg_kwargs = {}
        if aspect_ratio and aspect_ratio != "auto":
            cfg_kwargs["aspect_ratio"] = aspect_ratio
        if negative_prompt:
            cfg_kwargs["negative_prompt"] = negative_prompt

        config = genai_types.GenerateVideosConfig(**cfg_kwargs)

        if ctx:
            await ctx.info(f"Generating {n} video(s) with {provider} model: {model_name}")
        
        # Validate provider
        if provider != "gemini":
            raise ValueError(f"Unsupported provider '{provider}' for video generation. Only 'gemini' is supported.")
        
        # Generate videos based on model capabilities
        operations = []
        
        if model_name in ["veo-2.0-generate-001", "veo-3.0-generate-001", "veo-3.0-fast-generate-001"]:
            # Stable models support up to 4 videos per request with explicit n parameter
            operation = client.models.generate_videos(
                model=model_name,
                prompt=prompt,
                config=config,
                n=n  # Use the n parameter for stable models
            )
            operations.append(operation)
            if ctx: await ctx.info(f"Started generation of {n} video(s) in single request: {operation.name}")
        else:
            # Preview models: make separate API calls for each video (limited to 1 per request)
            for i in range(n):
                operation = client.models.generate_videos(
                    model=model_name,
                    prompt=prompt,
                    config=config,
                )
                operations.append(operation)
                if ctx: await ctx.info(f"Started video generation {i+1}/{n}: {operation.name}")
        
        # Return primary operation ID and start background processing for all operations
        primary_operation = operations[0]
        
        if ctx:
            await ctx.info(f"Video generation started! Primary Operation ID: {primary_operation.name}")
            await ctx.info(f"Total operations: {len(operations)} (for {n} videos)")
            await ctx.info("Videos will be automatically uploaded to Google Drive when ready.")
        
        # Start background task to monitor all operations and upload when complete
        import time
        asyncio.create_task(_monitor_multiple_operations(
            operations=[op.name for op in operations],
            prompt=prompt,
            selected_model=model_name,
            folder_id=folder_id or "1y8eWyr68gPTiFTS2GuNODZp9zx4kg4FC",
            expected_count=n
        ))
        
        # Return operation info immediately
        return {
            "operation_id": primary_operation.name,
            "all_operations": [op.name for op in operations],
            "status": "started",
            "message": f"Video generation started with {provider} {model_name}. {len(operations)} operation(s) created for {n} video(s). Videos will be automatically uploaded to Google Drive when ready.",
            "estimated_time": "1-6 minutes per video",
            "expected_videos": n,
            "check_status_with": f"get_video_operation_status('{primary_operation.name}')"
        }
            
    except Exception as e:
        if ctx: await ctx.error(f"Video generation failed: {str(e)}")
        raise ValueError(f"Failed to generate video: {str(e)}")

# Duplicate create_video function removed - using the updated Veo3 version above

@mcp.tool()
async def create_video_from_image(
    image: str,
    prompt: str,
    ctx: Context = None,
    model: str = "gemini:veo-3.0-generate-001",
    aspect_ratio: Literal["16:9", "9:16"] = "16:9",
    negative_prompt: Optional[str] = None,
    folder_id: Optional[str] = None,
    resolution: Literal["720p", "1080p"] = "720p",
    generate_audio: bool = True,
    person_generation: bool = True,
    seed: Optional[int] = None
) -> str:
    """Generate video from an initial image using Google Veo3.
    
    Automatically uploads the generated video to Google Drive and returns web view URL.
    
    Args:
        image: Initial image (file path, URL, or base64 data)
        prompt: Text description of how to animate the image (max 32000 chars)
        model: Video generation model
        aspect_ratio: Video aspect ratio (16:9 or 9:16)
        negative_prompt: Elements to exclude from generation
        folder_id: Google Drive folder ID (defaults to Downloads folder)
        resolution: Video resolution (720p or 1080p)
        generate_audio: Whether to generate native audio
        person_generation: Whether to allow human figure generation
        seed: Optional seed for reproducible generation
        
    Returns:
        Google Drive web view URL for generated video
    """
    app_context = get_app_context()
    
    # Parse provider and model from standardized format
    if ":" in model:
        provider, model_name = model.split(":", 1)
    else:
        # Backwards compatibility for old format
        if model.startswith("veo-"):
            provider, model_name = "gemini", model
        else:
            provider, model_name = "gemini", model  # Default to gemini
    
    try:
        if not GENAI_AVAILABLE or not app_context.gemini_configured:
            raise ValueError("Google Gen AI SDK not available or Gemini API key not configured; set GEMINI_API_KEY")
        
        # Validate provider
        if provider != "gemini":
            raise ValueError(f"Unsupported provider '{provider}' for video generation. Only 'gemini' is supported.")
        
        if ctx: await ctx.info(f"Generating video from image with {provider} model {model_name}...")
        
        # Validate prompt length
        if len(prompt) > 32000:
            raise ValueError("Prompt must be 32000 characters or less")
        
        # Handle image input - convert to path if needed
        image_path = await get_file_path(image)
        
        # Configure the generation request
        generation_config = {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
            "person_generation": person_generation
        }
        
        if negative_prompt:
            generation_config["negative_prompt"] = negative_prompt
        if seed is not None:
            generation_config["seed"] = seed
            
        if ctx: await ctx.info("Starting video generation from image...")
        
        # Note: This is a placeholder for the actual Veo3 image-to-video API call
        # The actual implementation would use the Google Gen AI SDK
        # For now, we'll raise an informative error
        raise NotImplementedError(
            "Image-to-video generation is not yet implemented in the current Veo3 API integration. "
            "This feature requires additional API endpoints that are being developed."
        )
        
    except Exception as e:
        if ctx: await ctx.error(f"Image-to-video generation failed: {str(e)}")
        raise ValueError(f"Failed to generate video from image: {str(e)}")

@mcp.tool()
async def get_video_operation_status(
    operation_id: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Check the status of a video generation operation.
    
    Args:
        operation_id: The operation ID returned from video generation
        
    Returns:
        Operation status information including completion state, progress, and video URLs
    """
    try:
        if ctx: await ctx.info(f"Checking status for operation: {operation_id}")
        
        # Check stored results from background processing
        logger.info(f"🔍 Checking operation {operation_id} in local storage")
        logger.info(f"🗃️ Currently stored operations: {list(_operation_results.keys())}")
        
        if operation_id not in _operation_results:
            logger.warning(f"⚠️ Operation {operation_id} not found in local storage")
            
            # Try to get status directly from API as fallback
            if not GENAI_AVAILABLE:
                return {
                    "status": "unknown",
                    "error": "Operation not found in local storage and Google Gen AI SDK not available",
                    "debug_info": {
                        "operation_id": operation_id,
                        "stored_operations": list(_operation_results.keys()),
                        "genai_available": GENAI_AVAILABLE
                    }
                }
            
            try:
                app_context = get_app_context()
                if not app_context.gemini_configured:
                    return {
                        "status": "unknown",
                        "error": "Gemini API not configured",
                        "debug_info": {
                            "operation_id": operation_id,
                            "gemini_configured": app_context.gemini_configured
                        }
                    }
                
                client = genai_sdk.Client(api_key=app_context.gemini_api_key)
                
                # Create operation object from ID string
                logger.info(f"🔍 Attempting to fetch operation {operation_id} directly from API")
                operation = genai_types.GenerateVideosOperation(name=operation_id)
                operation = client.operations.get(operation)
                
                logger.info(f"✅ Found operation {operation_id} via API: done={operation.done}")
                
                return {
                    "status": "completed" if operation.done else "in_progress",
                    "operation_id": operation_id,
                    "message": "Operation found via API, but auto-upload may not be active for this operation. The background monitoring may have failed.",
                    "debug_info": {
                        "found_via_api": True,
                        "operation_done": operation.done,
                        "background_monitoring_active": False
                    }
                }
            except Exception as e:
                logger.error(f"❌ Failed to fetch operation {operation_id} from API: {e}")
                return {
                    "status": "not_found",
                    "error": f"Operation {operation_id} not found in local storage or API: {str(e)}",
                    "debug_info": {
                        "operation_id": operation_id,
                        "api_error": str(e),
                        "stored_operations": list(_operation_results.keys())
                    }
                }
        
        # Get stored results
        result = _operation_results[operation_id]
        
        if result["status"] == "in_progress":
            import time
            elapsed = time.time() - result["started_at"]
            return {
                "status": "in_progress",
                "operation_id": operation_id,
                "prompt": result["prompt"],
                "model": result["model"],
                "expected_videos": result["expected_videos"],
                "elapsed_time": f"{elapsed:.0f} seconds",
                "message": "Video generation in progress. Videos will be automatically uploaded when ready."
            }
        
        elif result["status"] == "completed":
            return {
                "status": "completed",
                "operation_id": operation_id,
                "prompt": result["prompt"],
                "model": result["model"],
                "total_videos": result["total_videos"],
                "videos": result["videos"],
                "errors": result.get("errors", []),
                "processing_time": f"{result['completed_at'] - result['started_at']:.0f} seconds",
                "message": f"✅ Generation complete! {result['total_videos']} video(s) uploaded to Google Drive."
            }
        
        elif result["status"] == "failed":
            return {
                "status": "failed",
                "operation_id": operation_id,
                "error": result["error"],
                "message": "❌ Video generation failed. Check the error details above."
            }
        
        else:
            return {
                "status": "unknown",
                "operation_id": operation_id,
                "result": result
            }
            
    except Exception as e:
        if ctx: await ctx.error(f"Failed to check operation status: {str(e)}")
        raise ValueError(f"Failed to check operation status: {str(e)}")

@mcp.tool()
async def cancel_video_generation(
    operation_id: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Cancel an in-progress video generation operation.
    
    Args:
        operation_id: The operation ID to cancel
        
    Returns:
        Cancellation result
    """
    try:
        if not GENAI_AVAILABLE:
            raise ValueError("Google Gen AI SDK not available")
            
        if ctx: await ctx.info(f"Canceling operation: {operation_id}")
        
        # Note: This is a placeholder for the actual cancellation API call
        raise NotImplementedError(
            "Operation cancellation is not yet implemented. "
            "This feature requires additional API integration with Google's operation management system."
        )
        
    except Exception as e:
        if ctx: await ctx.error(f"Failed to cancel operation: {str(e)}")
        raise ValueError(f"Failed to cancel operation: {str(e)}")

@mcp.tool()
async def optimize_video_prompt(
    prompt: str,
    style_hints: Optional[List[str]] = None,
    camera_angles: Optional[List[str]] = None,
    ctx: Context = None
) -> str:
    """Optimize and enhance a video generation prompt using Veo3's prompt rewriting capabilities.
    
    Args:
        prompt: Original prompt to optimize
        style_hints: Optional style guidance (e.g., ["cinematic", "documentary", "animation"])
        camera_angles: Optional camera positioning (e.g., ["close-up", "wide shot", "tracking"])
        
    Returns:
        Optimized prompt with enhanced descriptions
    """
    try:
        if ctx: await ctx.info("Optimizing video prompt...")
        
        # Build enhanced prompt with structure recommendations from Veo3 docs
        optimized_parts = []
        
        # Start with original prompt
        optimized_parts.append(prompt)
        
        # Add style guidance if provided
        if style_hints:
            style_text = f"Style: {', '.join(style_hints)}"
            optimized_parts.append(style_text)
            
        # Add camera positioning if provided
        if camera_angles:
            camera_text = f"Camera: {', '.join(camera_angles)}"
            optimized_parts.append(camera_text)
            
        # Add recommended structure elements
        optimized_parts.append("High quality, detailed, professional lighting")
        
        optimized_prompt = ". ".join(optimized_parts)
        
        if ctx: await ctx.info(f"Prompt optimized from {len(prompt)} to {len(optimized_prompt)} characters")
        
        return optimized_prompt
        
    except Exception as e:
        if ctx: await ctx.error(f"Failed to optimize prompt: {str(e)}")
        raise ValueError(f"Failed to optimize prompt: {str(e)}")

@mcp.tool()
async def batch_video_generation(
    base_prompt: str,
    variations: List[str],
    ctx: Context = None,
    model: Literal["veo-2.0-generate-001", "veo-3.0-generate-001", "veo-3.0-fast-generate-001"] = "veo-3.0-generate-001",
    folder_id: Optional[str] = None,
    aspect_ratio: Literal["16:9", "9:16"] = "16:9",
    resolution: Literal["720p", "1080p"] = "720p"
) -> List[str]:
    """Generate multiple videos with prompt variations for testing different concepts.
    
    Args:
        base_prompt: Base prompt that all variations will build upon
        variations: List of prompt variations or additions
        model: Video generation model (fast model recommended for batch operations)
        folder_id: Google Drive folder ID for all generated videos
        aspect_ratio: Video aspect ratio for all videos
        resolution: Video resolution for all videos
        
    Returns:
        List of Google Drive web view URLs for all generated videos
    """
    try:
        if not variations:
            raise ValueError("At least one variation must be provided")
            
        if len(variations) > 5:
            raise ValueError("Maximum 5 variations allowed to respect API limits")
            
        if ctx: await ctx.info(f"Starting batch generation of {len(variations)} video variations...")
        
        results = []
        total_variations = len(variations)
        
        for i, variation in enumerate(variations):
            if ctx:
                await ctx.report_progress(i, total_variations, f"Generating video {i+1}/{total_variations}")
            
            # Combine base prompt with variation
            combined_prompt = f"{base_prompt}. {variation}"
            
            # Generate video using existing create_video function
            try:
                video_url = await create_video(
                    prompt=combined_prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    folder_id=folder_id,
                    ctx=ctx
                )
                results.append(video_url)
                
                if ctx: await ctx.info(f"Successfully generated variation {i+1}: {variation[:50]}...")
                
            except Exception as e:
                error_msg = f"Failed to generate variation {i+1}: {str(e)}"
                if ctx: await ctx.error(error_msg)
                results.append(f"ERROR: {error_msg}")
        
        if ctx: await ctx.info(f"Batch generation completed. {len([r for r in results if not r.startswith('ERROR')])} successful, {len([r for r in results if r.startswith('ERROR')])} failed")
        
        return results
        
    except Exception as e:
        if ctx: await ctx.error(f"Batch generation failed: {str(e)}")
        raise ValueError(f"Failed to perform batch generation: {str(e)}")

@mcp.tool()
async def get_video_metadata(
    video_url_or_path: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Extract metadata from a video file or URL.
    
    Args:
        video_url_or_path: Video file path, URL, or Google Drive URL
        
    Returns:
        Video metadata including duration, resolution, format, file size, etc.
    """
    try:
        if ctx: await ctx.info(f"Extracting metadata from: {video_url_or_path}")
        
        # Handle different input types
        if video_url_or_path.startswith(('http://', 'https://', 'drive://')):
            # For URLs, we would need to download or stream to analyze
            # This is a placeholder for the actual implementation
            return {
                "source": video_url_or_path,
                "type": "url",
                "status": "analysis_not_implemented",
                "message": "Video metadata extraction for URLs is not yet implemented. Please provide a local file path."
            }
        else:
            # Local file path
            video_path = await get_file_path(video_url_or_path)
            
            # Basic file info
            from pathlib import Path
            path_obj = Path(video_path)
            
            if not path_obj.exists():
                raise ValueError(f"Video file not found: {video_path}")
                
            stat = path_obj.stat()
            
            metadata = {
                "file_path": str(video_path),
                "filename": path_obj.name,
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "extension": path_obj.suffix.lower(),
                "status": "basic_info_only",
                "message": "Advanced video analysis (duration, resolution, codec) requires additional dependencies (opencv-python, moviepy)"
            }
            
            if ctx: await ctx.info(f"Basic metadata extracted for {path_obj.name}")
            return metadata
            
    except Exception as e:
        if ctx: await ctx.error(f"Failed to extract video metadata: {str(e)}")
        raise ValueError(f"Failed to extract video metadata: {str(e)}")


# =============================================================================
# GOOGLE DRIVE TOOLS FOR VIDEOS
# =============================================================================

@mcp.tool()
async def search_videos(
    query: str = "",
    folder_id: Optional[str] = None,
    max_results: int = 50,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search for video files in Google Drive.
    
    Args:
        query: Search query (e.g., "vacation videos", "name contains 'generated'")
        folder_id: Optional folder to search within
        max_results: Maximum number of results (1-1000)
        
    Returns:
        List of video files with download URLs and metadata
    """
    try:
        app_context = get_app_context()
        drive_service = app_context.drive_service
        
        if not drive_service:
            return {"error": "Google Drive not configured. Please set GOOGLE_OAUTH_TOKEN or GOOGLE_SERVICE_ACCOUNT_JSON", "success": False}
        
        if ctx:
            await ctx.info(f"Searching for videos: {query}")
        
        # Build search query for videos  
        q_parts = ["trashed=false", "mimeType contains 'video/'"]
        
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


# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Tool MCP Server")
    parser.add_argument("--transport", default="streamable-http", choices=["http", "stdio", "streamable-http"], 
                       help="Transport method (http or stdio)")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                       help="Host to bind to (http mode only)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8080)),
                       help="Port to bind to (http mode only)")
    
    args = parser.parse_args()
    
    # Set global transport mode for automatic output mode detection
    _transport_mode = args.transport
    
    if args.transport == "stdio":
        # In stdio mode, logging goes to stderr (configured above)
        mcp.run(transport="stdio")
    else:
        # HTTP mode - show configuration
        print("=" * 50)
        print("🚀 STARTING VIDEO-TOOL-MCP SERVER")
        print("=" * 50)
        print(f"🔧 Server configuration:")
        print(f"  Transport: {args.transport}")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        
        logger.info("=" * 50)
        logger.info("STARTING VIDEO-TOOL-MCP SERVER")
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