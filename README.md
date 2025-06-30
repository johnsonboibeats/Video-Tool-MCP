# Images Tool MCP

A professional Model Context Protocol (MCP) server for OpenAI's image generation and analysis capabilities, built with FastMCP and featuring both Images API and Vision API integration.

## 🚀 Features

### 🎨 Image Generation (OpenAI Images API)
- **create-image**: Generate images from text prompts using gpt-image-1
- **edit-image**: Edit existing images with mask support and intelligent prompts
- **generate-variations**: Create variations of existing images

### 🔍 Image Analysis (OpenAI Vision API)  
- **analyze-image**: Detailed image analysis and description
- **extract-text**: OCR functionality for text extraction
- **compare-images**: Compare two images and identify differences

### 🧠 Hybrid Intelligence Tools
- **smart-edit**: AI-guided editing (analyze first, then edit intelligently)
- **describe-and-recreate**: Recreate images with optional style modifications
- **prompt-from-image**: Generate optimized prompts from existing images

### 🛠️ Utility Tools
- **image-metadata**: Extract EXIF data and image properties
- **transform-image**: Basic transformations (resize, crop, rotate, flip, convert)
- **batch-process**: Process multiple images with progress tracking

## 📋 Requirements

- Python 3.11+
- OpenAI API key
- Optional: Azure OpenAI credentials

## 🔧 Installation

1. **Clone or create the project directory:**
   ```bash
   mkdir openai-image-mcp-python
   cd openai-image-mcp-python
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv init .
   uv add "mcp[cli]>=1.0.0" "openai>=1.54.0" "pillow>=10.0.0" "httpx>=0.27.0" "python-dotenv>=1.0.0" "pydantic>=2.0.0" "aiofiles>=24.0.0"
   
   # Or using pip
   pip install "mcp[cli]>=1.0.0" "openai>=1.54.0" "pillow>=10.0.0" "httpx>=0.27.0" "python-dotenv>=1.0.0" "pydantic>=2.0.0" "aiofiles>=24.0.0"
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## ⚙️ Configuration

Create a `.env` file with your API credentials:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Azure OpenAI (use instead of OpenAI)
# AZURE_OPENAI_API_KEY=your_azure_openai_key
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-02-01

# Optional: Custom temp directory
MCP_TEMP_DIR=/tmp/openai_image_mcp
```

## 🏃‍♂️ Usage

### Development Mode
```bash
# Run in development mode with hot reload
mcp dev server.py

# With additional dependencies
mcp dev server.py --with openai --with pillow
```

### Production Installation
```bash
# Install for Claude Desktop
mcp install server.py --name "Images Tool MCP"

# With environment variables
mcp install server.py --name "Images Tool MCP" -f .env
```

### Direct Execution
```bash
# Run directly
python server.py

# Or with uv
uv run server.py
```

## 🛠️ Tool Reference

### Image Generation Tools

#### create-image
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

#### edit-image
Edit existing images with optional mask support.

**Parameters:**
- `image` (str): Image path or base64 string
- `prompt` (str): Edit description
- `mask` (str, optional): Mask image path or base64
- `model` (str): "gpt-image-1"
- `n` (int): Number of variations (1-10)
- `quality` (str): Quality level
- `output_mode` (str): "base64" or "file"

#### generate-variations
Create variations of existing images.

**Parameters:**
- `image` (str): Source image path or base64
- `n` (int): Number of variations (1-10)
- `size` (str): Output dimensions
- `output_mode` (str): "base64" or "file"

### Image Analysis Tools

#### analyze-image
Comprehensive image analysis using Vision API.

**Parameters:**
- `image` (str): Image path or base64 string
- `prompt` (str): Analysis prompt (default: detailed description)
- `model` (str): Vision model ("gpt-4o", "gpt-4o-mini")
- `max_tokens` (int): Response length limit
- `detail` (str): "low", "high", "auto"

#### extract-text
OCR functionality for text extraction.

**Parameters:**
- `image` (str): Image containing text
- `model` (str): Vision model
- `language` (str): Expected language ("auto" for detection)
- `preserve_formatting` (bool): Maintain original formatting

#### compare-images
Compare two images and analyze differences.

**Parameters:**
- `image1` (str): First image
- `image2` (str): Second image
- `comparison_prompt` (str): What to compare
- `model` (str): Vision model

### Hybrid Intelligence Tools

#### smart-edit
AI-guided editing with analysis-driven prompts.

**Parameters:**
- `image` (str): Image to edit
- `edit_request` (str): Desired edits
- `analyze_first` (bool): Whether to analyze before editing
- `model` (str): Vision model for analysis
- `edit_model` (str): Image model for editing

#### describe-and-recreate
Recreate images with optional style modifications.

**Parameters:**
- `image` (str): Source image
- `style_modification` (str): Optional style changes
- `model` (str): Vision model
- `generation_model` (str): Image generation model

#### prompt-from-image
Generate optimized prompts from images.

**Parameters:**
- `image` (str): Source image
- `style_focus` (str): "general", "artistic", "photographic", "technical"
- `model` (str): Vision model

### Utility Tools

#### image-metadata
Extract comprehensive image metadata.

**Parameters:**
- `image` (str): Image file path

#### transform-image
Basic image transformations using PIL.

**Parameters:**
- `image` (str): Source image path
- `operation` (str): "resize", "crop", "rotate", "flip", "convert"
- `parameters` (dict): Operation-specific parameters
- `output_path` (str): Output file path

#### batch-process
Process multiple images with progress tracking.

**Parameters:**
- `images` (list): List of image paths
- `operation` (str): Operation to apply
- `operation_params` (dict): Parameters for operation
- `output_directory` (str): Output directory

## 🔒 Security Features

- ✅ Absolute path validation for file operations
- ✅ Image format validation
- ✅ Base64 data validation
- ✅ Automatic temp file cleanup
- ✅ Error handling and logging
- ✅ Resource management with proper cleanup

## 🚀 Advanced Features

- **Progress Tracking**: Real-time progress for batch operations
- **Smart Output Handling**: Automatic file/base64 switching based on size
- **Lifespan Management**: Proper resource initialization and cleanup
- **Azure OpenAI Support**: Seamless switching between OpenAI and Azure
- **Context Awareness**: Tools have access to shared application context
- **Error Recovery**: Robust error handling with detailed messages

## 🧪 Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=server

# Type checking
uv run pyright

# Linting
uv run ruff check .
uv run ruff format .
```

## 📝 Examples

### Basic Image Generation
```python
# Generate a simple image
result = await create_image(
    prompt="A serene mountain landscape at sunset",
    quality="high",
    output_format="png"
)
```

### Smart Image Editing
```python
# Intelligently edit an image
result = await smart_edit(
    image="/path/to/image.jpg",
    edit_request="Add warm golden hour lighting",
    analyze_first=True
)
```

### Batch Processing
```python
# Process multiple images
results = await batch_process(
    images=["/path/to/img1.jpg", "/path/to/img2.jpg"],
    operation="analyze_image",
    operation_params={"prompt": "Describe the mood and atmosphere"},
    output_directory="/path/to/output"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run linting and tests
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation above
2. Review error messages in logs
3. Verify API key configuration
4. Check file path permissions
5. Open an issue with detailed information

---

**Built with ❤️ using FastMCP and OpenAI APIs**