@mcp.tool()
async def create_video(
    prompt: str,
    ctx: Context = None,
    model: Literal["veo-3.0-generate-preview", "veo-3.0-fast-generate-preview", "auto"] = "auto",
    aspect_ratio: Literal["16:9", "9:16", "auto"] = "auto",
    negative_prompt: Optional[str] = None,
    n: int = 1,
    folder_id: Optional[str] = None
) -> Union[str, list[str]]:
    """Generate videos from text prompts using Google Veo3.
    
    Automatically uploads all generated videos to Google Drive and returns web view URLs.
    
    Args:
        prompt: Text description of the video to generate (max 32000 chars)
        model: Video generation model (veo-3.0-generate-preview or veo-3.0-fast-generate-preview)
        aspect_ratio: Video aspect ratio (16:9 or 9:16)
        negative_prompt: Elements to exclude from generation
        n: Number of videos to generate (1-2, limited by Veo3)
        folder_id: Google Drive folder ID (defaults to Downloads folder)
        
    Returns:
        Google Drive web view URL(s) for generated videos
    """
    # Get application context
    app_context = get_app_context()

    # Determine model selection (scoped to create_video only)
    env_model = os.getenv("CREATE_VIDEO_MODEL")
    selected_model = model if model and model != "auto" else (env_model or "veo-3.0-generate-preview")
    
    # Always log model selection for debugging
    logger.info(f"🎬 VIDEO MODEL DEBUG: input='{model}', env='{env_model}', selected='{selected_model}'")
    if ctx: 
        await ctx.info(f"Model selection: input='{model}', env='{env_model}', selected='{selected_model}'")

    # Validate inputs according to Veo3 limits
    if len(prompt) > 32000:
        raise ValueError("Prompt must be 32000 characters or less")
    
    if n < 1 or n > 2:  # Veo3 limit: maximum 2 videos per request
        raise ValueError("Number of videos must be between 1 and 2 (Veo3 API limit)")
    
    # Progress tracking for batch generation
    if n > 1 and ctx:
        await ctx.report_progress(0, n, f"Starting generation of {n} videos...")
    
    try:
        if not GENAI_AVAILABLE or not app_context.vertex_project:
            raise ValueError("Google Gen AI SDK not available or Vertex project not configured; set GOOGLE_CLOUD_PROJECT/VERTEX_PROJECT and VERTEX_LOCATION")
        
        # Use Google Gen AI SDK (Vertex mode) for Veo3
        client = genai_sdk.Client(
            vertexai=True,
            project=app_context.vertex_project,
            location=app_context.vertex_location or "us-central1",
        )

        # Prepare video generation config
        cfg_kwargs = {}
        if aspect_ratio and aspect_ratio != "auto":
            cfg_kwargs["aspect_ratio"] = aspect_ratio
        if negative_prompt:
            cfg_kwargs["negative_prompt"] = negative_prompt

        config = genai_types.GenerateVideosConfig(**cfg_kwargs)

        if ctx:
            await ctx.info(f"Generating {n} video(s) with Veo3 model: {selected_model}")
        
        # Generate videos
        operation = client.models.generate_videos(
            model=selected_model,
            prompt=prompt,
            config=config,
        )
        
        # Wait for completion (Veo3 takes 11 seconds to 6 minutes)
        if ctx:
            await ctx.info("Video generation in progress... This may take 1-6 minutes.")
        
        # TODO: Implement proper async waiting for operation completion
        # This is a placeholder - actual implementation needs operation polling
        
        # Process results
        videos = []
        
        for i in range(n):
            if n > 1 and ctx:
                await ctx.report_progress(i + 1, n, f"Processing video {i + 1}/{n}")
            
            # Generate filename for the video
            if n > 1:
                filename = f"generated_video_{uuid.uuid4().hex[:8]}_{i+1}.mp4"
            else:
                filename = f"generated_video_{uuid.uuid4().hex[:8]}.mp4"
            
            # TODO: Extract video data from operation result
            # For now, create placeholder
            placeholder_message = f"🎬 Video generation queued: {filename}"
            
            # In actual implementation:
            # 1. Extract video data from operation result
            # 2. Upload to Google Drive using _upload_to_drive
            # web_view_link = await _upload_to_drive(
            #     file_data=video_data,
            #     filename=filename,
            #     description=f"Generated video with {selected_model}",
            #     folder_id=folder_id or "1y8eWyr68gPTiFTS2GuNODZp9zx4kg4FC",
            #     ctx=ctx
            # )
            
            videos.append(placeholder_message)
        
        # Log response preparation
        if ctx: 
            await ctx.info(f"Video generation template completed for {len(videos)} video(s)")
        
        # Return results
        result = videos if n > 1 else videos[0]
        return result
            
    except Exception as e:
        if ctx: await ctx.error(f"Video generation failed: {str(e)}")
        raise ValueError(f"Failed to generate video: {str(e)}")


@mcp.tool()
async def create_video_from_image(
    image: str,
    prompt: str,
    ctx: Context = None,
    model: Literal["veo-3.0-generate-preview", "veo-3.0-fast-generate-preview", "auto"] = "auto",
    aspect_ratio: Literal["16:9", "9:16", "auto"] = "auto",
    motion_style: Literal["slow", "medium", "fast", "auto"] = "auto",
    folder_id: Optional[str] = None
) -> str:
    """Generate video from an image using Google Veo3 image-to-video capability.
    
    Automatically uploads generated video to Google Drive and returns web view URL.
    
    Args:
        image: Source image (file path or base64)
        prompt: Text description of desired video motion/content
        model: Video generation model
        aspect_ratio: Video aspect ratio
        motion_style: Speed/intensity of motion
        folder_id: Google Drive folder ID (defaults to Downloads folder)
        
    Returns:
        Google Drive web view URL for generated video
    """
    # Get application context
    app_context = get_app_context()
    
    # Validate inputs
    if len(prompt) > 32000:
        raise ValueError("Prompt must be 32000 characters or less")
    
    try:
        if not GENAI_AVAILABLE or not app_context.vertex_project:
            raise ValueError("Google Gen AI SDK not available or Vertex project not configured")
        
        # Load and process input image
        image_data, mime_type = await load_image_as_base64(image)
        
        # Use Google Gen AI SDK for Veo3 image-to-video
        client = genai_sdk.Client(
            vertexai=True,
            project=app_context.vertex_project,
            location=app_context.vertex_location or "us-central1",
        )

        # Determine model
        env_model = os.getenv("CREATE_VIDEO_MODEL")
        selected_model = model if model and model != "auto" else (env_model or "veo-3.0-generate-preview")
        
        if ctx:
            await ctx.info(f"Generating video from image with model: {selected_model}")
        
        # Prepare config for image-to-video
        cfg_kwargs = {}
        if aspect_ratio and aspect_ratio != "auto":
            cfg_kwargs["aspect_ratio"] = aspect_ratio
            
        config = genai_types.GenerateVideosConfig(**cfg_kwargs)
        
        # TODO: Implement image-to-video API call
        # This would include the image as input along with the prompt
        
        # Generate filename
        filename = f"image_to_video_{uuid.uuid4().hex[:8]}.mp4"
        
        # Placeholder response
        if ctx:
            await ctx.info("⚠️ Image-to-video template implementation. Actual Veo3 integration required.")
        
        return f"🎬 Image-to-video generation queued: {filename}"
            
    except Exception as e:
        if ctx: await ctx.error(f"Image-to-video generation failed: {str(e)}")
        raise ValueError(f"Failed to generate video from image: {str(e)}")


@mcp.tool()
async def analyze_video(
    video: str,
    prompt: str = "Describe this video in detail, including visual content, motion, audio elements, and overall narrative.",
    ctx: Context = None,
    model: Literal["gpt-4o", "vertex:gemini-2.5-pro", "auto"] = "auto",
    frame_sampling: int = 10
) -> str:
    """Analyze video content with detailed descriptions using GPT-4o or Gemini.
    
    Args:
        video: Video file path or base64 data
        prompt: Analysis prompt
        model: Analysis model to use
        frame_sampling: Number of frames to sample for analysis
        
    Returns:
        Detailed video analysis
    """
    app_context = get_app_context()
    
    # Determine model selection
    env_model = os.getenv("ANALYZE_VIDEO_MODEL")
    selected_model = model if model and model != "auto" else (env_model or "gpt-4o")
    
    logger.info(f"🎬 VIDEO ANALYSIS MODEL: {selected_model}")
    if ctx:
        await ctx.info(f"Analyzing video with model: {selected_model}")
    
    try:
        # TODO: Implement video frame extraction and analysis
        # 1. Extract frames from video at regular intervals
        # 2. Convert frames to base64
        # 3. Send to selected model for analysis
        
        if ctx:
            await ctx.info("⚠️ Video analysis template implementation. Frame extraction required.")
        
        return f"🎬 Video analysis placeholder for: {video[:100]}..."
            
    except Exception as e:
        if ctx: await ctx.error(f"Video analysis failed: {str(e)}")
        raise ValueError(f"Failed to analyze video: {str(e)}")


@mcp.tool()
async def extract_video_frames(
    video: str,
    timestamps: Optional[List[float]] = None,
    frame_count: int = 10,
    output_format: Literal["png", "jpeg"] = "png",
    ctx: Context = None,
    folder_id: Optional[str] = None
) -> List[str]:
    """Extract specific frames from videos as images.
    
    Automatically uploads extracted frames to Google Drive and returns web view URLs.
    
    Args:
        video: Video file path or base64 data
        timestamps: Specific timestamps to extract (seconds)
        frame_count: Number of frames to extract if timestamps not provided
        output_format: Image format for extracted frames
        folder_id: Google Drive folder ID (defaults to Downloads folder)
        
    Returns:
        List of Google Drive web view URLs for extracted frames
    """
    try:
        if ctx:
            await ctx.info(f"Extracting {frame_count} frames from video")
        
        # TODO: Implement video frame extraction using OpenCV or moviepy
        # 1. Load video file
        # 2. Extract frames at specified timestamps or intervals
        # 3. Save frames as images
        # 4. Upload to Google Drive
        
        frames = []
        for i in range(frame_count):
            filename = f"frame_{uuid.uuid4().hex[:8]}_{i+1}.{output_format}"
            frames.append(f"🎬 Frame extraction placeholder: {filename}")
        
        if ctx:
            await ctx.info("⚠️ Frame extraction template implementation. Video processing required.")
        
        return frames
            
    except Exception as e:
        if ctx: await ctx.error(f"Frame extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract frames: {str(e)}")


@mcp.tool()
async def video_metadata(
    video: str,
    include_audio_info: bool = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """Extract technical information from videos.
    
    Args:
        video: Video file path or base64 data
        include_audio_info: Include audio track information
        
    Returns:
        Dictionary containing video metadata
    """
    try:
        if ctx:
            await ctx.info("Extracting video metadata")
        
        # TODO: Implement video metadata extraction
        # 1. Load video file
        # 2. Extract technical information (resolution, fps, duration, codecs, etc.)
        # 3. Extract audio information if requested
        
        placeholder_metadata = {
            "filename": video.split("/")[-1] if "/" in video else "unknown",
            "format": "mp4",
            "duration": "8.0 seconds",
            "resolution": "720p",
            "fps": 24,
            "aspect_ratio": "16:9",
            "file_size": "Unknown",
            "video_codec": "h264",
            "audio_codec": "aac" if include_audio_info else None,
            "note": "Placeholder metadata - actual implementation required"
        }
        
        if ctx:
            await ctx.info("⚠️ Video metadata template implementation. Video processing required.")
        
        return placeholder_metadata
            
    except Exception as e:
        if ctx: await ctx.error(f"Video metadata extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract video metadata: {str(e)}")


@mcp.tool()
async def batch_video_process(
    videos: List[str],
    operation: Literal["analyze_video", "video_metadata", "extract_frames"],
    batch_settings: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> List[Any]:
    """Process multiple videos with various operations.
    
    Args:
        videos: List of video file paths or base64 data
        operation: Operation to perform on each video
        batch_settings: Additional settings for the operation
        
    Returns:
        List of results from processing each video
    """
    if not videos:
        raise ValueError("No videos provided for batch processing")
    
    if len(videos) > 10:
        raise ValueError("Maximum 10 videos allowed for batch processing")
    
    results = []
    
    if ctx:
        await ctx.info(f"Starting batch processing of {len(videos)} videos with operation: {operation}")
    
    try:
        for i, video in enumerate(videos):
            if ctx:
                await ctx.report_progress(i + 1, len(videos), f"Processing video {i + 1}/{len(videos)}")
            
            if operation == "analyze_video":
                result = await analyze_video(video, ctx=ctx)
            elif operation == "video_metadata":
                result = await video_metadata(video, ctx=ctx)
            elif operation == "extract_frames":
                result = await extract_video_frames(video, ctx=ctx)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            results.append(result)
        
        if ctx:
            await ctx.info(f"Batch processing completed for {len(results)} videos")
        
        return results
        
    except Exception as e:
        if ctx: await ctx.error(f"Batch video processing failed: {str(e)}")
        raise ValueError(f"Failed to process videos in batch: {str(e)}")