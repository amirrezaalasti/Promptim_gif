"""Manim scene rendering and video processing utilities."""

import os
import sys
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional, Tuple


def extract_scene_name(code: str) -> str:
    """Extract scene class name from Manim code."""
    match = re.search(r"class\s+(\w+)\s*\([^)]*Scene[^)]*\)", code)
    if match:
        return match.group(1)
    return "GeneratedScene"


def render_manim_scene(
    code: str,
    resolution: str = "720p",
    fps: int = 30,
    quality: str = "medium",
    output_dir: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Render Manim code to video file.

    Args:
        code: Manim Python code to render
        resolution: Output resolution (480p, 720p, 1080p)
        fps: Frames per second
        quality: Rendering quality (low, medium, high)
        output_dir: Directory for output files

    Returns:
        Tuple of (video_path or None, log_message)
    """
    # Create temp directory for scene file
    temp_dir = Path(tempfile.mkdtemp())
    scene_file = temp_dir / "generated_scene.py"

    # Replace relative "assets/" paths with absolute paths
    # This is crucial because the script runs in a temp dir
    project_root = Path(__file__).parent
    assets_dir_abs = (project_root / "assets").resolve()
    
    # Simple string replacement for "assets/" and 'assets/'
    # Using forward slashes for cross-platform compatibility in Python strings
    assets_path_str = str(assets_dir_abs).replace("\\", "/")
    if not assets_path_str.endswith("/"):
        assets_path_str += "/"
        
    code = code.replace('"assets/', f'"{assets_path_str}')
    code = code.replace("'assets/", f"'{assets_path_str}")

    # Write code to file
    with open(scene_file, "w") as f:
        f.write(code)

    # Determine resolution
    res_map = {"480p": "854,480", "720p": "1280,720", "1080p": "1920,1080"}
    resolution_str = res_map.get(resolution, "1280,720")

    # Output directory
    if output_dir is None:
        output_dir = temp_dir / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get scene name
    scene_name = extract_scene_name(code)

    # Build command - use sys.executable to ensure access to all packages
    # Use absolute path for scene_file to avoid path resolution issues
    scene_file_abs = str(scene_file.resolve())
    
    # Base manim command
    manim_cmd = [
        sys.executable,
        "-m",
        "manimlib",
        scene_file_abs,
        scene_name,
        "-w",  # Write to file
        "--video_dir",
        str(output_dir),
        "-r",
        resolution_str.replace(",", "x"),  # Format: 1280x720
        "--fps",
        str(fps),
    ]
    
    # Check if we're in a headless environment (no DISPLAY set)
    # If so, wrap with xvfb-run to provide a virtual framebuffer
    is_headless = not os.environ.get("DISPLAY")
    
    if is_headless:
        # Use xvfb-run to provide a virtual display for OpenGL/pyglet
        cmd = [
            "xvfb-run",
            "-a",  # Auto-select display number
            "--server-args=-screen 0 1920x1080x24",  # Virtual screen size
        ] + manim_cmd
    else:
        cmd = manim_cmd

    # Add quality flag
    if quality == "low":
        cmd.append("-l")
    elif quality == "high":
        cmd.append("--high_quality")

    try:
        # Environment setup - use the current Python environment
        env = os.environ.copy()

        # Ensure PATH includes the Python environment (just in case)
        python_bin = str(Path(sys.executable).parent)
        if python_bin not in env.get("PATH", ""):
            env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")

        # Verify environment setup by testing colour import
        # This helps catch environment issues early
        test_result = subprocess.run(
            [sys.executable, "-c", "import colour"],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if test_result.returncode != 0:
            # If colour can't be imported, try to add more paths
            # Sometimes packages are in different locations
            import importlib.util

            try:
                spec = importlib.util.find_spec("colour")
                if spec and spec.origin:
                    colour_dir = str(Path(spec.origin).parent)
                    current_pp = env.get("PYTHONPATH", "")
                    if colour_dir not in current_pp:
                         env["PYTHONPATH"] = current_pp + os.pathsep + colour_dir
            except Exception:
                pass  # If we can't find it, proceed anyway

        result = subprocess.run(
            cmd,
            cwd=str(temp_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        log_message = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

        if result.returncode != 0:
            return None, f"Manim rendering failed:\n{log_message}"

        # Find output video
        video_files = list(output_dir.glob("**/*.mp4"))
        if not video_files:
            # Also check in the temp directory
            video_files = list(temp_dir.glob("**/*.mp4"))

        if not video_files:
            return None, f"No video file generated.\n{log_message}"

        return str(video_files[0]), log_message

    except subprocess.TimeoutExpired:
        return None, "Rendering timed out after 5 minutes"
    except Exception as e:
        return None, f"Rendering error: {str(e)}"


def video_to_gif(
    video_path: str,
    fps: int = 15,
    max_width: int = 800,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Convert video to GIF.

    Args:
        video_path: Path to input video
        fps: Target FPS for GIF
        max_width: Maximum width (GIF will be resized if larger)
        output_path: Output path for GIF (auto-generated if None)

    Returns:
        Path to generated GIF or None on failure
    """
    if output_path is None:
        output_path = video_path.replace(".mp4", ".gif")

    # Try using imageio first
    try:
        import imageio.v3 as iio
        from PIL import Image

        frames = iio.imread(video_path, plugin="pyav")

        processed_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize(
                    (max_width, int(img.height * ratio)), Image.Resampling.LANCZOS
                )
            processed_frames.append(img)

        # Save as GIF
        if processed_frames:
            processed_frames[0].save(
                output_path,
                save_all=True,
                append_images=processed_frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True,
            )
            return output_path

    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to ffmpeg
    try:
        palette_path = video_path.replace(".mp4", "_palette.png")

        # Generate palette
        palette_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={fps},scale={max_width}:-1:flags=lanczos,palettegen",
            palette_path,
        ]
        subprocess.run(palette_cmd, check=True, capture_output=True)

        # Generate GIF using palette
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            palette_path,
            "-lavfi",
            f"fps={fps},scale={max_width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            output_path,
        ]
        subprocess.run(gif_cmd, check=True, capture_output=True)

        # Cleanup palette
        if os.path.exists(palette_path):
            os.remove(palette_path)

        return output_path

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Simple ffmpeg fallback without palette
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={fps},scale={max_width}:-1:flags=lanczos",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_video_info(video_path: str) -> dict:
    """Get information about a video file."""
    try:
        import imageio.v3 as iio

        props = iio.improps(video_path, plugin="pyav")
        meta = iio.immeta(video_path, plugin="pyav")

        return {
            "duration": meta.get("duration", 0),
            "fps": meta.get("fps", 30),
            "width": props.shape[2] if len(props.shape) > 2 else 0,
            "height": props.shape[1] if len(props.shape) > 1 else 0,
            "frames": props.shape[0] if len(props.shape) > 0 else 0,
        }
    except Exception:
        return {"duration": 0, "fps": 30, "width": 0, "height": 0, "frames": 0}
