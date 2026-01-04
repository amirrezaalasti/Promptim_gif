"""Streamlit app for generating Manim animations from prompts."""

import os
import sys
from pathlib import Path

# Add parent directory for manim imports
sys.path.insert(0, str(Path(__file__).parent.parent / "manim"))

import streamlit as st

from code_generator import plan_scene, generate_manim_code, fix_code
from manim_renderer import render_manim_scene, video_to_gif

# Page config
st.set_page_config(
    page_title="Promptim - Animation Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better aesthetics
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 210, 255, 0.3);
    }
    .example-btn {
        background: transparent;
        border: 1px solid #3a7bd5;
        color: #3a7bd5;
    }
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }
    .success-box {
        background-color: rgba(0, 255, 127, 0.1);
        border: 1px solid rgba(0, 255, 127, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: rgba(255, 69, 58, 0.1);
        border: 1px solid rgba(255, 69, 58, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<h1 class="main-header">Promptim</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Generate beautiful mathematical animations from simple prompts</p>',
    unsafe_allow_html=True,
)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # LLM Provider
    llm_provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "DeepSeek (Local)"],
        help="Choose the language model for code generation",
    )

    if llm_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key", type="password", help="Your OpenAI API key"
        )
        model_name = st.selectbox(
            "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0
        )
        model_path = None
    else:
        api_key = None
        model_name = None
        model_path = st.text_input(
            "Model Path",
            value="./deepseek-manim-finetuned",
            help="Path to fine-tuned DeepSeek model",
        )

    st.divider()

    # Render settings
    st.subheader("Render Settings")
    resolution = st.selectbox("Resolution", ["480p", "720p", "1080p"], index=1)
    fps = st.slider("FPS", 15, 60, 30)
    quality = st.selectbox("Quality", ["low", "medium", "high"], index=1)

    st.divider()

    # Auto-fix settings
    st.subheader("Auto-Fix")
    auto_fix = st.checkbox("Auto-fix errors with LLM", value=True)
    max_attempts = st.slider("Max fix attempts", 1, 10, 5) if auto_fix else 1

    st.divider()

    # Image Uploader
    st.subheader("🖼️ Assets")
    
    # Ensure assets directory exists
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = assets_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} image(s)")
        
    # List available assets
    available_images = [f.name for f in assets_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if available_images:
        st.markdown("**Available Images:**")
        for img in available_images:
            st.caption(f"📄 {img}")

# Example prompts data
examples = [
    (
        "Gradient Descent",
        "Show gradient descent on a parabola with a dot moving to the minimum",
    ),
    (
        "Pythagorean Theorem",
        "Visualize the Pythagorean theorem with squares on triangle sides",
    ),
    ("Sine Wave", "Create a sine wave being drawn with a rotating vector"),
    (
        "Matrix Transform",
        "Show how a 2D transformation matrix rotates and scales a square",
    ),
    (
        "Fourier Series",
        "Demonstrate adding sine waves to approximate a square wave",
    ),
]

# Initialize session state for selected example
if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Enter Your Prompt")

    # Get default value from session state
    default_value = st.session_state.selected_example

    prompt = st.text_area(
        "Describe the animation you want:",
        value=default_value,
        height=150,
        placeholder="Example: Create an animation showing gradient descent on a quadratic function with a ball rolling down to the minimum...",
    )

    # Example prompts
    st.markdown("**Quick Examples:**")

    example_cols = st.columns(3)
    for i, (name, desc) in enumerate(examples):
        with example_cols[i % 3]:
            if st.button(name, key=f"ex_{i}", use_container_width=True):
                st.session_state.selected_example = desc
                st.rerun()

    st.divider()

    generate_btn = st.button(
        "Generate Animation",
        type="primary",
        use_container_width=True,
        disabled=not prompt,
    )

with col2:
    st.subheader("Output")

    # State management initialization
    if "should_render" not in st.session_state:
        st.session_state.should_render = False

    if generate_btn:
        if not prompt:
            st.error("Please enter a prompt and verify your API settings")
        else:
            st.session_state.should_render = True
            st.session_state.force_new_generation = True

    # Main Rendering Flow
    if st.session_state.should_render:
        # Step 1: Plan the scene (Only if new generation needed)
        plan = None
        if st.session_state.get("force_new_generation", False):
            with st.status("Step 1: Planning the scene...", expanded=True) as status:
                try:
                    if llm_provider == "OpenAI":
                        if not api_key:
                            st.error("Please enter your OpenAI API key in the sidebar")
                            st.stop()
                        plan = plan_scene(
                            prompt, provider="openai", api_key=api_key, model=model_name, image_files=available_images
                        )
                    else:
                        plan = plan_scene(
                            prompt, provider="deepseek", model_path=model_path, image_files=available_images
                        )

                    st.session_state.current_plan = plan
                    st.write("Scene planned successfully")
                    status.update(label="Scene planned", state="complete")

                except Exception as e:
                    status.update(label="Planning failed", state="error")
                    st.error(f"Error: {str(e)}")
                    st.session_state.should_render = False
                    st.stop()

            # Show the plan
            with st.expander("View Scene Plan", expanded=True):
                st.markdown(plan)

            # Step 2: Generate code from plan
            code = None
            with st.status("Step 2: Generating Manim code...", expanded=True) as status:
                try:
                    if llm_provider == "OpenAI":
                        code = generate_manim_code(
                            plan, provider="openai", api_key=api_key, model=model_name, image_files=available_images
                        )
                    else:
                        code = generate_manim_code(
                            plan, provider="deepseek", model_path=model_path, image_files=available_images
                        )

                    st.session_state.generated_code = code
                    st.write("Code generated successfully")
                    status.update(label="Code generated", state="complete")

                except Exception as e:
                    status.update(label="Code generation failed", state="error")
                    st.error(f"Error: {str(e)}")
                    st.session_state.should_render = False
                    st.stop()
            
            # Reset force flag after successful generation
            st.session_state.force_new_generation = False

        # --- PHASE 2: RENDERING & FIXING ---
        if "generated_code" in st.session_state:
            code = st.session_state.generated_code
            
            # Show generated code
            with st.expander("View Generated Code", expanded=False):
                st.code(code, language="python")

            # Render animation with iterative fixing
            video_path = None
            current_code = code
            attempts = 0
            max_fix_attempts = max_attempts if auto_fix else 1
            last_error_log = None

        with st.status("Rendering animation...", expanded=True) as status:
            while attempts < max_fix_attempts:
                attempts += 1
                if max_fix_attempts > 1:
                    st.write(f"Attempt {attempts}/{max_fix_attempts}...")
                else:
                    st.write("Rendering...")

                video_path, log = render_manim_scene(
                    current_code, resolution=resolution, fps=fps, quality=quality
                )

                if video_path and os.path.exists(video_path):
                    status.update(label="Animation rendered", state="complete")
                    break

                # Rendering failed, try to fix the code
                last_error_log = log
                if auto_fix and attempts < max_fix_attempts:
                    st.write("Rendering failed. LLM is fixing the code...")
                    try:
                        # Ensure we have a meaningful error message
                        if not log or len(log.strip()) < 10:
                            log = (
                                "Rendering failed - no detailed error message available"
                            )

                        if llm_provider == "OpenAI":
                            current_code = fix_code(
                                current_code,
                                log,
                                provider="openai",
                                api_key=api_key,
                                model=model_name,
                            )
                        else:
                            current_code = fix_code(
                                current_code,
                                log,
                                provider="deepseek",
                                model_path=model_path,
                            )
                        st.write("Code fixed, retrying...")
                    except Exception as e:
                        error_msg = str(e)
                        if len(error_msg) < 5:
                            error_msg = f"Error during code fixing: {type(e).__name__}"
                        st.warning(f"Could not fix code: {error_msg}")
                        break
                else:
                    status.update(label="Rendering failed", state="error")
                    st.error(f"Rendering failed after {attempts} attempt(s)")
                    break
        
        # Check if rendering failed and show error log (outside st.status)
        if not video_path or not os.path.exists(video_path):
            if last_error_log:
                with st.expander("View Error"):
                    st.text(last_error_log)
            st.error("Could not generate animation")
            st.stop()

        # Display video
        st.video(video_path)

        # Convert to GIF
        with st.spinner("Converting to GIF..."):
            gif_path = video_to_gif(video_path, fps=min(fps, 20))
            if gif_path and os.path.exists(gif_path):
                st.divider()
                st.subheader("Download")
                with open(gif_path, "rb") as f:
                    st.download_button(
                        "Download GIF",
                        data=f.read(),
                        file_name="manim_animation.gif",
                        mime="image/gif",
                        use_container_width=True,
                    )

    elif st.session_state.get("generated_code"):
         # Show editor if code exists but no new prompt processing
         pass
    elif not prompt:
        st.info("Enter a prompt and click 'Generate Animation' to get started")

    # Code Editor Section
    if st.session_state.get("generated_code"):
        st.markdown("---")
        st.subheader("Edit Code")
        
        edited_code = st.text_area(
            "Manim Code",
            value=st.session_state.generated_code,
            height=400,
            key="code_editor"
        )
        
        if st.button("Re-render Changes"):
            st.session_state.generated_code = edited_code
            # Trigger re-render logic by setting flag or calling function
            # Since the main render loop is above, we might need to refactor or just rerun
            # Simpler: just rerun script with new code in state
            st.rerun()

# Logic to handle re-rendering from manual edits needs to be integrated into the main flow
# Ideally, we separate "generation" from "rendering".

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with ManimGL + Streamlit | 
        <a href="https://github.com/3b1b/manim" style="color: #3a7bd5;">ManimGL Docs</a>
    </div>
    """,
    unsafe_allow_html=True,
)
