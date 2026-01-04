# ManimForge

A Streamlit web application that generates beautiful mathematical animations from simple text prompts using ManimGL and LLMs.

## Features

- **Natural Language to Animation**: Describe what you want to see, get a Manim animation
- **Multiple LLM Support**: Use OpenAI GPT models or your fine-tuned DeepSeek model
- **Export Options**: Download as GIF or PowerPoint presentation
- **Customizable Rendering**: Adjust resolution, FPS, and quality settings
- **Modern UI**: Clean, dark-themed interface with intuitive controls

## Installation

1. Make sure you have ManimGL installed in the parent directory (`../manim`)

2. Install dependencies:
```bash
cd manim_generator
pip install -r requirements.txt
```

3. For GIF conversion, ensure you have either:
   - `imageio` with `imageio-ffmpeg` (installed via requirements.txt)
   - OR `ffmpeg` installed on your system

## Usage

### Running the App

```bash
cd manim_generator
streamlit run app.py
```

This will open the app in your browser at `http://localhost:8501`

### Using OpenAI

1. Select "OpenAI" as the LLM Provider in the sidebar
2. Enter your OpenAI API key
3. Choose a model (gpt-4o-mini recommended for speed)
4. Enter your prompt and generate!

### Using Fine-tuned DeepSeek (Local)

1. First, train your model using `../llm_finetune/finetune_deepseek.ipynb`
2. Select "DeepSeek (Local)" in the sidebar
3. Enter the path to your fine-tuned model
4. Generate animations locally without API costs

## Example Prompts

- "Show gradient descent on a parabola with a dot moving to the minimum"
- "Visualize the Pythagorean theorem with squares on each side of a triangle"
- "Create a sine wave being drawn with a rotating vector"
- "Demonstrate matrix multiplication with animated vectors"
- "Show the relationship between sine and cosine functions"

## Project Structure

```
manim_generator/
├── app.py              # Main Streamlit application
├── code_generator.py   # LLM code generation module
├── manim_renderer.py   # Manim rendering utilities
├── ppt_creator.py      # PowerPoint creation
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── temp/               # Temporary files (auto-created)
```

## Configuration Options

### LLM Settings
- **Provider**: OpenAI or DeepSeek (local)
- **Model**: GPT-4o-mini, GPT-4o, or GPT-4-turbo (OpenAI)

### Output Format
- **GIF**: Animated GIF for embedding in presentations or sharing
- **PowerPoint**: PPTX file with the animation embedded

### Render Settings
- **Resolution**: 480p, 720p, or 1080p
- **FPS**: 15-60 frames per second
- **Quality**: Low, medium, or high (affects render time)

## Notes

- **Rendering Time**: Depending on complexity, rendering can take 1-5 minutes
- **PowerPoint GIFs**: GIFs will animate during slideshow mode in PowerPoint
- **GPU Recommended**: For local DeepSeek inference, a GPU with 16GB+ VRAM is recommended

## Troubleshooting

### "No video file generated"
- Check if ManimGL is properly installed
- Ensure the generated code has a valid Scene class
- Check the logs in the "View Logs" expander

### "OpenAI API error"
- Verify your API key is correct
- Check your OpenAI account has credits
- Ensure you have access to the selected model

### "DeepSeek model not found"
- Run the fine-tuning notebook first
- Verify the model path is correct
- Check if the adapter files exist

## License

MIT License

