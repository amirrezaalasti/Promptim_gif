"""LLM-based Manim code generation from natural language prompts."""

import re
from pathlib import Path
from typing import Optional, Tuple

PLANNING_PROMPT = """You are an expert at planning mathematical animations. Given a simple user request, generate a detailed scene description.

Think step by step about:
1. What visual elements are needed (shapes, text, equations, coordinate systems)
2. The sequence of animations and transitions
3. How concepts connect visually
4. The timing and flow of the explanation
5. Colors and positioning that will make it clear

Be specific and actionable. Your description will be used to generate Manim code.

Output a structured plan with:
- Visual Elements: What objects will appear
- Animation Sequence: Step-by-step what happens
- Timing: How long each part takes
- Positioning: Where things are placed

Format your response as a clear, detailed plan."""

MANIM_EXAMPLES = """
# EXAMPLE 1: Coordinate System, Graph, and Proper Alignment
class GraphExample(Scene):
    def construct(self):
        # Always define ranges with 3 values [min, max, step]
        axes = Axes(x_range=[-3, 10, 1], y_range=[-1, 8, 1], height=6)
        axes.add_coordinate_labels()
        self.play(Write(axes))

        sin_graph = axes.get_graph(lambda x: 2 * math.sin(x), color=BLUE)
        self.play(ShowCreation(sin_graph))

        # ALIGNMENT: Use axes.c2p(x, y) to place objects relative to the chart
        dot_pos = axes.c2p(2, 2 * math.sin(2)) 
        dot = Dot(point=dot_pos, color=RED)
        
        # LABELING: Use next_to with the mobject
        label = Text("Point P", font_size=24).next_to(dot, UP)
        self.play(FadeIn(dot), Write(label))
        
        # MOVEMENT: Use axes.i2gp (input to graph point) to follow a curve
        self.play(dot.animate.move_to(axes.i2gp(4, sin_graph)), run_time=2)

# EXAMPLE 2: Dual-Domain Transitions (Fourier Transform Style)
class FourierTransition(Scene):
    def construct(self):
        # Position axes specifically to avoid center-screen overlap
        axes_time = Axes(x_range=[0, 10, 1], y_range=[-2, 2, 1], height=3).shift(LEFT * 3)
        wave = axes_time.get_graph(lambda x: np.sin(x), color=BLUE)
        self.add(axes_time, wave)
        
        axes_freq = Axes(x_range=[0, 5, 1], y_range=[0, 2, 1], height=3).shift(RIGHT * 3)
        self.play(Write(axes_freq))
        
        # Representing a frequency spike using c2p mapping
        spike = Line(axes_freq.c2p(2, 0), axes_freq.c2p(2, 1.5), color=YELLOW)
        self.play(ShowCreation(spike))
"""

SYSTEM_PROMPT = f"""You are an expert Manim developer. Given a scene description, generate working Manim (manimgl) Python code.

REFERENCE EXAMPLES:
{MANIM_EXAMPLES}

CRITICAL RULES - MUST FOLLOW:

1. TEXT SPACING & FONT FIXES:
   - DO NOT use math strings for standard text (e.g., avoid Tex("DomainSignal")).
   - For titles or labels, ALWAYS use Text("Title With Proper Spaces") or provide spaces in Tex(r"Domain Signal").
   - Use font_size=36 for titles and 24-28 for labels to prevent overlapping.

2. COORDINATE ALIGNMENT (THE "ALIGNMENT FIX"):
   - NEVER place a Dot or Arrow using raw coordinates like `np.array([-2, 4, 0])`.
   - ALWAYS use `axes.c2p(x, y)` to map mathematical values to the screen.
   - ALWAYS use `axes.i2gp(x, graph)` to anchor dots to function lines.

3. NO HALLUCINATED METHODS:
   - These DO NOT EXIST: .bounce(), .jump(), .shimmer(), .pulse(), .blink().
   - Use rate functions instead: .animate(rate_func=there_and_back).shift(UP*0.3).

4. LAYOUT & SCREEN FITTING:
   - Titles: text.to_edge(UP, buff=0.3).
   - Dual Graphs: Use shift(LEFT*3) and shift(RIGHT*3) to prevent collision in the center.

5. MANIMGL SPECIFICS:
   - Use `Tex(r"...")`, NEVER `MathTex`.
   - DO NOT use $ delimiters inside `Tex(r"...")`. ManimGL handles math mode globally.

Output ONLY the Python code. No explanations, no markdown blocks."""

def _clean_code(code: str) -> str:
    """Clean generated code by removing markdown and ensuring proper imports."""
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()

    lines = code.split("\n")
    fixed_lines = []
    for line in lines:
        # Fix invalid color names
        color_replacements = {
            "LIGHT_BLUE": "BLUE_A", "DARK_BLUE": "BLUE_E",
            "LIGHT_RED": "RED_A", "DARK_RED": "RED_E",
            "LIGHT_GREEN": "GREEN_A", "DARK_GREEN": "GREEN_E",
            "ORANGE_A": "ORANGE", "PINK_A": "LIGHT_PINK",
            "GRAY": "GREY", "DARK_GRAY": "GREY_D", "LIGHT_GRAY": "GREY_A"
        }
        for invalid, valid in color_replacements.items():
            line = line.replace(invalid, valid)

        # Fix Tex() usage - aggressively remove hallucinated $ delimiters
        if "Tex(" in line:
            line = re.sub(r'Tex\(r"(\$[^$]+\$)"\)', r'Tex(r"\1")', line)
            line = line.replace(r"$", "")

        # Fix MathTex -> Tex (ManimGL compatibility)
        line = line.replace("MathTex", "Tex")

        # Fix Axes range format: ensure [min, max, step]
        if "range=[" in line:
            line = re.sub(r"range=\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]", r"range=[\1, \2, 1]", line)

        # Correct .animate syntax errors
        if "self.play(" in line and ".move_to," in line:
            line = re.sub(r"self\.play\((\w+)\.move_to,\s*([^)]+)\)", r"self.play(\1.animate.move_to(\2))", line)

        fixed_lines.append(line)

    code = "\n".join(fixed_lines)
    
    # Ensure all required imports are present
    required_imports = ["from manimlib import *", "import numpy as np", "import math"]
    for imp in reversed(required_imports):
        if imp not in code:
            code = f"{imp}\n" + code
            
    return code.strip()

def plan_scene(
    prompt: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    model_path: Optional[str] = None,
    image_files: list[str] = None,
) -> str:
    """Plan a scene by thinking through the animation step by step.

    Args:
        prompt: Simple user request
        provider: "openai" or "deepseek"
        api_key: API key for OpenAI
        model: Model name for OpenAI
        model_path: Path to local DeepSeek model
        image_files: List of available image filenames
    """
    
    # Inject available images into prompt
    if image_files:
        prompt += f"\n\nAvailable images: {', '.join(image_files)}"
        prompt += "\nNOTE: Only use these images if the user explicitly asks for them in the prompt."

    if provider == "openai":
        return _plan_with_openai(prompt, api_key, model)
    elif provider == "deepseek":
        return _plan_with_deepseek(prompt, model_path)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_manim_code(
    plan: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    model_path: Optional[str] = None,
    image_files: list[str] = None,
) -> str:
    """Generate Manim code from a detailed scene plan.

    Args:
        plan: Detailed scene description/plan
        provider: "openai" or "deepseek"
        api_key: API key for OpenAI
        model: Model name for OpenAI
        model_path: Path to local DeepSeek model
        image_files: List of available image filenames
    """
    
    # Inject available images into plan so code generator knows about them
    if image_files:
        plan += f"\n\nAvailable images: {', '.join(image_files)}"
        plan += "\nNOTE: Only use these images if the plan explicitly calls for them."

    if provider == "openai":
        return _generate_with_openai(plan, api_key, model)
    elif provider == "deepseek":
        return _generate_with_deepseek(plan, model_path)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_manim_code_with_planning(
    prompt: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    model_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate Manim code with a planning step first.

    Args:
        prompt: Simple user request
        provider: "openai" or "deepseek"
        api_key: API key for OpenAI
        model: Model name for OpenAI
        model_path: Path to local DeepSeek model

    Returns:
        Tuple of (plan, code)
    """
    plan = plan_scene(prompt, provider, api_key, model, model_path)
    code = generate_manim_code(plan, provider, api_key, model, model_path)
    return plan, code


def _plan_with_openai(prompt: str, api_key: str, model: str) -> str:
    """Plan scene using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    if not api_key:
        raise ValueError("OpenAI API key is required")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": PLANNING_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=2048
    )

    return response.choices[0].message.content.strip()


def _plan_with_deepseek(prompt: str, model_path: Optional[str]) -> str:
    """Plan scene using local DeepSeek model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers torch")

    if model_path is None:
        model_path = "./deepseek-manim-finetuned"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    full_prompt = f"""<|begin_of_sentence|>### System:
{PLANNING_PROMPT}

### User:
{prompt}

### Assistant:
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()

    return response


def _generate_with_openai(prompt: str, api_key: str, model: str) -> str:
    """Generate code using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    if not api_key:
        raise ValueError("OpenAI API key is required")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=4096
    )

    code = response.choices[0].message.content.strip()
    return _clean_code(code)


def _generate_with_deepseek(prompt: str, model_path: Optional[str]) -> str:
    """Generate code using local DeepSeek model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers torch")

    if model_path is None:
        model_path = "./deepseek-manim-finetuned"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Check if this is a LoRA adapter or full model
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    full_prompt = f"""<|begin_of_sentence|>### System:
{SYSTEM_PROMPT}

### User:
{prompt}

### Assistant:
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()

    return _clean_code(response)

def _fix_indentation(code: str) -> str:
    """Fix common indentation errors in generated code."""
    lines = code.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append("")
            continue

        # If line is just a closing paren/bracket, check if it's mis-indented
        if stripped in (")", "]", "}", "),", "],", "},") and i > 0:
            # Look backwards for matching opening
            for j in range(i - 1, max(-1, i - 20), -1):
                if j >= 0 and j < len(lines):
                    prev = lines[j].rstrip()
                    if prev and (
                        prev.endswith("(") or prev.endswith("[") or prev.endswith("{")
                    ):
                        # Match indentation of the line with opening
                        expected_indent = (
                            len(lines[j]) - len(lines[j].lstrip()) if j > 0 else 0
                        )
                        if j > 0:
                            expected_indent = len(lines[j]) - len(lines[j].lstrip())
                        else:
                            expected_indent = 0
                        fixed_lines.append(" " * expected_indent + stripped)
                        break
            else:
                # No matching opening found, keep original
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


FIX_CODE_PROMPT = """You are an expert Manim developer debugging code. The following Manim code produced an error.

Fix the code to resolve the error. Output ONLY the corrected Python code, no explanations.

Common fixes:
- Use manimgl syntax, not manim community edition
- get_axis_labels() takes string arguments, not Tex objects
- Use ShowCreation instead of Create
- Use self.play() not self.wait() for animations (BUT use self.wait(1) for pauses, NOT self.play(Wait(1)))
- Ensure all mobjects are created before animating
- Use proper numpy imports (import numpy as np)
- Fix indentation errors - Python requires consistent indentation (4 spaces per level)
- Ensure all parentheses, brackets, and braces are properly closed
- Check that method definitions (def construct(self):) are properly indented
- SCREEN FITTING: Reduce font sizes (48->36, 36->28, 32->24, 24->20)
- SCREEN FITTING: Reduce spacing (LEFT*3->LEFT*2, UP*2->UP*1, buff=1->buff=0.3)
- SCREEN FITTING: Make shapes smaller (width=2->1.5, height=1->0.8, radius=1->0.8)
- SCREEN FITTING: Always position titles with to_edge(UP, buff=0.3)
- SCREEN FITTING: Always position bottom text with to_edge(DOWN, buff=0.3)
- SCREEN FITTING: Keep main content centered around ORIGIN, not spread out
- CRITICAL: For animating mobject methods, use .animate syntax:
  * WRONG: self.play(mobject.move_to, position)
  * CORRECT: self.play(mobject.animate.move_to(position))
  * WRONG: self.play(mobject.scale, 2)
  * CORRECT: self.play(mobject.animate.scale(2))
  * WRONG: self.play(mobject.set_color, RED)
  * CORRECT: self.play(mobject.animate.set_color(RED))
- CRITICAL: NO HALLUCINATED METHODS:
  * The .bounce() method DOES NOT EXIST. Use .animate.shift(UP).set_rate_func(there_and_back)
  * The .jump() method DOES NOT EXIST. Use .animate.shift(UP)
  * The .shimmer() method DOES NOT EXIST. Use .animate.set_opacity(0.8).set_rate_func(there_and_back)
  * The .Wait() class DOES NOT EXIST. Use self.wait(seconds) instead of self.play(Wait(seconds))
- CRITICAL: `get_tangent_line` usage:
  * WRONG: axes.get_tangent_line(x, graph, color=RED)
  * CORRECT: axes.get_tangent_line(x, graph).set_color(RED)
- CRITICAL: Use only valid ManimGL colors:
  * Valid: BLUE, RED, YELLOW, GREEN, WHITE, PINK, PURPLE, ORANGE, GREY, BLACK
  * Valid variants (A-E): BLUE, TEAL, GREEN, YELLOW, GOLD, RED, MAROON, PURPLE, GREY
  * NO VARIANTS: ORANGE (use GOLD_A-E for variants), PINK (use LIGHT_PINK), WHITE, BLACK
  * WRONG: ORANGE_A, ORANGE_B, PINK_A, DARK_RED (use RED_E)
  * Use BLUE_A instead of LIGHT_BLUE, BLUE_E instead of DARK_BLUE
- CRITICAL: Tex() for LaTeX math - DO NOT use $ delimiters:
  * CORRECT: Tex(r"\\theta = 0") or Tex(r"\\frac{a}{b}")
  * WRONG: Tex(r"$\\theta = 0$") - ManimGL handles math mode automatically!
  * The $ signs cause LaTeX compilation errors

ORIGINAL CODE:
```python
{code}
```

ERROR:
{error}

Output the FIXED code only:"""


def fix_code(
    code: str,
    error: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    model_path: Optional[str] = None,
) -> str:
    """Fix Manim code based on error message.

    Args:
        code: The original code that failed
        error: The error message from rendering
        provider: "openai" or "deepseek"
        api_key: API key for OpenAI
        model: Model name for OpenAI
        model_path: Path to local DeepSeek model

    Returns:
        Fixed Manim Python code
    """
    # Validate inputs
    if not code or len(code.strip()) < 10:
        raise ValueError("Invalid code provided for fixing")

    if not error or len(error.strip()) < 5:
        error = "Rendering failed - please check the code for syntax errors, missing imports, or incorrect ManimGL API usage"

    # Truncate error if too long (to avoid token limits)
    if len(error) > 2000:
        error = error[:1900] + "\n... (error truncated)"

    # Escape curly braces in code/error to prevent KeyError in format()
    # Double all { and } so they're treated as literals, not format placeholders
    escaped_code = code.replace("{", "{{").replace("}", "}}")
    escaped_error = error.replace("{", "{{").replace("}", "}}")

    # Now format with escaped strings - the {code} and {error} in the template will be replaced
    try:
        fix_prompt = FIX_CODE_PROMPT.format(code=escaped_code, error=escaped_error)
    except KeyError as e:
        # Fallback: manual replacement if format() fails
        fix_prompt = FIX_CODE_PROMPT.replace("{code}", escaped_code).replace(
            "{error}", escaped_error
        )

    if provider == "openai":
        return _fix_with_openai(fix_prompt, api_key, model)
    elif provider == "deepseek":
        return _fix_with_deepseek(fix_prompt, model_path)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _fix_with_openai(prompt: str, api_key: str, model: str) -> str:
    """Fix code using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    if not api_key:
        raise ValueError("OpenAI API key is required")

    client = OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": "You are an expert Manim developer. Fix the code and output only the corrected Python code.",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, max_tokens=4096
    )

    code = response.choices[0].message.content.strip()
    return _clean_code(code)


def _fix_with_deepseek(prompt: str, model_path: Optional[str]) -> str:
    """Fix code using local DeepSeek model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers torch")

    if model_path is None:
        model_path = "./deepseek-manim-finetuned"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    full_prompt = f"""<|begin_of_sentence|>### System:
You are an expert Manim developer. Fix the code and output only the corrected Python code.

### User:
{prompt}

### Assistant:
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()

    return _clean_code(response)
