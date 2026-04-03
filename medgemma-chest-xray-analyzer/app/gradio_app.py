"""
Enhanced Gradio web interface for MedGemma Chest X-Ray Analyzer.

Launch:
    python app/gradio_app.py
    python -m app.gradio_app
"""

import os
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from configs.model_config import (
    AUTO_DETECT_PROMPT,
    AUTO_DETECT_SYSTEM_PROMPT,
    DISCLAIMER,
    IMAGE_TYPE_DETAILED_PROMPTS,
    IMAGE_TYPE_LABELS,
    IMAGE_TYPE_SYSTEM_PROMPTS,
    MODEL_ID,
    PROMPT_LABELS,
    PROMPTS,
    SAMPLE_XRAYS,
)
from src.analyzer import ChestXRayAnalyzer
from src.image_utils import load_image_from_url, validate_xray_image

load_dotenv()

# ─── Globals ──────────────────────────────────────────────────────────────────
_analyzer: Optional[ChestXRayAnalyzer] = None
_current_mode: str = "quantized"

ANALYSIS_CHOICES = list(PROMPT_LABELS.values())
ANALYSIS_KEY_MAP = {v: k for k, v in PROMPT_LABELS.items()}
SAMPLE_CHOICES = ["-- Upload your own --"] + list(SAMPLE_XRAYS.keys())

# Maps UI label → internal key, e.g. "🧠 Brain MRI" → "mri_brain"
IMAGE_TYPE_CHOICES = list(IMAGE_TYPE_LABELS.values())
IMAGE_TYPE_LABEL_TO_KEY = {v: k for k, v in IMAGE_TYPE_LABELS.items()}

REPORT_PLACEHOLDER = "*Your radiology report will appear here after analysis...*"

CSS = """
/* ── Layout ──────────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* ── Header ──────────────────────────────────────────────────────────────── */
#app-header {
    text-align: center;
    padding: 28px 0 12px;
    background: linear-gradient(135deg, #1d4ed8 0%, #0891b2 100%);
    border-radius: 14px;
    margin-bottom: 16px;
    color: white;
}
#app-header h1 { font-size: 2.1em; margin: 0 0 6px; color: white; }
#app-header p  { font-size: 1em; margin: 0; opacity: 0.9; }

/* ── Disclaimer ───────────────────────────────────────────────────────────── */
#disclaimer {
    background: #fef3c7;
    border-left: 5px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 0.88em;
    color: #92400e;
    margin-bottom: 12px;
}

/* ── Image info bar ───────────────────────────────────────────────────────── */
#img-info-bar {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 0.84em;
    color: #1e40af;
    margin-top: 6px;
    min-height: 30px;
}

/* ── Analyze button ───────────────────────────────────────────────────────── */
#analyze-btn { font-size: 1.08em !important; height: 50px !important; }

/* ── Report Markdown panel ────────────────────────────────────────────────── */
#report-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 22px 26px;
    min-height: 440px;
    font-size: 0.91em;
    line-height: 1.75;
    overflow-y: auto;
}

/* ── Status box ───────────────────────────────────────────────────────────── */
#status-row textarea { font-size: 0.88em !important; color: #374151; }

/* ── Footer ───────────────────────────────────────────────────────────────── */
#footer {
    text-align: center;
    font-size: 0.78em;
    color: #9ca3af;
    padding: 14px 0 6px;
    border-top: 1px solid #f3f4f6;
    margin-top: 10px;
}
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

# Known image type keys (excludes "auto" which is a meta-option, not a real type)
_KNOWN_TYPE_KEYS = [k for k in IMAGE_TYPE_LABELS if k != "auto"]


def _detect_image_type(analyzer: ChestXRayAnalyzer, image: Image.Image) -> str:
    """Run a short classification inference to identify the imaging modality.

    Sends a constrained prompt asking MedGemma to reply with exactly one label
    from the known type list.  We cap the response at 30 tokens — only a single
    label word is expected back, so there is no need to generate a full report.

    Returns the matched image_type key (e.g. 'mri_brain') or 'chest_xray' as
    a safe fallback if the response cannot be matched to any known type.
    """
    try:
        result = analyzer.analyze(
            image=image,
            custom_prompt=AUTO_DETECT_PROMPT,
            system_prompt=AUTO_DETECT_SYSTEM_PROMPT,
            max_new_tokens=30,   # short response — we only need one label word
        )
        response = result["report"].strip().lower()
        # Scan the response for the first recognised type key
        for key in _KNOWN_TYPE_KEYS:
            if key in response:
                return key
    except Exception:
        pass  # fall through to the default below

    # Default fallback — chest X-ray is the most common medical image type
    return "chest_xray"


def _get_analyzer(mode: str) -> ChestXRayAnalyzer:
    global _analyzer, _current_mode
    if _analyzer is None or _current_mode != mode:
        _analyzer = ChestXRayAnalyzer(mode=mode)
        _analyzer.load_model()
        _current_mode = mode
    return _analyzer


def _format_report_markdown(result: dict, elapsed: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return "\n".join([
        "## 🫁 Radiology Report",
        "",
        "| | |",
        "|---|---|",
        f"| **Model** | `{result['model']}` |",
        f"| **Analysis** | {result['prompt_used']} |",
        f"| **Inference time** | {elapsed:.1f}s |",
        f"| **Generated** | {ts} |",
        "",
        "---",
        "",
        result["report"],
        "",
        "---",
        "",
        f"> ⚠️ *{DISCLAIMER}*",
    ])


def _save_report(report_md: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=f"xray_report_{ts}_",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(report_md)
    tmp.close()
    return tmp.name


# ─── Callbacks ────────────────────────────────────────────────────────────────

def analyze_xray(uploaded_image, sample_choice, image_type, analysis_type, custom_prompt, model_mode):
    """Returns (report_md, status, img_info_html, download_file_path).

    image_type  — UI label like "🧠 Brain MRI"; controls which specialist system
                  prompt and which detailed-report prompt MedGemma receives.
                  This is the fix for the bug where brain MRI images were analyzed
                  as chest X-rays (lungs/heart reported instead of brain structures).
    """

    # 1. Resolve image
    image: Optional[Image.Image] = None
    if uploaded_image is not None:
        image = uploaded_image
    elif sample_choice and sample_choice != "-- Upload your own --":
        url = SAMPLE_XRAYS.get(sample_choice)
        if url is None:
            return REPORT_PLACEHOLDER, "❌ Unknown sample.", "", None
        try:
            image = load_image_from_url(url)
        except Exception as exc:
            return REPORT_PLACEHOLDER, f"❌ Download failed: {exc}", "", None
    else:
        return REPORT_PLACEHOLDER, "⚠️ Please upload an image or select a sample.", "", None

    # 2. Validate image dimensions
    val = validate_xray_image(image)
    warn_str = ""
    if val["warnings"]:
        warn_str = "  ⚠️ " + "; ".join(val["warnings"])
    img_info_html = (
        f'<div id="img-info-bar">📐 {val["width"]} × {val["height"]} px'
        f'&nbsp;|&nbsp;Mode: {val["mode"]}{warn_str}</div>'
    )

    # 3. Load model
    try:
        analyzer = _get_analyzer(model_mode)
    except Exception as exc:
        return REPORT_PLACEHOLDER, f"❌ Model load failed: {exc}", img_info_html, None

    # 4. Resolve image type key from the UI label
    #    e.g. "🧠 Brain MRI" → "mri_brain", "🔍 Auto-detect" → "auto"
    image_type_key = IMAGE_TYPE_LABEL_TO_KEY.get(image_type, "chest_xray")

    # ── Auto-detection ─────────────────────────────────────────────────────
    # When the user selects "Auto-detect", run a short classification inference
    # (capped at 30 tokens) to ask MedGemma what type of image it sees.
    # This prevents the common mistake of uploading a brain MRI and forgetting
    # to change the dropdown — the model will identify the modality itself.
    auto_note = ""
    if image_type_key == "auto":
        detected_key = _detect_image_type(analyzer, image)
        detected_label = IMAGE_TYPE_LABELS.get(detected_key, detected_key)
        auto_note = f" | Auto-detected: {detected_label}"
        image_type_key = detected_key   # use detected type for the main analysis

    # Pick the specialist system prompt for this modality.
    # This tells MedGemma WHICH doctor to act as — neuroradiologist for brain MRI,
    # pathologist for slides, chest radiologist for X-rays, etc.
    sys_prompt = IMAGE_TYPE_SYSTEM_PROMPTS.get(
        image_type_key, IMAGE_TYPE_SYSTEM_PROMPTS["chest_xray"]
    )

    prompt_key = ANALYSIS_KEY_MAP.get(analysis_type, "detailed_report")
    user_custom = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    # For "detailed_report", use the modality-specific prompt that covers the
    # correct anatomical sections (brain MRI → ventricles/white matter, not lungs).
    # Skip this override if the user typed a custom prompt.
    if prompt_key == "detailed_report" and not user_custom:
        user_custom = IMAGE_TYPE_DETAILED_PROMPTS.get(
            image_type_key, IMAGE_TYPE_DETAILED_PROMPTS["chest_xray"]
        )

    # 5. Run main analysis inference with the modality-aware prompts
    try:
        t0 = time.time()
        result = analyzer.analyze(
            image=image,
            prompt_key=prompt_key,
            custom_prompt=user_custom,
            system_prompt=sys_prompt,
        )
        elapsed = time.time() - t0
    except Exception as exc:
        return REPORT_PLACEHOLDER, f"❌ Analysis failed: {exc}", img_info_html, None

    report_md = _format_report_markdown(result, elapsed)
    download_path = _save_report(report_md)
    detected_type_label = IMAGE_TYPE_LABELS.get(image_type_key, image_type_key)
    status = f"✅ Done in {elapsed:.1f}s  |  {val['width']}×{val['height']}px  |  {detected_type_label}{auto_note}{warn_str}"
    return report_md, status, img_info_html, download_path


def load_sample_image(sample_choice: str):
    if not sample_choice or sample_choice == "-- Upload your own --":
        return None
    url = SAMPLE_XRAYS.get(sample_choice)
    if not url:
        return None
    try:
        return load_image_from_url(url)
    except Exception:
        return None


def clear_all():
    return (
        None,                        # image_input
        "-- Upload your own --",     # sample_dropdown
        IMAGE_TYPE_CHOICES[0],       # image_type_dropdown (reset to Chest X-Ray)
        ANALYSIS_CHOICES[0],         # analysis_dropdown
        "",                          # custom_prompt_box
        "",                          # status_box
        REPORT_PLACEHOLDER,          # report_panel
        "",                          # img_info_html
        None,                        # download_file
    )


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="MedGemma Chest X-Ray Analyzer") as demo:

        # Header
        gr.HTML("""
        <div id="app-header">
            <h1>🫁 MedGemma Chest X-Ray Analyzer</h1>
            <p>AI-powered radiology report generation using <strong>Google's MedGemma 1.5 (4B)</strong></p>
        </div>
        """)

        gr.HTML(
            '<div id="disclaimer">⚠️ <strong>Research Use Only</strong> — '
            'Not for clinical diagnosis. Always consult a qualified radiologist or physician.</div>'
        )

        with gr.Tabs():

            # ── Tab 1: Analyzer ───────────────────────────────────────────────
            with gr.Tab("🔬 Analyzer"):
                with gr.Row(equal_height=False):

                    # ── Left: Inputs ──────────────────────────────────────────
                    with gr.Column(scale=1, min_width=320):

                        gr.Markdown("### 📤 Input Image")

                        image_input = gr.Image(
                            label="Upload Chest X-Ray",
                            type="pil",
                            height=290,
                        )

                        sample_dropdown = gr.Dropdown(
                            choices=SAMPLE_CHOICES,
                            value="-- Upload your own --",
                            label="Or load a sample X-ray",
                            interactive=True,
                        )

                        img_info_html = gr.HTML("")

                        gr.Markdown("### ⚙️ Analysis Options")

                        # ── Image Type selector ───────────────────────────────
                        # IMPORTANT: Select the modality that matches your image.
                        # This controls which specialist role MedGemma plays and
                        # which anatomical sections appear in the report.
                        # Uploading a brain MRI with "Chest X-Ray" selected will
                        # produce a report about lungs — always match this to
                        # your actual image type.
                        image_type_dropdown = gr.Dropdown(
                            choices=IMAGE_TYPE_CHOICES,
                            value=IMAGE_TYPE_CHOICES[0],   # defaults to Chest X-Ray
                            label="Image Type (match this to your image!)",
                            interactive=True,
                        )

                        analysis_dropdown = gr.Dropdown(
                            choices=ANALYSIS_CHOICES,
                            value=ANALYSIS_CHOICES[0],
                            label="Analysis Type",
                            interactive=True,
                        )

                        custom_prompt_box = gr.Textbox(
                            label="Custom Prompt (optional — overrides Analysis Type)",
                            placeholder="e.g. Focus on the left lower lobe consolidation...",
                            lines=2,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            mode_radio = gr.Radio(
                                choices=["quantized", "pipeline", "full"],
                                value="quantized",
                                label="Model Mode",
                                info=(
                                    "quantized ≈ 6 GB VRAM (recommended)  |  "
                                    "pipeline ≈ 14 GB  |  full = 4-bit BnB"
                                ),
                            )

                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔬 Analyze X-Ray",
                                variant="primary",
                                size="lg",
                                elem_id="analyze-btn",
                                scale=3,
                            )
                            clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)

                    # ── Right: Output ─────────────────────────────────────────
                    with gr.Column(scale=1, min_width=400):

                        gr.Markdown("### 📋 Radiology Report")

                        status_box = gr.Textbox(
                            label="Status",
                            lines=1,
                            interactive=False,
                            elem_id="status-row",
                        )

                        report_panel = gr.Markdown(
                            value=REPORT_PLACEHOLDER,
                            elem_id="report-panel",
                        )

                        download_file = gr.File(
                            label="Download Report (.txt)",
                            interactive=False,
                        )

            # ── Tab 2: How to Use ─────────────────────────────────────────────
            with gr.Tab("📖 How to Use"):
                gr.Markdown("""
## How to Use

### Steps
1. **Upload** a chest X-ray image (PNG, JPG, TIFF) — or pick a **sample** from the dropdown
2. **Choose** an analysis type
3. Optionally enter a **custom prompt** for a specific clinical question
4. Click **Analyze X-Ray** and wait for the report
5. **Download** the report as a `.txt` file using the download button

---

### Analysis Types

| Type | What it does |
|------|-------------|
| 📋 **Detailed Radiology Report** | Full structured report covering Lungs, Pleura, Heart, Mediastinum, Bones + Impression |
| 🔍 **General Description** | High-level summary of visible structures |
| 📌 **Key Findings (Bullet Points)** | Concise bulleted list of findings |
| ⚖️ **Compare to Normal** | Highlights deviations from a normal chest X-ray |
| 💬 **Patient-Friendly Explanation** | Plain language explanation for non-medical readers |

---

### Model Modes

| Mode | VRAM | Notes |
|------|------|-------|
| **quantized** | ~6 GB | Pre-quantized 4-bit via Unsloth — fastest, recommended |
| **pipeline** | ~14 GB | Full bfloat16 via HuggingFace pipeline |
| **full** | ~6 GB | BitsAndBytes 4-bit on-the-fly quantization |

---

### Tips
- Use **Custom Prompt** to ask specific clinical questions (e.g. *"Is there evidence of pneumothorax?"*)
- For best results use high-resolution PA (posterior-anterior) view X-rays
- Running locally requires a CUDA GPU with ≥6 GB VRAM
- For CPU-only machines use [Google Colab](https://colab.research.google.com/) with a T4 GPU
                """)

            # ── Tab 3: About ──────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
## About This Project

**MedGemma Chest X-Ray Analyzer** is a PhD research project demonstrating how Google's MedGemma 1.5 (4B) medical vision-language model can power automated chest X-ray report generation.

---

### Model Details

| Field | Value |
|-------|-------|
| **Model ID** | `google/medgemma-1.5-4b-it` |
| **Architecture** | Gemma 3 backbone + medical domain adaptation |
| **Parameters** | 4 billion |
| **Input** | Image + text prompt |
| **Output** | Structured text report |
| **License** | Requires HuggingFace account + license acceptance |

---

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Model inference | HuggingFace Transformers, PyTorch |
| Quantization | BitsAndBytes / Unsloth |
| Web interface | Gradio |
| Image processing | Pillow |
| Config/secrets | python-dotenv |

---

### Research Context

Developed as part of PhD research in AI-assisted radiology. The goal is to explore how large multimodal models can generate clinically structured reports from raw imaging data, bridging the gap between deep learning and practical radiology workflows.

---

### Disclaimer

> {DISCLAIMER}
                """)

        # Footer
        gr.HTML(
            '<div id="footer">'
            '🤖 Powered by <code>google/medgemma-1.5-4b-it</code> &nbsp;|&nbsp; '
            'Built for PhD Research &nbsp;|&nbsp; '
            'Not for clinical use'
            '</div>'
        )

        # ── Event Wiring ──────────────────────────────────────────────────────
        sample_dropdown.change(
            fn=load_sample_image,
            inputs=[sample_dropdown],
            outputs=[image_input],
        )

        analyze_btn.click(
            fn=analyze_xray,
            # image_type_dropdown is now passed as the 3rd argument so the callback
            # can look up the right specialist system prompt and detailed-report prompt.
            inputs=[image_input, sample_dropdown, image_type_dropdown,
                    analysis_dropdown, custom_prompt_box, mode_radio],
            outputs=[report_panel, status_box, img_info_html, download_file],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[
                image_input, sample_dropdown, image_type_dropdown,
                analysis_dropdown, custom_prompt_box, status_box,
                report_panel, img_info_html, download_file,
            ],
        )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
