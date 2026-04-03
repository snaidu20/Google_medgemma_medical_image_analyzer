# 🫁 MedGemma Medical Image Analyzer

> AI-powered **multi-modality** medical image analysis using **Google's MedGemma 1.5 (4B)** — a vision-language model trained specifically on medical data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Research%20Only-red)](#disclaimer)

---

## Table of Contents

- [What is MedGemma?](#what-is-medgemma)
- [What Can It Analyze?](#what-can-it-analyze)
- [Supported Image Types & Auto-Detection](#supported-image-types--auto-detection)
- [Why Not Local CPU?](#why-not-local-cpu)
- [Project Structure](#project-structure)
- [Run on Google Colab](#option-a-google-colab-recommended)
- [Run on Kaggle](#option-b-kaggle)
- [Analysis Types](#analysis-types)
- [Sample Output](#sample-output)
- [Tech Stack](#tech-stack)
- [Changelog](#changelog)
- [Disclaimer](#disclaimer)

---

## What is MedGemma?

**MedGemma 1.5 (4B)** is Google's open medical vision-language model built on the Gemma 3 architecture. It combines:

- A **SigLIP vision encoder** — pre-trained on de-identified medical imaging data
- A **4B parameter language model** — generates structured clinical text
- **128K token context window** — supports long medical documents and reports

| Property | Value |
|----------|-------|
| Model ID | `google/medgemma-1.5-4b-it` |
| Architecture | Gemma 3 + SigLIP vision encoder |
| Parameters | 4 Billion |
| Context Length | 128K tokens |
| Max Output | 8,192 tokens |
| Image Resolution | Normalized to 896×896 px (256 tokens/image) |
| Precision | bfloat16 |
| Decoding | Greedy (`do_sample=False`) |
| License | [Google Health AI Developer Foundations](https://huggingface.co/google/medgemma-1.5-4b-it) |

Unlike general-purpose vision models, MedGemma was trained on large-scale medical datasets including **32.5 million histopathology patch-text pairs**, TCGA (The Cancer Genome Atlas), CAMELYON, MIMIC-CXR, EyePACS, and de-identified EHR data — giving it deep domain knowledge across specialties.

**Real-world deployments:**
- 🇲🇾 **Malaysia — Qmed Asia**: Clinical practice guideline navigation
- 🇹🇼 **Taiwan — National Health Insurance Administration**: Preoperative lung cancer surgery assessment, processing 30,000+ pathology reports

---

## What Can It Analyze?

MedGemma 1.5 is not limited to chest X-rays. The analyzer now supports **10 distinct imaging modalities**, each with a dedicated specialist system prompt so the model acts as the correct domain expert (neuroradiologist for brain MRI, pathologist for slides, etc.).

### 🩻 Radiology

| Modality | Benchmark | Performance | Notes |
|----------|-----------|-------------|-------|
| **Chest X-ray** | MIMIC macro F1 | **88.9%** | Multi-label disease classification |
| **Chest X-ray (OOD)** | CXR14 macro F1 | 50.1% | Out-of-distribution generalization |
| **Anatomical Localization** | Chest ImaGenome IoU | **38.0%** | +35% improvement over MedGemma v1 |
| **CT Scan** | 7-condition classification | **61.1%** macro accuracy | +3% over v1 |
| **MRI** | 10-condition classification | **64.7%** macro accuracy | +14% over v1 |
| **Bone / Musculoskeletal X-ray** | — | General radiology reasoning | Fractures, lytic/blastic lesions |

### 🔬 Pathology & Oncology

| Modality | Benchmark | Performance | Notes |
|----------|-----------|-------------|-------|
| **Histopathology (WSI)** | PathMCQA | **70.0%** accuracy | Cancer ID, grading, subtype classification |
| **Colorectal Cancer Slides** | CRC100k weighted F1 | **94.5%** (fine-tuned) | Tissue-level classification |
| **Whole-Slide Diagnosis** | WSI-Path ROUGE-L | 49.4 | Full pathology report generation |
| **Lymph Node Metastasis** | CAMELYON | Trained dataset | Metastasis detection |

**Supported cancer types** (per official training data):
- Lung cancer (preoperative assessment, nodule detection)
- Breast cancer (identification, grading, subtype)
- Prostate cancer (Gleason grading)
- Cervical cancer / cervical dysplasia
- Colorectal adenocarcinoma
- General lymphoma (lymph node involvement)

### 👁️ Ophthalmology

| Modality | Benchmark | Performance |
|----------|-----------|-------------|
| **Retinal Fundus** | EyePACS diabetic retinopathy | **76.8%** accuracy |
| **OCT scans** | Retinal layer analysis | 2D cross-section support |

### 🩺 Dermatology

| Modality | Examples |
|----------|---------|
| **Clinical photography** | Skin lesions, rashes, moles (ABCDE evaluation) |
| **Dermoscopy** | Magnified lesion analysis, differential diagnosis |

### 🔊 Ultrasound & Other

| Modality | Examples |
|----------|---------|
| **Abdominal ultrasound** | Liver, kidney, gallbladder, masses |
| **Obstetric ultrasound** | Fetal imaging |
| **Cardiac ultrasound** | Echocardiography frames |
| **Mammography** | Breast density, BI-RADS assessment |

### 📄 Medical Documents (Text-based)

| Type | Benchmark | Performance |
|------|-----------|-------------|
| **EHR Q&A** | EHRQA | **89.6%** accuracy (+22% over v1) |
| **Lab Report Extraction** | Structured data F1 | **78%** retrieval F1 (+18% over v1) |
| **Medical QA** | MedQA | **69.1%** accuracy |
| **Radiology Report Generation** | RadGraph F1 | 29.5 F1 (81% same/superior clinical decision in human eval) |

> **Note:** MedGemma works on **2D images only**. For 3D volumetric CT/MRI, export individual axial/coronal slices and analyze them one at a time.

---

## Supported Image Types & Auto-Detection

### Image Type Dropdown (Gradio UI & Notebooks)

The analyzer now has an **Image Type** selector that controls which specialist role MedGemma plays and which anatomical sections appear in the report. **Always match this to your image.**

| UI Label | Internal Key | Specialist Role |
|----------|-------------|-----------------|
| 🔍 Auto-detect (let MedGemma identify) | `auto` | Runs a 30-token classification first, then picks the correct specialist |
| 🫁 Chest X-Ray | `chest_xray` | Chest radiologist — Lungs, Pleura, Heart, Mediastinum, Bones |
| 🔬 CT Scan | `ct_scan` | CT specialist — density, Hounsfield units, slice findings |
| 🧠 Brain MRI | `mri_brain` | Neuroradiologist — parenchyma, ventricles, white matter, midline shift |
| 🧲 Other MRI | `mri_other` | MRI generalist — spine, abdomen, MSK, pelvic |
| 🔮 Mammogram | `mammogram` | Breast radiologist — BI-RADS framework, density, calcifications |
| 🩺 Skin Lesion | `skin_lesion` | Dermatologist — ABCDE criteria, differential diagnosis |
| 🔬 Pathology Slide | `pathology_slide` | Pathologist — cell morphology, architecture, mitotic activity |
| 👁️ Retinal Fundus | `fundus` | Ophthalmologist — optic disc, macula, vasculature |
| 〰️ Ultrasound | `ultrasound` | Ultrasound specialist — echogenicity, organ morphology, cysts |
| 🦴 Bone X-Ray | `bone_xray` | MSK radiologist — cortex, trabecular bone, joint spaces, fractures |

### How Auto-Detection Works

When **Auto-detect** is selected, the analyzer runs two inferences:

1. **Classification inference** (30 tokens max) — asks MedGemma to return exactly one modality label (e.g., `mri_brain`)
2. **Full analysis inference** (1024 tokens) — runs with the detected modality's specialist system prompt and report structure

This prevents the common mistake of uploading a brain MRI and forgetting to update the dropdown — the model identifies the modality itself and generates the correct report sections.

> ⚠️ **Important:** Without correct image type selection, MedGemma will analyze a brain MRI as a chest X-ray and report findings about lungs, heart, and pleura instead of brain structures. Always use Auto-detect or manually select the correct type.

### Notebooks: `IMAGE_TYPE` variable

In the Colab and Kaggle notebooks, the last cell uses an `IMAGE_TYPE` variable:

```python
IMAGE_TYPE = 'auto'          # Let MedGemma detect the modality
# IMAGE_TYPE = 'chest_xray'
# IMAGE_TYPE = 'mri_brain'
# IMAGE_TYPE = 'ct_scan'
# IMAGE_TYPE = 'mammogram'
# IMAGE_TYPE = 'skin_lesion'
# IMAGE_TYPE = 'pathology_slide'
# IMAGE_TYPE = 'fundus'
# IMAGE_TYPE = 'ultrasound'
# IMAGE_TYPE = 'bone_xray'
# IMAGE_TYPE = 'mri_other'
```

The notebook automatically selects the correct system prompt and report structure based on this value.

---

## Why Not Local CPU?

You may wonder why this project cannot simply run on your laptop. Here's why:

### The Hardware Math

| Component | Requirement | Typical Laptop CPU |
|-----------|-------------|-------------------|
| Model size | ~8 GB in bfloat16 | ✅ RAM fits |
| Inference speed | GPU: ~10 sec/report | ❌ CPU: 30–90 min/report |
| Parallel matrix ops | Needs thousands of CUDA cores | ❌ CPU has 8–16 cores |
| Memory bandwidth | ~900 GB/s (GPU) | ❌ CPU: ~50 GB/s |

### The Software Conflict

The specific stack this model requires creates an impossible conflict on CPU:

```
MedGemma 1.5 (Gemma 3 architecture)
    └── requires transformers >= 5.x
            └── requires PyTorch >= 2.6 for masking operations
                    └── PyTorch 2.6 requires CUDA GPU (sm_70+)
```

- **PyTorch 2.6+** dropped support for CPU-only inference for this model's attention mechanism
- Running on CPU even with the right versions results in **30–90 minute inference times** per image — essentially unusable
- The 4B model needs sustained **high memory bandwidth** that only GPU VRAM provides

### The Solution: Free Cloud GPUs

| Platform | GPU | VRAM | Free? | Best For |
|----------|-----|------|-------|---------|
| Google Colab | T4 | 16 GB | ✅ Yes | Quick experiments |
| Kaggle | T4 x1 | 16 GB | ✅ Yes | Notebooks, datasets |

Both provide **~10 second inference** vs 30–90 minutes on CPU.

---

## Project Structure

```
medgemma-chest-xray-analyzer/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── run_analysis.py                  # CLI entry point
│
├── src/
│   ├── analyzer.py                  # Core MedGemma inference engine
│   └── image_utils.py               # Image loading and validation
│
├── app/
│   └── gradio_app.py                # Gradio web interface (GPU required)
│
├── configs/
│   └── model_config.py              # Model IDs, prompts, per-modality settings
│
├── notebooks/
│   ├── medgemma_chest_xray_colab.ipynb   # Google Colab notebook
│   └── medgemma_chest_xray_kaggle.ipynb  # Kaggle notebook
│
├── data/
│   └── sample_xrays/
│       └── download_samples.py      # Downloads public domain X-rays
│
├── tests/
│   └── test_image_utils.py
│
└── docs/
    └── screenshots/
```

---

## Option A: Google Colab (Recommended)

**Best for:** First-time use, quick experiments, no account setup needed beyond Google.

### Step-by-Step

**1. Open the notebook**
- Go to [colab.research.google.com](https://colab.research.google.com)
- File → Upload notebook → select `notebooks/medgemma_chest_xray_colab.ipynb`

**2. Enable T4 GPU**
- Runtime → Change runtime type
- Hardware accelerator → **T4 GPU**
- Click Save

**3. Get a HuggingFace token**
- Create a free account at [huggingface.co](https://huggingface.co)
- Profile → Settings → Access Tokens → **New token** (role: Read)
- Copy the token (starts with `hf_...`)

**4. Accept the MedGemma license**
- Visit [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- Click **Agree and access repository**

**5. Add your token to Colab Secrets**
- Click the 🔑 key icon in the left sidebar
- Name: `HF_TOKEN` | Value: `hf_your_token`
- Toggle **Notebook access** ON

**6. Run all cells**
- Runtime → **Run all** (Ctrl+F9)
- Cell 0 auto-checks for CUDA version mismatches and reinstalls torchvision if needed — **if it restarts the runtime, run all cells again**
- First run downloads the model (~8 GB) — takes 2–4 minutes
- Inference takes ~10 seconds per image

**7. Analyze your own image**
- In the last cell, upload your image via the file picker
- Set `IMAGE_TYPE = 'auto'` (or specify the modality manually)
- Run the cell — the model will generate a report structured for your image type

---

## Option B: Kaggle

**Best for:** Working with your own datasets, persistent storage, more GPU hours per week.

### Step-by-Step

**1. Create a Kaggle account**
- Sign up at [kaggle.com](https://kaggle.com)
- Go to **Settings → Phone Verification** and verify your phone number
  *(Required to unlock GPU access — without this, GPU option is grayed out)*

**2. Create a new notebook**
- Click **Create** → **New Notebook**
- Or upload: File → Import Notebook → select `notebooks/medgemma_chest_xray_kaggle.ipynb`

**3. Enable T4 GPU**
- Right panel → **Session options** → Accelerator → **GPU T4 x1**
  > ⚠️ Do NOT use P100 — it uses CUDA sm_60 which is incompatible with PyTorch 2.6+

**4. Add your HuggingFace token**
- Top menu → **Add-ons → Secrets**
- Click **+ Add a New Secret**
- Name: `HF_TOKEN` | Value: `hf_your_token`
- Toggle **Notebook access** ON (turns blue)

**5. Accept the MedGemma license**
- Visit [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- Click **Agree and access repository**

**6. Run all cells**
- Run → **Run All**
- First run downloads the model (~8 GB) — takes 2–4 minutes
- Inference takes ~10 seconds per image

**7. Upload and analyze your own image**
- Right panel → **Input → Upload → New Dataset**
- Drag and drop your image file, give the dataset a name (e.g. `my-xray`)
- Find the exact path by running this in a cell:
  ```python
  import os
  for root, dirs, files in os.walk('/kaggle/input'):
      for f in files:
          print(os.path.join(root, f))
  ```
- Update `IMAGE_PATH` and `IMAGE_TYPE` in the last cell:
  ```python
  IMAGE_PATH = '/kaggle/input/my-xray/brain_mri.png'
  IMAGE_TYPE  = 'mri_brain'   # or 'auto' to let MedGemma detect
  ```

---

## Analysis Types

The model supports multiple analysis modes selectable via the **Analysis Type** dropdown (Gradio UI) or by passing a `prompt_key` in the notebooks.

| Analysis Type | Dropdown Label | Behavior |
|--------------|----------------|----------|
| **Detailed Radiology Report** | 📋 Detailed Radiology Report | Full structured report with modality-specific sections |
| **General Description** | 🔍 General Description | Describe everything visible in the image |
| **Key Findings (Bullet Points)** | 📌 Key Findings (Bullet Points) | Concise bulleted list of findings |
| **Compare to Normal** | ⚖️ Compare to Normal | Highlights deviations from a healthy reference |
| **Patient-Friendly Explanation** | 💬 Patient-Friendly Explanation | Simple language for non-medical readers |
| **Custom Prompt** | *(type in the box)* | Any free-text instruction you provide |

> **Note:** When **Detailed Radiology Report** is selected, the report sections automatically match the image type — brain MRI generates ventricles/white matter sections, not lungs/heart sections.

### Custom Prompts for Different Modalities (Notebooks)

```python
# MRI brain tumor
analyze_xray(img, 'Analyze this brain MRI for any mass lesions. Describe location, signal intensity, surrounding edema, midline shift, and differential diagnosis.', image_type='mri_brain')

# Pathology slide — cancer grading
analyze_xray(img, 'Analyze this histopathology slide. Describe cell morphology, tissue architecture, and any features of malignancy. Provide grading if applicable.', image_type='pathology_slide')

# CT scan — lung nodule
analyze_xray(img, 'Analyze this chest CT slice for pulmonary nodules or masses. Describe size, location, margins, and likelihood of malignancy.', image_type='ct_scan')

# Mammogram — BI-RADS
analyze_xray(img, 'Analyze this mammogram. Describe breast density, any masses or calcifications, and provide a BI-RADS category assessment.', image_type='mammogram')

# Retinal fundus — diabetic retinopathy
analyze_xray(img, 'Analyze this fundus photograph for signs of diabetic retinopathy. Grade severity and describe microaneurysms, exudates, or neovascularization.', image_type='fundus')
```

---

## Sample Output

### Chest X-Ray (detailed report)

```
════════════════════════════════════════════════════════════════
  RADIOLOGY REPORT — MedGemma 1.5 (4B)
════════════════════════════════════════════════════════════════

LUNGS:
The lungs are well-inflated bilaterally. No focal consolidation,
pleural effusion, or pneumothorax is identified. The lung fields
are clear without evidence of infiltrates or nodular lesions.

PLEURA:
No pleural effusion or pneumothorax identified.

HEART:
The cardiac silhouette is within normal limits. The cardiothoracic
ratio is less than 50%. No cardiomegaly.

MEDIASTINUM:
The mediastinum is midline and of normal width. No mediastinal
widening or hilar adenopathy identified.

BONES:
The visualized osseous structures are intact. No acute fractures,
lytic lesions, or significant degenerative changes noted.

IMPRESSION:
No acute cardiopulmonary abnormality identified.

⚠️  For research purposes only. Not for clinical diagnosis.
════════════════════════════════════════════════════════════════
```

### Brain MRI (detailed report — different sections)

```
════════════════════════════════════════════════════════════════
  RADIOLOGY REPORT — MedGemma 1.5 (4B)
════════════════════════════════════════════════════════════════

BRAIN PARENCHYMA:
No focal signal abnormality identified in the cortical or
deep gray matter. Cerebellum is unremarkable.

VENTRICLES AND CSF SPACES:
Ventricles are normal in size and configuration. No hydrocephalus.

WHITE MATTER:
No significant white matter signal changes. No periventricular
or subcortical lesions.

POSTERIOR FOSSA:
Brainstem and cerebellum appear normal.

MIDLINE:
No midline shift or herniation identified.

IMPRESSION:
Normal brain MRI. No acute intracranial abnormality.

⚠️  For research purposes only. Not for clinical diagnosis.
════════════════════════════════════════════════════════════════
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |
| Inference | HuggingFace Transformers `pipeline("image-text-to-text")` |
| Quantization | BitsAndBytes 4-bit NF4 (applied inline at load time) |
| Web UI | Gradio 4.x / 5.x |
| Image Processing | Pillow |
| GPU Acceleration | PyTorch 2.6+ with CUDA |
| Secrets Management | `python-dotenv` (local) / Colab Secrets / Kaggle Secrets |

---

## Changelog

### v2.0 — Multi-Modality Support + Bug Fixes

**New features**
- Added support for **10 imaging modalities**: chest X-ray, CT scan, brain MRI, other MRI, mammogram, skin lesion, pathology slide, retinal fundus, ultrasound, bone X-ray
- Added **Auto-detect** mode — runs a 30-token classification inference before the main analysis to identify the modality, then applies the correct specialist prompt automatically
- Added per-modality **specialist system prompts** in `configs/model_config.py` (`IMAGE_TYPE_SYSTEM_PROMPTS`) — prevents brain MRI from being analyzed as a chest X-ray
- Added per-modality **detailed report prompts** (`IMAGE_TYPE_DETAILED_PROMPTS`) — each modality generates report sections relevant to that anatomy (e.g., ventricles/white matter for brain MRI instead of lungs/pleura)
- Added `IMAGE_TYPE` variable to notebooks for easy modality selection
- Gradio UI: new **Image Type** dropdown with Auto-detect as the default

**Bug fixes**
- Fixed **CUDA version mismatch** (`PyTorch CUDA 13.0` vs `torchvision CUDA 12.8`) — Cell 0 in the Colab notebook now auto-detects the mismatch, reinstalls `torchvision` with the correct CUDA index URL, and triggers a runtime restart
- Fixed `dtype=torch.bfloat16` → `torch_dtype=torch.bfloat16` (wrong parameter name in `pipeline()`) in both Colab and Kaggle notebooks
- Fixed **system message format** bug in `src/analyzer.py` — system content must be a list `[{"type": "text", "text": ...}]`, not a bare string, to match the MedGemma message schema
- Fixed `SyntaxError: unterminated f-string literal` caused by `\n` inside f-string literals stored as literal characters in the Colab notebook JSON
- Changed `quantized` mode from unverified third-party upload to the **official `google/medgemma-1.5-4b-it`** model with **BitsAndBytes 4-bit NF4 quantization** applied inline at load time
- Fixed `TypeError: analyze_xray() got an unexpected keyword argument 'system_prompt'` — renamed `system_role` → `system_prompt` in the inference cell
- Fixed `NameError: name 'SYSTEM_PROMPT_RADIOLOGIST' is not defined` — updated cell to use `SYSTEM_PROMPT` (the variable defined in the session) instead of the renamed constant

---

## References

- [MedGemma 1.5 — HuggingFace Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma 1.5 — Google Developers Model Card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [MedGemma 1.5 Technical Report — arXiv](https://arxiv.org/html/2507.05201v3)
- [Next Generation Medical Image Interpretation with MedGemma 1.5 — Google Research Blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- [MedGemma — Google DeepMind](https://deepmind.google/models/gemma/medgemma/)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Sample X-Ray: Wikimedia Commons, CC0](https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png)

---

## Disclaimer

> ⚠️ **This tool is strictly for research and educational purposes.**
>
> MedGemma is an AI model and its outputs are **not validated for clinical use**. Do not use this tool for actual medical diagnosis, treatment planning, or any clinical decision-making. Always consult a licensed radiologist or physician for interpretation of medical imaging.
