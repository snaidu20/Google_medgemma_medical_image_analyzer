"""
Central configuration for MedGemma Chest X-Ray Analyzer.
"""

# ─── Model IDs ────────────────────────────────────────────────────────────────
MODEL_ID = "google/medgemma-1.5-4b-it"
# MODEL_ID_QUANTIZED is kept as a reference only.
# The 'quantized' mode in analyzer.py now applies 4-bit BnB quantization
# directly on MODEL_ID instead of relying on this third-party upload.
MODEL_ID_QUANTIZED = "google/medgemma-1.5-4b-it"  # same base; quant applied at load time

# ─── Generation Settings ──────────────────────────────────────────────────────
MAX_NEW_TOKENS = 1024
DO_SAMPLE = False  # Greedy decoding (Jan 2026 recommended)

# ─── Legacy System Prompts (kept for backward compatibility) ──────────────────
# These are chest-X-ray-specific prompts used when no image type is selected.
# Prefer IMAGE_TYPE_SYSTEM_PROMPTS[key] for new code paths.
SYSTEM_PROMPT_RADIOLOGIST = (
    "You are an expert radiologist with over 20 years of clinical experience in "
    "interpreting chest X-rays. Analyze the provided chest X-ray image and generate "
    "a comprehensive, structured radiology report. Your report must systematically "
    "cover the following anatomical regions and findings:\n\n"
    "1. **Lungs**: Evaluate lung fields bilaterally — look for consolidation, "
    "infiltrates, nodules, masses, hyperinflation, atelectasis, or interstitial patterns.\n"
    "2. **Pleura**: Assess for pleural effusion (free or loculated), pneumothorax, "
    "pleural thickening, or calcification.\n"
    "3. **Heart**: Evaluate cardiac size (cardiothoracic ratio), contour, and position. "
    "Note cardiomegaly, pericardial effusion, or chamber enlargement if visible.\n"
    "4. **Mediastinum**: Assess mediastinal width, contour, and position. Note "
    "tracheal deviation, widening, or hilar abnormalities.\n"
    "5. **Bones**: Review visible ribs, clavicles, scapulae, and spine for fractures, "
    "lytic lesions, sclerotic changes, or degenerative disease.\n\n"
    "Conclude with an **Impression** summarizing the most clinically significant findings "
    "and recommended next steps if applicable.\n\n"
    "Use professional medical terminology. If a region appears normal, explicitly state "
    "'No acute abnormality identified' for that region."
)

# Generic patient-friendly prompt — works across all imaging modalities.
SYSTEM_PROMPT_SIMPLE = (
    "You are a compassionate medical professional explaining medical imaging findings "
    "to a patient with no medical background. Use clear, simple, everyday language — "
    "avoid jargon. Explain what was seen in the image in a friendly and reassuring tone, "
    "what it might mean, and suggest they follow up with their doctor for any concerns. "
    "Keep the explanation concise, warm, and easy to understand."
)

# ─── Generic Analysis Prompts ─────────────────────────────────────────────────
# These prompts work across all imaging modalities.
# IMPORTANT: "detailed_report" is overridden per image type by
# IMAGE_TYPE_DETAILED_PROMPTS — the generic version below is a fallback only.
PROMPTS = {
    "detailed_report": "Generate a detailed structured report for this medical image.",
    "describe":        "Describe what you see in this medical image.",
    "findings_only":   "List the key findings in this medical image as bullet points.",
    "compare":         "Compare this image to a normal/healthy reference and highlight any abnormalities.",
    "simple":          "Explain what you see in this image in simple terms a patient could understand.",
}

# ─── Prompt Labels (for UI dropdowns) ─────────────────────────────────────────
PROMPT_LABELS = {
    "detailed_report": "📋 Detailed Radiology Report",
    "describe":        "🔍 General Description",
    "findings_only":   "📌 Key Findings (Bullet Points)",
    "compare":         "⚖️ Compare to Normal",
    "simple":          "💬 Patient-Friendly Explanation",
}

# ─── Auto-detection prompt ────────────────────────────────────────────────────
# Used when the user selects "Auto-detect".  Sent as a short classification
# inference BEFORE the main analysis so the model can identify the modality.
# We limit the response to 30 tokens — we only need one label word back.
AUTO_DETECT_SYSTEM_PROMPT = (
    "You are a medical imaging classifier. "
    "Look at the image and identify the imaging modality. "
    "Reply with exactly one label from the provided list — nothing else."
)

AUTO_DETECT_PROMPT = (
    "What type of medical image is this? "
    "Reply with exactly one label from this list: "
    "chest_xray, ct_scan, mri_brain, mri_other, mammogram, "
    "skin_lesion, pathology_slide, fundus, ultrasound, bone_xray. "
    "Output only the label, no punctuation, no explanation."
)

# ─── Image Type Labels (for UI dropdowns) ─────────────────────────────────────
# Maps the internal image_type key to the human-readable label shown in the UI.
# Add new modalities here and mirror them in IMAGE_TYPE_SYSTEM_PROMPTS and
# IMAGE_TYPE_DETAILED_PROMPTS below.
# "auto" is always first so it is the default selection in the dropdown.
IMAGE_TYPE_LABELS = {
    "auto":            "🔍 Auto-detect (let MedGemma identify)",
    "chest_xray":      "🫁 Chest X-Ray",
    "ct_scan":         "🔬 CT Scan",
    "mri_brain":       "🧠 Brain MRI",
    "mri_other":       "🧲 Other MRI",
    "mammogram":       "🔮 Mammogram",
    "skin_lesion":     "🩺 Skin Lesion",
    "pathology_slide": "🔬 Pathology Slide",
    "fundus":          "👁️ Retinal Fundus",
    "ultrasound":      "〰️ Ultrasound",
    "bone_xray":       "🦴 Bone X-Ray",
}

# ─── Per-Modality System Prompts ─────────────────────────────────────────────
# Each entry tells MedGemma what specialist role to play and what anatomical
# structures to focus on.  Selecting the right modality here is the most
# important step — it prevents the model from reporting on lungs when looking
# at a brain MRI, for example.
IMAGE_TYPE_SYSTEM_PROMPTS = {

    # Chest radiologist: covers lungs/pleura/heart/mediastinum/bones
    "chest_xray": (
        "You are an expert radiologist with 20 years of experience in chest imaging. "
        "Analyze the provided chest X-ray and generate a comprehensive structured report "
        "covering: Lungs, Pleura, Heart, Mediastinum, Bones, and Impression. "
        "Use professional medical terminology. State 'No acute abnormality identified' "
        "for any region that appears normal."
    ),

    # CT specialist: focus on density, Hounsfield units, slice-by-slice findings
    "ct_scan": (
        "You are an expert radiologist specializing in CT interpretation with 20 years "
        "of experience. Analyze this CT scan slice systematically. Describe visible "
        "structures, density characteristics, any masses, nodules, lesions, or "
        "abnormalities, and provide a clinical impression with differential diagnosis."
    ),

    # Neuroradiologist: brain parenchyma, ventricles, white matter, midline shift
    "mri_brain": (
        "You are an expert neuroradiologist with 20 years of experience interpreting "
        "brain MRI studies. Analyze this brain MRI systematically. Evaluate: "
        "brain parenchyma signal, ventricle size and symmetry, sulci/gyri pattern, "
        "white matter integrity, posterior fossa structures, any masses or signal "
        "abnormalities, midline shift, and vascular territory involvement. "
        "Use standard MRI terminology (T1/T2 signal, FLAIR, DWI as applicable). "
        "Provide a structured impression with differential diagnosis."
    ),

    # MRI generalist: works for spine, abdomen, musculoskeletal, pelvic MRI
    "mri_other": (
        "You are an expert radiologist specializing in MRI interpretation with 20 years "
        "of experience. Analyze this MRI image. Describe the anatomical structures "
        "visible, their signal intensities on the apparent pulse sequence, any abnormal "
        "signal regions, mass effect, and provide a clinical impression."
    ),

    # Breast radiologist: BI-RADS framework, density, masses, calcifications
    "mammogram": (
        "You are an expert breast radiologist with 20 years of mammography experience. "
        "Analyze this mammogram using BI-RADS criteria. Describe: breast composition "
        "category, any masses (shape, margin, density), calcifications (morphology, "
        "distribution), architectural distortion, or asymmetries. "
        "Assign a BI-RADS category (0–6) and provide the appropriate management recommendation."
    ),

    # Dermatologist/dermoscopist: ABCDE criteria, pigmentation, lesion morphology
    "skin_lesion": (
        "You are an expert dermatologist and dermoscopist with 20 years of clinical "
        "experience. Evaluate this skin lesion using the ABCDE dermoscopy criteria: "
        "Asymmetry, Border irregularity, Color variation, Diameter, and Evolution features. "
        "Describe lesion morphology, surface texture, and pigmentation patterns. "
        "Provide a differential diagnosis ranked by clinical likelihood."
    ),

    # Pathologist: cell morphology, architecture, staining, malignancy features
    "pathology_slide": (
        "You are an expert pathologist with 20 years of experience in histopathology "
        "and oncologic pathology. Analyze this histopathology slide. Describe: "
        "cell morphology, nuclear-to-cytoplasmic ratio, mitotic activity, tissue "
        "architecture, staining characteristics, and any features suggestive of "
        "malignancy, dysplasia, inflammation, or other pathology. "
        "Provide a diagnostic impression."
    ),

    # Ophthalmologist: optic disc, macula, vessels, retinal pathology
    "fundus": (
        "You are an expert ophthalmologist specializing in retinal imaging with "
        "20 years of experience. Analyze this fundus photograph. Systematically "
        "evaluate: optic disc (cup-to-disc ratio, margins, color), macula (foveal "
        "reflex, drusen, exudates, pigment changes), retinal vasculature (AV ratio, "
        "caliber, crossing changes), peripheral retina, and any hemorrhages, "
        "neovascularization, or detachment."
    ),

    # Ultrasound specialist: echogenicity, organ morphology, cysts, vascularity
    "ultrasound": (
        "You are an expert radiologist specializing in diagnostic ultrasound with "
        "20 years of experience. Analyze this ultrasound image. Identify the organ(s) "
        "in view, describe echogenicity (anechoic, hypoechoic, isoechoic, hyperechoic), "
        "texture, borders, any cystic or solid masses, fluid collections, and vascular "
        "findings if Doppler is present. Provide a clinical impression."
    ),

    # MSK radiologist: cortex, trabecular bone, joint spaces, fractures, lesions
    "bone_xray": (
        "You are an expert musculoskeletal radiologist with 20 years of experience. "
        "Analyze this bone or joint X-ray. Evaluate: cortical integrity, bone density "
        "(osteopenia/sclerosis), trabecular pattern, joint spaces and alignment, any "
        "fractures (location, type, displacement, angulation), lytic or blastic lesions, "
        "periosteal reaction, and soft tissue swelling. Provide a structured impression."
    ),
}

# ─── Per-Modality Detailed Report Prompts ────────────────────────────────────
# These override PROMPTS["detailed_report"] when a specific image type is
# selected in the UI or notebook.  Each prompt directs MedGemma to cover the
# sections that are clinically relevant for that modality — this prevents the
# model from structuring a brain MRI report the same way as a chest X-ray.
IMAGE_TYPE_DETAILED_PROMPTS = {

    "chest_xray": (
        "Generate a detailed radiology report for this chest X-ray covering: "
        "1. Lungs  2. Pleura  3. Heart  4. Mediastinum  5. Bones  6. Impression."
    ),

    "ct_scan": (
        "Analyze this CT scan slice. Describe all visible structures, attenuation "
        "values, any masses, nodules, lesions, or abnormalities. "
        "Provide a clinical impression with differential diagnosis."
    ),

    # Brain MRI: explicitly requests the neuroimaging sections — NOT lungs/heart
    "mri_brain": (
        "Generate a detailed radiology report for this brain MRI covering: "
        "1. Brain parenchyma (cortex, deep gray matter, cerebellum)  "
        "2. Ventricles and CSF spaces  "
        "3. White matter signal  "
        "4. Posterior fossa (brainstem, cerebellum)  "
        "5. Any masses, lesions, or signal abnormalities  "
        "6. Midline shift or herniation  "
        "7. Impression and differential diagnosis."
    ),

    "mri_other": (
        "Analyze this MRI image. Describe anatomical structures, signal intensities, "
        "any abnormal signal regions or masses, and provide a clinical impression."
    ),

    "mammogram": (
        "Analyze this mammogram. Describe breast composition category, any masses "
        "(shape, margin, density), calcifications, architectural distortion, or "
        "asymmetries. Assign a BI-RADS category and management recommendation."
    ),

    "skin_lesion": (
        "Evaluate this skin lesion using ABCDE criteria (Asymmetry, Border, Color, "
        "Diameter, Evolution). Describe morphology, surface features, and pigmentation. "
        "Provide a ranked differential diagnosis."
    ),

    "pathology_slide": (
        "Analyze this histopathology slide. Describe cell morphology, "
        "nuclear features, tissue architecture, staining patterns, mitotic activity, "
        "and provide a diagnostic impression."
    ),

    "fundus": (
        "Analyze this fundus photograph. Describe the optic disc (CDR, margins), "
        "macula (foveal reflex, exudates, drusen), retinal vasculature (caliber, AV ratio), "
        "and any hemorrhages, neovascularization, or pigment changes."
    ),

    "ultrasound": (
        "Analyze this ultrasound image. Describe the visible organ(s), echogenicity, "
        "borders, any cystic or solid masses, fluid collections, and provide a "
        "clinical impression."
    ),

    "bone_xray": (
        "Analyze this bone X-ray. Describe cortical integrity, bone density, "
        "joint spaces, any fractures (type, displacement), lytic or sclerotic lesions, "
        "periosteal reaction, and provide a clinical impression."
    ),
}

# ─── Sample X-Ray URLs (Public Domain / CC0) ──────────────────────────────────
SAMPLE_XRAYS = {
    "Normal PA Chest X-Ray": (
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    ),
    "Chest X-Ray (Wikimedia)": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/"
        "Chest_X-ray_in_influenza_and_Haemophilus_influenzae.jpg/"
        "800px-Chest_X-ray_in_influenza_and_Haemophilus_influenzae.jpg"
    ),
}

# ─── Disclaimer ───────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️  For research purposes only. Not for clinical diagnosis. "
    "Always consult a qualified radiologist or physician for medical decisions."
)
