"""
Core MedGemma inference engine for chest X-ray analysis.
"""

import os
import sys
import time
from typing import Optional

import torch
from PIL import Image

# Resolve project root so we can import configs regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs.model_config import (
    DISCLAIMER,
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    PROMPTS,
    SYSTEM_PROMPT_RADIOLOGIST,
    SYSTEM_PROMPT_SIMPLE,
)

VALID_MODES = ("pipeline", "quantized", "full")


class ChestXRayAnalyzer:
    """MedGemma-powered chest X-ray analysis engine.

    Modes
    -----
    pipeline  : HuggingFace pipeline API with bfloat16 (requires ~14 GB VRAM)
    quantized : Pre-quantized 4-bit model via unsloth (~6 GB VRAM)
    full      : AutoModel + BitsAndBytesConfig 4-bit quant (~6 GB VRAM)
    """

    def __init__(
        self,
        mode: str = "quantized",
        hf_token: Optional[str] = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: {', '.join(VALID_MODES)}"
            )
        self.mode = mode
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model = None
        self.processor = None
        self._loaded = False

        if self.device == "cpu":
            print(
                "⚠️  WARNING: No CUDA GPU detected. Running on CPU will be very slow "
                "and may run out of memory. A GPU with ≥6 GB VRAM is recommended."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Download and initialize the MedGemma model."""
        if self._loaded:
            return

        print(f"🔄 Loading MedGemma model in '{self.mode}' mode on {self.device}...")
        t0 = time.time()

        if self.mode in ("pipeline", "quantized"):
            self._load_pipeline_mode()
        else:
            self._load_full_mode()

        elapsed = time.time() - t0
        print(f"✅ Model loaded in {elapsed:.1f}s")
        self._loaded = True

    def _load_pipeline_mode(self) -> None:
        from transformers import BitsAndBytesConfig, pipeline as hf_pipeline

        if self.mode == "pipeline":
            self.pipe = hf_pipeline(
                "image-text-to-text",
                model=MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.hf_token,
            )
            print(f"   Model: {MODEL_ID}")
        else:  # quantized — official model loaded with 4-bit BnB via pipeline
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.pipe = hf_pipeline(
                "image-text-to-text",
                model=MODEL_ID,
                model_kwargs={"quantization_config": bnb_config},
                device_map="auto",
                token=self.hf_token,
            )
            print(f"   Model: {MODEL_ID} (4-bit NF4 via BitsAndBytes)")

    def _load_full_mode(self) -> None:
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.hf_token,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=self.hf_token,
        )
        print(f"   Model: {MODEL_ID} (full + 4-bit BnB quantization)")

    # ──────────────────────────────────────────────────────────────────────────
    # Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        image: Image.Image,
        prompt_key: str = "detailed_report",
        custom_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        """Run inference on a medical image.

        Args:
            image: PIL Image (RGB).
            prompt_key: Key from PROMPTS dict (ignored if custom_prompt given).
            custom_prompt: Override prompt text.
            system_prompt: Override system prompt (defaults to radiologist prompt).
            max_new_tokens: Cap output length. Defaults to MAX_NEW_TOKENS (1024).
                            Pass a small value (e.g. 30) for classification calls
                            where only a short label is needed.

        Returns:
            dict with keys: report, prompt_used, system_prompt_used, model.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() before analyze()."
            )

        # Use caller-supplied token limit or fall back to the config default
        n_tokens = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

        # Resolve prompts
        user_prompt = custom_prompt or PROMPTS.get(
            prompt_key, PROMPTS["detailed_report"]
        )
        sys_prompt = system_prompt or (
            SYSTEM_PROMPT_SIMPLE
            if prompt_key == "simple"
            else SYSTEM_PROMPT_RADIOLOGIST
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        if self.mode in ("pipeline", "quantized"):
            report = self._infer_pipeline(messages, n_tokens)
            model_used = MODEL_ID if self.mode == "pipeline" else f"{MODEL_ID} [4-bit]"
        else:
            report = self._infer_full(messages, n_tokens)
            model_used = MODEL_ID

        return {
            "report": report,
            "prompt_used": user_prompt,
            "system_prompt_used": sys_prompt,
            "model": model_used,
        }

    def _infer_pipeline(self, messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        output = self.pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
        )
        return output[0]["generated_text"][-1]["content"]

    def _infer_full(self, messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=DO_SAMPLE,
            )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[:, input_len:]
        return self.processor.decode(new_tokens[0], skip_special_tokens=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"ChestXRayAnalyzer(mode='{self.mode}', device='{self.device}', {status})"
