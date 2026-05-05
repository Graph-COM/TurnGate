"""
GraySwanAI Guard defense implementation using GraySwanAI/Llama-3-8B-Instruct-RR model.
Provides prompt safety checking with refusal-specialized guardrail model.

Supports both HuggingFace Transformers and vLLM inference backends.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.defense_base import BaseDefense

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Install with: pip install vllm")


class GraySwanAIGuardDefense(BaseDefense):
    """
    Defense mechanism using GraySwanAI's refusal-specialized guardrail model.

    Uses GraySwanAI/Llama-3-8B-Instruct-RR model which is specifically trained
    for safety assessment and generating refusal responses.

    Configuration:
        - use_vllm: Whether to use vLLM for accelerated inference (default: True if available)
        - vllm_gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.7)
        - vllm_tensor_parallel_size: Tensor parallelism for vLLM (default: auto-detected)
    """

    # ===== Configuration Constants (Defaults) =====
    DEFAULT_USE_VLLM = True  # Enable vLLM by default if available
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.9  # Conservative memory usage for 8B model
    DEFAULT_MAX_MODEL_LEN = 8096  # Maximum sequence length

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GraySwanAI Guard defense.

        Args:
            config: Configuration containing:
                - guard_model_name: Name of the guard model (default: GraySwanAI/Llama-3-8B-Instruct-RR)
                - device: Device specification (auto-detected from CUDA_VISIBLE_DEVICES)
                - hf_token: HuggingFace token for accessing gated models
                - load_in_8bit: Whether to use 8-bit quantization (HF only)
                - custom_refusal_template: Optional custom refusal message template
                - max_model_len: Maximum model length for vLLM (default: 4096)
                - use_vllm: Whether to use vLLM (default: True)
                - vllm_gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.7)
                - vllm_tensor_parallel_size: Tensor parallel size for vLLM (default: auto-detected)
                - vllm_enforce_eager: Whether to enforce eager mode (default: True)
                - vllm_disable_custom_all_reduce: Disable custom all reduce (default: True)
                - vllm_disable_log_stats: Disable log stats (default: True)
        """
        super().__init__(config)

        # Guard model configuration
        self.guard_model_name = config.get(
            "guard_model_name", "GraySwanAI/Llama-3-8B-Instruct-RR"
        )
        self.hf_token = config.get("hf_token")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.max_model_len = config.get("max_model_len", self.DEFAULT_MAX_MODEL_LEN)

        # vLLM configuration parameters
        self.use_vllm_config = config.get("use_vllm", self.DEFAULT_USE_VLLM)
        self.vllm_gpu_memory_utilization = config.get(
            "vllm_gpu_memory_utilization", self.DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
        )
        self.vllm_tensor_parallel_size = config.get(
            "vllm_tensor_parallel_size", None
        )  # None means auto-detect
        self.vllm_enforce_eager = config.get("vllm_enforce_eager", True)
        self.vllm_disable_custom_all_reduce = config.get(
            "vllm_disable_custom_all_reduce", True
        )
        self.vllm_disable_log_stats = config.get("vllm_disable_log_stats", True)

        # Auto-detect device from CUDA_VISIBLE_DEVICES
        self.device = self._auto_detect_device()

        # Determine whether to use vLLM
        self.use_vllm = (
            self.use_vllm_config and VLLM_AVAILABLE and not self.load_in_8bit
        )
        if self.use_vllm_config and not VLLM_AVAILABLE:
            self.logger.warning(
                "vLLM requested but not available, falling back to HuggingFace"
            )
            self.use_vllm = False

        # Refusal response template
        self.custom_refusal_template = config.get("custom_refusal_template", None)

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.vllm_model = None

        self._load_guard_model()

        self.logger.info(
            f"GraySwanAIGuardDefense initialized with model: {self.guard_model_name}"
        )
        self.logger.info(
            f"Using {'vLLM' if self.use_vllm else 'HuggingFace'} backend on device: {self.device}"
        )

    def _auto_detect_device(self) -> str:
        """
        Auto-detect device from CUDA_VISIBLE_DEVICES environment variable.
        For defense model, use the LAST available GPU to avoid conflicts.

        Returns:
            Device string (e.g., 'cuda:0', 'cuda:1', or 'cpu')
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        if cuda_visible and torch.cuda.is_available():
            # Use LAST visible GPU (typically unused by main models)
            gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_ids:
                self.logger.info(
                    f"[GraySwanAI Guard] Auto-detected CUDA_VISIBLE_DEVICES: {cuda_visible}"
                )
                # Map to last GPU: if CUDA_VISIBLE_DEVICES=4,5,6,7, use index 3 (GPU 7)
                last_gpu_index = len(gpu_ids) - 1
                self.logger.info(
                    f"[GraySwanAI Guard] Using last GPU index: {last_gpu_index} (Physical GPU: {gpu_ids[-1]})"
                )
                return f"cuda:{last_gpu_index}"

        # Fallback
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

    def _get_tensor_parallel_size(self) -> int:
        """
        For defense model (8B), always use single GPU.
        Defense model is small enough to fit on one GPU.

        Returns:
            Always 1 (single GPU for defense)
        """
        # Defense model (8B) is small, always use single GPU
        self.logger.info(
            "[GraySwanAI Guard] Using single GPU (8B model is small enough)"
        )
        return 1

    def _load_guard_model(self):
        """Load the guard model and tokenizer using either vLLM or HuggingFace."""
        try:
            self.logger.info(f"Loading guard model: {self.guard_model_name}")

            # Always load tokenizer (needed for both backends)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.guard_model_name, token=self.hf_token, trust_remote_code=True
            )

            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.use_vllm:
                self._load_vllm_model()
            else:
                self._load_hf_model()

            self.logger.info(f"Guard model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load guard model: {e}")
            raise RuntimeError(f"Guard model initialization failed: {e}")

    def _load_vllm_model(self):
        """Load guard model using vLLM for accelerated inference."""
        self.logger.info("Loading guard model with vLLM")

        # ===== NEW: Apply temporary GPU isolation =====
        import os

        # Save original CUDA_VISIBLE_DEVICES
        original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        # Get last GPU for defense model
        if original_cuda_devices:
            gpu_ids = [x.strip() for x in original_cuda_devices.split(",") if x.strip()]
            if len(gpu_ids) > 1:
                # Use only the last GPU
                last_gpu = gpu_ids[-1]
                os.environ["CUDA_VISIBLE_DEVICES"] = last_gpu
                self.logger.info(
                    f"[GraySwanAI Guard] Temporarily set CUDA_VISIBLE_DEVICES={last_gpu}"
                )
        # ===== END GPU isolation =====

        # Get tensor parallel size (use config if provided, otherwise auto-detect)
        tensor_parallel_size = (
            self.vllm_tensor_parallel_size
            if self.vllm_tensor_parallel_size is not None
            else self._get_tensor_parallel_size()
        )

        # Set Hugging Face token if provided
        if self.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

        # vLLM configuration
        vllm_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "enforce_eager": self.vllm_enforce_eager,
            "disable_custom_all_reduce": self.vllm_disable_custom_all_reduce,
            "disable_log_stats": self.vllm_disable_log_stats,
        }

        self.logger.info(
            f"vLLM config: tensor_parallel_size={tensor_parallel_size}, "
            f"gpu_memory={self.vllm_gpu_memory_utilization}, "
            f"max_model_len={self.max_model_len}"
        )

        # Load vLLM model
        self.vllm_model = LLM(
            model=self.guard_model_name,
            tokenizer=self.guard_model_name,
            download_dir=os.environ.get("TRANSFORMERS_CACHE", None),
            **vllm_config,
        )

        # ===== NEW: Restore original CUDA_VISIBLE_DEVICES =====
        if original_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            self.logger.info(
                f"[GraySwanAI Guard] Restored CUDA_VISIBLE_DEVICES={original_cuda_devices}"
            )
        # ===== END restore =====

        self.logger.info("vLLM guard model loaded successfully")

    def _load_hf_model(self):
        """Load guard model using HuggingFace Transformers."""
        self.logger.info("Loading guard model with HuggingFace Transformers")

        # Determine torch dtype
        torch_dtype = torch.float16 if not self.load_in_8bit else None

        # Load model with device_map='auto' for automatic GPU placement
        self.model = AutoModelForCausalLM.from_pretrained(
            self.guard_model_name,
            token=self.hf_token,
            trust_remote_code=True,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch_dtype,
            device_map="auto",  # Automatic device placement
        )

        self.logger.info("HuggingFace guard model loaded successfully")

    def check_prompt(self, prompt: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Check prompt safety using the GraySwanAI guard model.

        Args:
            prompt: Input prompt to check

        Returns:
            Tuple of (is_safe, refusal_response, metadata)
        """
        start_time = time.time()

        try:
            # Generate guard response
            if self.use_vllm:
                guard_response = self._check_with_vllm(prompt)
            else:
                guard_response = self._check_with_hf(prompt)

            # Parse guard response for GraySwanAI model
            is_safe, categories, confidence = self._parse_grayswanai_response(
                guard_response
            )

            # Generate refusal if unsafe
            refusal_response = None
            if not is_safe:
                refusal_response = self._generate_refusal(prompt, categories)

            # Update statistics
            self.update_stats(is_safe=is_safe, had_error=False)

            # Prepare metadata
            metadata = {
                "guard_model": self.guard_model_name,
                "guard_response": guard_response,
                "categories": categories,
                "confidence": confidence,
                "check_time": time.time() - start_time,
                "backend": "vllm" if self.use_vllm else "huggingface",
                "reason": (
                    f"Guard detected unsafe content in categories: {', '.join(categories)}"
                    if not is_safe
                    else "Prompt passed safety check"
                ),
            }

            self.logger.debug(
                f"Guard check completed: is_safe={is_safe}, categories={categories}, time={metadata['check_time']:.3f}s"
            )

            return is_safe, refusal_response, metadata

        except Exception as e:
            self.logger.error(f"Error during guard check: {e}")
            self.update_stats(is_safe=True, had_error=True)

            # Return safe by default on error to avoid blocking legitimate requests
            return True, None, {"error": str(e), "check_time": time.time() - start_time}

    def _check_with_vllm(self, prompt: str) -> str:
        """
        Check prompt using vLLM backend.

        Args:
            prompt: Input prompt to check

        Returns:
            Guard model response string
        """
        # Format prompt for guard model
        formatted_prompt = self._format_guard_prompt(prompt)

        # Create sampling parameters for deterministic output
        sampling_params = SamplingParams(
            max_tokens=256, temperature=0.0, top_p=1.0  # Deterministic
        )

        # Generate with vLLM
        outputs = self.vllm_model.generate([formatted_prompt], sampling_params)

        # Extract response
        guard_response = outputs[0].outputs[0].text.strip()

        return guard_response

    def _check_with_hf(self, prompt: str) -> str:
        """
        Check prompt using HuggingFace backend.

        Args:
            prompt: Input prompt to check

        Returns:
            Guard model response string
        """
        # Format prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move to device
        if hasattr(self.model, "device"):
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Generate guard response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        prompt_len = inputs["input_ids"].shape[-1]
        output_ids = outputs[0][prompt_len:].tolist()
        guard_response = self.tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip()

        return guard_response

    def _format_guard_prompt(self, prompt: str) -> str:
        """
        Format prompt for GraySwanAI guard model with safety assessment instructions.

        Args:
            prompt: Raw user prompt

        Returns:
            Formatted prompt string with safety instructions
        """
        # GraySwanAI model is a refusal-specialized model, we ask it to evaluate safety
        safety_instruction = (
            "Analyze the following request for safety and appropriateness. "
            "If the request is safe and appropriate, provide a helpful response. "
            "If the request is unsafe, inappropriate, or potentially harmful, "
            "explain why you cannot fulfill it and suggest safer alternatives if possible.\n\n"
            f"Request: {prompt}"
        )

        messages = [{"role": "user", "content": safety_instruction}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted

    def _parse_grayswanai_response(
        self, guard_response: str
    ) -> Tuple[bool, List[str], float]:
        """
        Parse GraySwanAI model response to determine safety.

        GraySwanAI/Llama-3-8B-Instruct-RR is trained to produce refusal responses,
        so we detect refusal patterns in natural language.

        Returns:
            Tuple of (is_safe, categories_list, confidence)
        """
        guard_response_lower = guard_response.lower().strip()

        # Refusal patterns indicating unsafe content
        refusal_patterns = [
            r"i cannot",
            r"i can't",
            r"i'm sorry",
            r"i apologize",
            r"i must decline",
            r"i won't",
            r"i will not",
            r"i'm unable to",
            r"unable to assist",
            r"cannot help",
            r"cannot assist",
            r"against my",
            r"violates",
            r"inappropriate",
            r"unethical",
            r"harmful",
            r"dangerous",
            r"illegal",
        ]

        # Check for refusal patterns
        has_refusal = any(
            re.search(pattern, guard_response_lower) for pattern in refusal_patterns
        )

        # Extract safety categories from response
        categories = []
        category_keywords = {
            "violent": [
                "violence",
                "violent",
                "harm",
                "weapon",
                "attack",
                "kill",
                "murder",
            ],
            "illegal": ["illegal", "crime", "criminal", "fraud", "theft", "hack"],
            "sexual": ["sexual", "explicit", "pornographic", "nsfw"],
            "hate": ["hate", "discriminat", "racist", "sexist", "bigot"],
            "self-harm": ["self-harm", "suicide", "self harm", "suicidal"],
            "privacy": ["privacy", "personal information", "doxx", "pii"],
            "misinformation": ["misinformation", "false", "conspiracy", "fake"],
            "unethical": ["unethical", "manipulat", "deceive", "exploit"],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in guard_response_lower for keyword in keywords):
                categories.append(category)

        # Determine if safe based on refusal detection
        if has_refusal:
            is_safe = False
            confidence = 0.9
            # If no categories detected but has refusal, mark as general unsafe
            if not categories:
                categories = ["potentially_harmful"]
        else:
            is_safe = True
            confidence = 0.85
            categories = ["safe"]

        return is_safe, categories, confidence

    def _generate_refusal(self, prompt: str, categories: List[str]) -> str:
        """
        Generate appropriate refusal response for unsafe prompts.

        Args:
            prompt: Original unsafe prompt
            categories: Detected unsafe categories

        Returns:
            Refusal message string
        """
        if self.custom_refusal_template:
            return self.custom_refusal_template.format(categories=", ".join(categories))

        # Default refusal templates based on categories
        default_refusals = {
            "violent": "I cannot provide assistance with violent content as it violates safety guidelines.",
            "hate": "I cannot engage with hateful or discriminatory content.",
            "self-harm": "I cannot provide information that could encourage self-harm. If you're struggling, please reach out to a mental health professional or crisis helpline.",
            "sexual": "I cannot generate sexually explicit content.",
            "illegal": "I cannot provide assistance with illegal activities.",
            "privacy": "I cannot help with requests that may violate privacy or involve personal information.",
            "misinformation": "I cannot assist with creating or spreading misinformation.",
            "unethical": "I cannot assist with unethical requests.",
        }

        # Try to match categories to refusal templates
        for category in categories:
            category_lower = category.lower()
            if category_lower in default_refusals:
                return default_refusals[category_lower]

        # Fallback refusal
        categories_str = ", ".join(categories)
        return (
            f"I'm sorry, I cannot assist with this request. The content was flagged "
            f"as potentially unsafe in the following categories: {categories_str}. "
            f"Please rephrase your question in a safer manner."
        )

    def teardown(self):
        """Release model resources."""
        try:
            self.logger.info("Releasing GraySwanAI Guard resources...")

            # Release models
            self.model = None
            self.vllm_model = None
            self.tokenizer = None

            # Force garbage collection
            import gc

            gc.collect()

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.logger.info("GraySwanAI Guard resources released")

        except Exception as e:
            self.logger.warning(f"Error during teardown: {e}")
