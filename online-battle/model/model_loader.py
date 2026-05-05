import os
import logging
import torch
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from defense.defense_factory import create_defense
from defense.defense_base import BaseDefense

# Import vLLM for accelerated inference
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Install with: pip install vllm")


class BaseModel:
    """Base class for all model types."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Common configuration
        self.max_length = config.get("max_length", 512)
        # Batch processing - separate dataset batch and vLLM parallel config
        self.batch_size = config.get("batch_size", 1)  # Dataset batch processing size

        # vLLM internal parallel configuration
        self.vllm_kwargs = config.get("vllm_kwargs", {})

        # Device configuration
        self.device = config.get("device", "cuda:0")
        self.device_map = config.get("device_map", None)

        # Model and tokenizer
        self.model = None
        self.tokenizer = None

        # vLLM configuration
        self.use_vllm = config.get("use_vllm", False)

        # Initialize vLLM model placeholder
        self.vllm_model = None  # vLLM model instance

        # Generation parameters (needed by both whitebox and blackbox models)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.do_sample = config.get("do_sample", True)

        # Defense mechanism initialization
        # Initialize defense mechanism if enabled in global config
        self.defense = None
        self._initialize_defense()

    def _initialize_defense(self):
        """
        Initialize defense mechanism based on global configuration.

        Defense is loaded from the global config passed during model initialization.
        If defense is enabled, creates appropriate defense instance that will
        pre-screen all prompts before they reach the target model.
        """
        try:
            # Check if defense configuration exists in config
            # Note: This relies on defense config being passed from ModelLoader
            defense_config = self.config.get("defense", {})

            if not defense_config:
                self.logger.debug("No defense configuration found")
                return

            enabled = defense_config.get("enabled", False)
            if not enabled:
                self.logger.info("Defense mechanism disabled by configuration")
                return

            defense_type = defense_config.get("type", "none")
            if defense_type == "none":
                self.logger.info("Defense type set to 'none', skipping initialization")
                return

            # Get defense-specific configuration
            defense_method_config = defense_config.get(defense_type, {})

            # Create defense instance
            self.defense = create_defense(defense_type, defense_method_config)

            if self.defense:
                self.logger.info(f"Defense mechanism initialized: {defense_type}")
            else:
                self.logger.warning(f"Failed to initialize defense: {defense_type}")

        except Exception as e:
            self.logger.error(f"Error initializing defense: {e}")
            self.defense = None  # Disable defense on error

    def _apply_defense_to_prompt(self, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Apply defense check/transformation to a single prompt.

        This helper method handles different defense behaviors:
        - Blocking defenses (LLM Guard): Return refusal if unsafe
        - Transforming defenses (Rephrasing, Perturbation): Return transformed prompt

        Args:
            prompt: Original prompt

        Returns:
            Tuple of (should_continue, prompt_or_refusal, metadata)
            - should_continue: True if should proceed to model, False if blocked
            - prompt_or_refusal: Transformed prompt (if continue) or refusal (if blocked)
            - metadata: Defense metadata
        """
        if self.defense is None:
            # No defense, return original prompt
            return True, prompt, {}

        # Apply defense check
        is_safe, defense_response, metadata = self.defense.check_prompt(prompt)

        # Determine defense behavior based on type
        defense_type = metadata.get("defense_type", "unknown")

        if defense_type in ["llm_guard", "grayswanai_guard"]:
            # LLM Guard / GraySwanAI Guard: Blocking defense
            if not is_safe:
                # Prompt blocked - return refusal without querying model
                self.logger.info(
                    f"Prompt blocked by {defense_type}: {metadata.get('categories', ['unknown'])}"
                )
                return False, defense_response, metadata
            else:
                # Prompt safe - proceed with original prompt
                self.logger.debug(f"Prompt passed {defense_type} check")
                return True, prompt, metadata

        elif defense_type in ["rephrasing", "perturbation"]:
            # Transforming defense: Use transformed prompt
            transformed_prompt = defense_response
            self.logger.debug(f"Prompt transformed by {defense_type} defense")
            return True, transformed_prompt, metadata

        else:
            # Unknown defense type - default to blocking behavior
            if not is_safe:
                return False, defense_response, metadata
            return True, prompt, metadata


class WhiteBoxModel(BaseModel):
    """White-box model for local inference using Transformers or vLLM."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        # Hugging Face specific configuration
        self.hf_token = config.get("hf_token")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.torch_dtype = config.get("torch_dtype", "float16")

        # vLLM configuration
        self.use_vllm = config.get("use_vllm", False)

        # Initialize vLLM model placeholder
        self.vllm_model = None  # vLLM model instance

        # Generation parameters (needed by both whitebox and blackbox models)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.do_sample = config.get("do_sample", True)

        # Controller compatibility flag (opt-in; default False so other methods are unaffected)
        self.controller_compat = bool(config.get("controller_compat", False))
        # For compat path, allow customizing input max_length (default 2048 like original controller)
        self.input_max_length = int(config.get("input_max_length", 2048))

        self.logger.info(f"Initialized WhiteBoxModel: {model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Use vLLM: {self.use_vllm}")
        self.logger.info(f"Batch size: {self.batch_size}")

    def load(self, hf_token: Optional[str] = None):
        """Load the model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer first (always needed)
        self._load_tokenizer(hf_token)

        # Check if using vLLM
        if self.use_vllm:
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is not available. Please install it with: pip install vllm"
                )

            self._load_vllm_model(hf_token)
        else:
            self._load_hf_model(hf_token)

        self.logger.info(f"Successfully loaded model: {self.model_name}")

    def _load_tokenizer(self, hf_token: Optional[str]):
        """Load tokenizer."""
        self.logger.info("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=hf_token or self.hf_token, trust_remote_code=True
        )

        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Tokenizer loaded successfully.")
        # Ensure left padding for decoder-only models
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass

    def _load_vllm_model(self, hf_token: Optional[str]):
        """Load model using vLLM for accelerated inference."""
        self.logger.info("Loading model with vLLM for accelerated inference")

        # Parse device configuration to determine tensor parallel size
        tensor_parallel_size = self._get_tensor_parallel_size()

        # ===== GPU allocation using centralized manager with backward compatibility =====
        from utils.gpu_manager import get_gpu_manager

        gpu_manager = get_gpu_manager()

        # Save original CUDA_VISIBLE_DEVICES
        original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        # Check for legacy _cka_gpu_override first (backward compatibility)
        gpu_override = self.config.get("_cka_gpu_override", None)

        if gpu_override is not None:
            # Legacy CKA-Agent GPU override
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_override)
            self.logger.info(
                f"[vLLM] Using legacy CKA GPU override: CUDA_VISIBLE_DEVICES={gpu_override}"
            )
            self.logger.info(
                f"[vLLM] Original CUDA_VISIBLE_DEVICES: {original_cuda_devices}"
            )
        else:
            # New centralized GPU allocation
            model_allocation = None
            if hasattr(self, "model_name"):
                # Check for target model allocation
                target_allocation = gpu_manager.get_allocation("target_model")
                if (
                    target_allocation
                    and self.model_name in target_allocation.description
                ):
                    model_allocation = target_allocation

                # Check for defense model allocation
                defense_allocation = gpu_manager.get_allocation("defense_model")
                if (
                    defense_allocation
                    and self.model_name in defense_allocation.description
                ):
                    model_allocation = defense_allocation

                # Check for attack method allocations
                for (
                    allocation_name,
                    allocation,
                ) in gpu_manager.get_all_allocations().items():
                    if self.model_name in allocation.description:
                        model_allocation = allocation
                        break

            if model_allocation:
                gpu_ids = ",".join(model_allocation.gpu_ids)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                self.logger.info(f"[vLLM] Using centralized GPU allocation: {gpu_ids}")
                self.logger.info(
                    f"[vLLM] Original CUDA_VISIBLE_DEVICES: {original_cuda_devices}"
                )
        # ===== END GPU allocation =====

        # Set up vLLM configuration from config
        vllm_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "max_logprobs": 10000,
            "max_model_len": self.vllm_kwargs.get("max_model_len", 4096),
            "gpu_memory_utilization": self.vllm_kwargs.get(
                "gpu_memory_utilization", 0.8
            ),
            "enforce_eager": self.vllm_kwargs.get("enforce_eager", True),
            "disable_custom_all_reduce": self.vllm_kwargs.get(
                "disable_custom_all_reduce", True
            ),
            "disable_log_stats": self.vllm_kwargs.get("disable_log_stats", True),
        }

        # Update with user-provided vLLM kwargs
        vllm_config.update(self.vllm_kwargs)

        # Override tensor_parallel_size if specified in vllm_kwargs
        if "tensor_parallel_size" in self.vllm_kwargs:
            vllm_config["tensor_parallel_size"] = self.vllm_kwargs[
                "tensor_parallel_size"
            ]

        self.logger.info(
            f"vLLM config: tensor_parallel_size={vllm_config['tensor_parallel_size']}"
        )
        self.logger.info(
            f"vLLM config: gpu_memory_utilization={vllm_config.get('gpu_memory_utilization', 'default')}"
        )

        # Set Hugging Face token if provided
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Load vLLM model
        self.vllm_model = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            download_dir=os.environ.get("TRANSFORMERS_CACHE", None),
            **vllm_config,
        )

        # ===== NEW: Restore original CUDA_VISIBLE_DEVICES =====
        if gpu_override is not None:
            if original_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
                self.logger.info(
                    f"[vLLM] Restored CUDA_VISIBLE_DEVICES to: {original_cuda_devices}"
                )
            else:
                # Original didn't have CUDA_VISIBLE_DEVICES set
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                self.logger.info(f"[vLLM] Cleared CUDA_VISIBLE_DEVICES override")
        # ===== END restore =====

        self.logger.info("vLLM model successfully loaded.")

    def _get_tensor_parallel_size(self) -> int:
        """Determine tensor parallel size based on device configuration and CUDA_VISIBLE_DEVICES."""
        # First check CUDA_VISIBLE_DEVICES environment variable
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            # Count available GPUs from CUDA_VISIBLE_DEVICES
            gpu_count = len([x for x in cuda_visible_devices.split(",") if x.strip()])
            self.logger.info(
                f"Detected {gpu_count} GPUs from CUDA_VISIBLE_DEVICES: {cuda_visible_devices}"
            )
            return gpu_count

        # Fallback to device configuration
        if self.device == "cpu":
            return 1
        elif self.device == "cuda":
            # Use all available GPUs
            return torch.cuda.device_count()
        elif "," in self.device and self.device.startswith("cuda:"):
            # Multiple GPUs specified (e.g., "cuda:0,2")
            gpu_part = self.device[5:]  # Remove "cuda:" prefix
            gpu_ids = [int(x.strip()) for x in gpu_part.split(",")]
            return len(gpu_ids)
        elif self.device.startswith("cuda:"):
            # Single GPU specified
            return 1
        else:
            return 1

    def _load_hf_model(self, hf_token: Optional[str]):
        """Load model using Hugging Face Transformers."""
        self.logger.info("Loading model with Hugging Face Transformers")

        # Determine torch dtype
        if self.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        # Load model
        if self.device_map is None:
            # Handle auto device selection for multi-GPU
            if isinstance(self.device, str) and self.device.lower() == "auto":
                # Use Transformers automatic device mapping across available GPUs
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=hf_token or self.hf_token,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                    trust_remote_code=True,
                )
                # Record effective device_map for downstream logic
                self.device_map = "auto"
            else:
                # Single device loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=hf_token or self.hf_token,
                    torch_dtype=torch_dtype,
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                    trust_remote_code=True,
                )
                self.model = self.model.to(self.device)
        else:
            # Multi-device loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token or self.hf_token,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                trust_remote_code=True,
            )

        self.logger.info("Hugging Face model successfully loaded.")

    def generate(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        if self.use_vllm and self.vllm_model is not None:
            return self._generate_vllm([prompt])[0]
        else:
            return self._generate_hf(prompt)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.use_vllm and self.vllm_model is not None:
            return self._generate_vllm(prompts)
        else:
            return [self._generate_hf(prompt) for prompt in prompts]

    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """Generate responses using vLLM with optimized batch processing."""
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Generate with vLLM - this is true batch processing
            self.logger.info(
                f"Generating {len(prompts)} prompts with vLLM batch processing"
            )
            outputs = self.vllm_model.generate(prompts, sampling_params)

            # Extract responses
            responses = []
            for output in outputs:
                response = output.outputs[0].text.strip()
                responses.append(response)

            self.logger.info(
                f"vLLM batch generation completed: {len(responses)} responses"
            )
            return responses

        except Exception as e:
            self.logger.error(f"vLLM generation failed: {e}")
            return [f"Error: {str(e)}"] * len(prompts)

    def _generate_hf(self, prompt: str) -> str:
        """Generate response using Hugging Face Transformers."""
        try:
            # Format prompt for Llama models
            if "llama" in self.model_name.lower():
                # Use Llama chat format
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt = prompt

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=(self.input_max_length if self.controller_compat else 1024),
            )

            # Move inputs to appropriate device
            if self.device_map is None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # When using device_map, we need to move inputs to the same device as the model
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Generation configuration
            gen_kwargs = {
                "max_new_tokens": self.max_length,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
                "top_p": self.top_p,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if not self.controller_compat:
                # Default path retains slight penalty to reduce loops
                gen_kwargs["repetition_penalty"] = 1.1
            generation_config = GenerationConfig(**gen_kwargs)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, generation_config=generation_config
                )

            # Decode response (exclude input tokens)
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"


class BlackBoxModel(BaseModel):
    """Black-box model for API-based inference."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        # API configuration
        self.api_key = config.get("api_key")  # API key from config
        self.base_url = config.get(
            "base_url", "https://api.openai.com/v1"
        )  # Base URL from config
        self.use_proxy = bool(config.get("use_proxy", False))
        self.proxy_api_key = config.get("proxy_api_key")

        # Generation parameters
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.timeout = config.get("timeout", 30)
        self.retry_delay = config.get("retry_delay", 1)
        # Unlimited retry backoff settings (exponential backoff with jitter)
        self.rate_limit_backoff_base = float(config.get("rate_limit_backoff_base", 1.0))
        self.rate_limit_backoff_max = float(config.get("rate_limit_backoff_max", 60.0))
        self.rate_limit_jitter = float(
            config.get("rate_limit_jitter", 0.2)
        )  # ±20% by default

        # Dataset batch processing configuration
        self.batch_size = config.get("batch_size", 1)  # Dataset batch processing size

        # API call configuration (simplified: no longer maintaining separate api_batch_size / rate and retry params)
        self.max_concurrent_requests = config.get("max_concurrent_requests", 10)

        self._setup_client()

    def load(self, hf_token: Optional[str] = None):
        """Load the blackbox model (API-based models don't need actual loading)."""
        self.logger.info(f"BlackBox model '{self.model_name}' is ready for API calls")
        self.logger.info(f"Provider: {self.config.get('provider', 'openai')}")
        self.logger.info(f"Base URL: {self.base_url}")
        self.logger.info(f"Max tokens: {self.max_tokens}")
        self.logger.info(f"Temperature: {self.temperature}")
        self.logger.info(f"Batch size: {self.batch_size}")

    def _setup_client(self):
        """Setup API client based on provider."""
        provider = self.config.get("provider", "openai")

        if provider == "openai":
            self._setup_openai_client()
        elif provider == "openrouter":
            self._setup_openai_client()  # OpenRouter uses OpenAI-compatible API
        elif provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic client not available. Install with: pip install anthropic"
                )
        elif provider == "cohere":
            try:
                import cohere

                self.client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError(
                    "Cohere client not available. Install with: pip install cohere"
                )
        elif provider == "together":
            try:
                import openai

                self.client = openai.OpenAI(
                    api_key=self.api_key, base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "OpenAI client (for Together-compatible endpoint) not available."
                )
        elif provider == "gemini":
            # Two modes: official google-genai, or OpenAI-compatible proxy
            if self.use_proxy:
                try:
                    import openai

                    key = self.proxy_api_key or self.api_key
                    self.client = openai.OpenAI(api_key=key, base_url=self.base_url)
                except ImportError:
                    raise ImportError(
                        "OpenAI client not available. Install with: pip install openai"
                    )
            else:
                try:
                    from google import genai

                    if not self.api_key:
                        raise ValueError("Gemini api_key is required for official mode")
                    self.gemini_client = genai.Client(api_key=self.api_key)
                except ImportError:
                    raise ImportError(
                        "google-genai client not available. Install with: pip install google-genai"
                    )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _setup_openai_client(self):
        """Setup OpenAI client."""
        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError(
                "OpenAI client not available. Install with: pip install openai"
            )

    # # Original version without defense
    # def generate(self, prompt: str) -> str:
    #     """Generate response for a single prompt (API only)."""
    #     return self._generate_api(prompt)

    # # New version with defense - llm_guard, grayswanai_guard
    # def generate(self, prompt: str) -> str:
    #     """Generate response for a single prompt (API only) with optional defense layer."""
    #     # ===== Apply defense check before API call =====
    #     if self.defense is not None:
    #         is_safe, refusal_response, metadata = self.defense.check_prompt(prompt)

    #         if not is_safe:
    #             # Prompt blocked by defense - return refusal without API call
    #             self.logger.info(f"Prompt blocked by defense: {metadata.get('category', 'unknown')}")
    #             return refusal_response

    #         # Log successful defense check
    #         self.logger.debug(f"Prompt passed defense check: {metadata.get('check_time', 0):.3f}s")

    #     return self._generate_api(prompt)

    # New version with defense - llm_guard, grayswanai_guard, rephrasing, perturbation
    def generate(self, prompt: str) -> str:
        """Generate response for a single prompt with optional defense layer."""
        # Apply defense check/transformation
        should_continue, prompt_or_refusal, metadata = self._apply_defense_to_prompt(
            prompt
        )

        if not should_continue:
            # Prompt blocked - return refusal without API call
            return prompt_or_refusal

        # Use potentially transformed prompt for API call
        prompt = prompt_or_refusal

        # Original API generation logic with potentially transformed prompt
        return self._generate_api(prompt)

    # Original version without defense
    # def generate_batch(self, prompts: List[str]) -> List[str]:
    #     """Generate responses for a batch of prompts (sequential API calls)."""
    #     responses = []
    #     for prompt in prompts:
    #         responses.append(self._generate_api(prompt))
    #     return responses

    # # New version with defense - llm_guard, grayswanai_guard
    # def generate_batch(self, prompts: List[str]) -> List[str]:
    #     """Generate responses for a batch of prompts with optional defense layer."""
    #     # Apply defense check to each prompt
    #     if self.defense is not None:
    #         filtered_prompts = []
    #         refusal_responses = []
    #         prompt_indices = []

    #         for idx, prompt in enumerate(prompts):
    #             is_safe, refusal_response, metadata = self.defense.check_prompt(prompt)

    #             if not is_safe:
    #                 refusal_responses.append((idx, refusal_response))
    #                 self.logger.info(f"Batch prompt {idx} blocked: {metadata.get('category', 'unknown')}")
    #             else:
    #                 filtered_prompts.append(prompt)
    #                 prompt_indices.append(idx)

    #         # If all blocked, return refusals
    #         if not filtered_prompts:
    #             return [r for _, r in sorted(refusal_responses, key=lambda x: x[0])]

    #         # Generate for safe prompts
    #         safe_responses = []
    #         for prompt in filtered_prompts:
    #             safe_responses.append(self._generate_api(prompt))

    #         # Reconstruct full list
    #         final_responses = [''] * len(prompts)
    #         for idx, response in refusal_responses:
    #             final_responses[idx] = response
    #         for idx, response in zip(prompt_indices, safe_responses):
    #             final_responses[idx] = response

    #         return final_responses

    #     # Original batch generation logic continues...
    #     responses = []
    #     for prompt in prompts:
    #         responses.append(self._generate_api(prompt))
    #     return responses

    # New version with defense - llm_guard, grayswanai_guard, rephrasing, perturbation
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts with optional defense layer."""
        # Apply defense to each prompt
        if self.defense is not None:
            final_responses = [None] * len(prompts)
            prompts_to_generate = []  # (original_idx, transformed_prompt)

            for idx, prompt in enumerate(prompts):
                should_continue, prompt_or_refusal, metadata = (
                    self._apply_defense_to_prompt(prompt)
                )

                if not should_continue:
                    # Blocked - store refusal
                    final_responses[idx] = prompt_or_refusal
                    defense_type = metadata.get("defense_type", "unknown")
                    self.logger.info(
                        f"Batch prompt {idx} blocked by {defense_type} defense"
                    )
                else:
                    # Allowed or transformed - queue for API call
                    prompts_to_generate.append((idx, prompt_or_refusal))

            # If all blocked, return early
            if not prompts_to_generate:
                return final_responses

            # Generate for allowed/transformed prompts
            safe_responses = []
            for _, transformed_prompt in prompts_to_generate:
                safe_responses.append(self._generate_api(transformed_prompt))

            # Fill responses at original positions
            for (original_idx, _), response in zip(prompts_to_generate, safe_responses):
                final_responses[original_idx] = response

            # Sanity check
            assert all(
                r is not None for r in final_responses
            ), "Some responses were not filled!"

            return final_responses

        # Original batch generation logic
        responses = []
        for prompt in prompts:
            responses.append(self._generate_api(prompt))
        return responses

    def _generate_api(self, prompt: str) -> str:
        """Generate response using API."""
        attempt = 0
        while True:
            try:
                provider = self.config.get("provider", "openai")
                if provider == "openai" or provider == "together":
                    # Check if using GPT-5 (new API format)
                    if self.model_name == "gpt-5":
                        response = self.client.responses.create(
                            model=self.model_name,
                            input=prompt,
                            reasoning={"effort": "low"},
                            text={"verbosity": "medium"},
                        )
                        content = response.output_text or ""
                        content = content.strip()
                        if content:
                            return content
                        raise RuntimeError("Empty response content")
                    else:
                        # Standard OpenAI API for other models
                        # Use max_completion_tokens for newer models (e.g. gpt-5.2, o1, o3)
                        # that don't support the legacy max_tokens parameter
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=self.max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                        )
                        content = response.choices[0].message.content or ""
                        content = content.strip()
                        if content:
                            return content
                        raise RuntimeError("Empty response content")
                elif provider == "openrouter":
                    # OpenRouter API (OpenAI-compatible)
                    extra_headers = self.config.get("extra_headers", {})
                    # Safely handle extra_headers
                    if extra_headers is None:
                        extra_headers = {}
                    elif isinstance(extra_headers, dict):
                        # Filter out commented-out config entries
                        extra_headers = {
                            k: v
                            for k, v in extra_headers.items()
                            if not k.startswith("#") and v is not None
                        }
                    else:
                        extra_headers = {}

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        extra_headers=extra_headers if extra_headers else None,
                        extra_body={},
                    )
                    content = response.choices[0].message.content or ""
                    content = content.strip()
                    if content:
                        return content
                    raise RuntimeError("Empty response content")
                elif provider == "anthropic":
                    msg = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if hasattr(msg, "content") and msg.content:
                        text = getattr(msg.content[0], "text", str(msg.content[0]))
                        text = (text or "").strip()
                        if text:
                            return text
                        raise RuntimeError("Empty response content")
                    raise RuntimeError("No content in response")
                elif provider == "cohere":
                    resp = self.client.chat(
                        model=self.model_name,
                        message=prompt,
                        temperature=self.temperature,
                    )
                    text = getattr(resp, "text", str(resp))
                    text = (text or "").strip()
                    if text:
                        return text
                    raise RuntimeError("Empty response content")
                elif provider == "gemini":
                    if self.use_proxy:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                        )
                        content = response.choices[0].message.content or ""
                        content = content.strip()
                        if content:
                            return content
                        raise RuntimeError("Empty response content")
                    # Official google-genai path
                    resp = self.gemini_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                    # Debug logging for Gemini official response safety/candidates info
                    try:
                        num_candidates = len(getattr(resp, "candidates", []) or [])
                        first_cand = (getattr(resp, "candidates", []) or [None])[0]
                        finish_reason = (
                            getattr(first_cand, "finish_reason", None)
                            if first_cand is not None
                            else None
                        )
                        # safety_ratings may live under candidate.safety_ratings or resp.safety_ratings depending on SDK
                        safety_ratings = None
                        if first_cand is not None and hasattr(
                            first_cand, "safety_ratings"
                        ):
                            safety_ratings = getattr(first_cand, "safety_ratings", None)
                        if safety_ratings is None and hasattr(resp, "safety_ratings"):
                            safety_ratings = getattr(resp, "safety_ratings", None)
                        self.logger.info(
                            f"Gemini response debug: candidates={num_candidates}, finish_reason={finish_reason}, has_safety_ratings={'yes' if safety_ratings else 'no'}"
                        )
                    except Exception as dbg_e:
                        self.logger.debug(
                            f"Gemini response debug logging failed: {dbg_e}"
                        )
                    if hasattr(resp, "text") and resp.text:
                        text = resp.text.strip()
                        if text:
                            return text
                    if hasattr(resp, "candidates") and resp.candidates:
                        cand = resp.candidates[0]
                        if (
                            hasattr(cand, "content")
                            and cand.content
                            and getattr(cand.content, "parts", None)
                        ):
                            parts = getattr(cand.content, "parts", None)
                            for p in parts:
                                if hasattr(p, "text") and p.text:
                                    text = p.text.strip()
                                    if text:
                                        return text
                    # Input prompt preview (kept via logger only)
                    try:
                        prompt_preview = (
                            (prompt[:800] + "...") if len(prompt) > 800 else prompt
                        )
                        self.logger.info(
                            f"Gemini empty response; prompt_len={len(prompt)} preview=\n{prompt_preview}"
                        )
                    except Exception:
                        pass
                    # Treat empty response as explicit refusal to avoid infinite retries
                    self.logger.info(
                        "Gemini returned empty content; treating as refusal and skipping retries."
                    )
                    return "I'm sorry, I can't assist with that."
                else:
                    raise ValueError(f"Unsupported provider for generation: {provider}")
            except Exception as e:
                # Check for OpenAI content safety rejection (400, invalid_prompt)
                # Error: {'type': 'invalid_request_error', 'code': 'invalid_prompt',
                #         'message': "Invalid prompt: we've limited access to this content for safety reasons..."}
                error_str = str(e)
                if (
                    "invalid_request_error" in error_str
                    and "invalid_prompt" in error_str
                    and "limited access" in error_str
                ):
                    self.logger.warning(
                        f"OpenAI content safety rejection (invalid_prompt). Returning refusal."
                    )
                    return "I'm sorry, but I cannot provide this information due to safety policy restrictions."

                # Check for Kimi K2 content moderation error (403 with specific message)
                if (
                    provider == "openrouter"
                    and "moonshotai/kimi-k2" in self.model_name
                    and "403" in error_str
                    and "requires moderation on OpenInference" in error_str
                    and "flagged for" in error_str
                ):

                    # Extract the flagged reason from error message
                    import re

                    flagged_match = re.search(r'flagged for "([^"]+)"', error_str)
                    flagged_reason = (
                        flagged_match.group(1)
                        if flagged_match
                        else "content policy violation"
                    )

                    self.logger.warning(
                        f"Kimi K2 content moderation triggered: {flagged_reason}. Returning refusal message."
                    )
                    return f'I cannot respond. Your input was flagged for "{flagged_reason}".'

                # Determine backoff (use Retry-After if available)
                retry_after = None
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        ra = (
                            resp.headers.get("Retry-After")
                            if hasattr(resp, "headers")
                            else None
                        )
                        if ra is not None:
                            retry_after = float(ra)
                except Exception:
                    retry_after = None
                attempt += 1
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(
                        self.rate_limit_backoff_base * (2 ** (attempt - 1)),
                        self.rate_limit_backoff_max,
                    )
                )
                # Apply jitter
                jitter_factor = 1.0 + random.uniform(
                    -self.rate_limit_jitter, self.rate_limit_jitter
                )
                delay = max(0.0, delay * jitter_factor)
                self.logger.warning(
                    f"BlackBoxModel API call failed (attempt {attempt}): {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

    def generate_batch_messages(
        self, messages_batches: List[List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate responses for a batch of OpenAI-style messages (API only)."""
        provider = self.config.get("provider", "openai")
        outputs: List[str] = []
        if provider in ("openai", "together") or (
            provider == "gemini" and self.use_proxy
        ):
            for messages in messages_batches:
                attempt = 0
                while True:
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_completion_tokens=self.max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                        )
                        content = resp.choices[0].message.content or ""
                        content = content.strip()
                        if content:
                            outputs.append(content)
                            break
                        raise RuntimeError("Empty response content")
                    except Exception as e:
                        # Check for OpenAI content safety rejection (400, invalid_prompt)
                        error_str = str(e)
                        if (
                            "invalid_request_error" in error_str
                            and "invalid_prompt" in error_str
                            and "limited access" in error_str
                        ):
                            self.logger.warning(
                                f"OpenAI content safety rejection (invalid_prompt). Returning refusal."
                            )
                            outputs.append(
                                "I'm sorry, but I cannot provide this information due to safety policy restrictions."
                            )
                            break

                        retry_after = None
                        try:
                            resp_ex = getattr(e, "response", None)
                            if resp_ex is not None:
                                ra = (
                                    resp_ex.headers.get("Retry-After")
                                    if hasattr(resp_ex, "headers")
                                    else None
                                )
                                if ra is not None:
                                    retry_after = float(ra)
                        except Exception:
                            retry_after = None
                        attempt += 1
                        delay = (
                            retry_after
                            if retry_after is not None
                            else min(
                                self.rate_limit_backoff_base * (2 ** (attempt - 1)),
                                self.rate_limit_backoff_max,
                            )
                        )
                        jitter_factor = 1.0 + random.uniform(
                            -self.rate_limit_jitter, self.rate_limit_jitter
                        )
                        delay = max(0.0, delay * jitter_factor)
                        self.logger.warning(
                            f"BlackBoxModel messages API failed (attempt {attempt}): {e}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
            return outputs
        elif provider == "anthropic":
            for messages in messages_batches:
                attempt = 0
                while True:
                    try:
                        # Anthropic supports proper multi-turn messages
                        # But requires alternating user/assistant roles
                        formatted_messages = []
                        for m in messages:
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            # Anthropic uses 'user' and 'assistant' roles
                            if role in ["user", "assistant"]:
                                formatted_messages.append(
                                    {"role": role, "content": content}
                                )
                            else:
                                # Default to user for system or unknown roles
                                formatted_messages.append(
                                    {"role": "user", "content": content}
                                )

                        msg = self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            messages=formatted_messages,  # Use proper multi-turn format
                        )
                        if hasattr(msg, "content") and msg.content:
                            outputs.append(
                                getattr(
                                    msg.content[0], "text", str(msg.content[0])
                                ).strip()
                            )
                            break
                        raise RuntimeError("No content in response")
                    except Exception as e:
                        attempt += 1
                        delay = min(
                            self.rate_limit_backoff_base * (2 ** (attempt - 1)),
                            self.rate_limit_backoff_max,
                        )
                        delay *= 1.0 + random.uniform(
                            -self.rate_limit_jitter, self.rate_limit_jitter
                        )
                        delay = max(0.0, delay)
                        self.logger.warning(
                            f"Anthropic messages API failed (attempt {attempt}): {e}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
            return outputs
        elif provider == "cohere":
            for messages in messages_batches:
                attempt = 0
                while True:
                    try:
                        # Cohere supports chat_history for multi-turn conversations
                        # Format: chat_history is list of previous messages, message is current user message
                        chat_history = []
                        current_message = ""

                        for i, m in enumerate(messages):
                            role = m.get("role", "user")
                            content = m.get("content", "")

                            if i < len(messages) - 1:
                                # Add to chat history (all but last message)
                                # Cohere uses 'USER' and 'CHATBOT' roles
                                cohere_role = "USER" if role == "user" else "CHATBOT"
                                chat_history.append(
                                    {"role": cohere_role, "message": content}
                                )
                            else:
                                # Last message should be user message for current prompt
                                current_message = content

                        # Cohere API call with chat_history
                        resp = self.client.chat(
                            model=self.model_name,
                            message=(
                                current_message
                                if current_message
                                else messages[-1].get("content", "")
                            ),
                            chat_history=chat_history if chat_history else None,
                            temperature=self.temperature,
                        )
                        content = getattr(resp, "text", str(resp)).strip()
                        if content:
                            outputs.append(content)
                            break
                        raise RuntimeError("Empty response content")
                    except Exception as e:
                        attempt += 1
                        delay = min(
                            self.rate_limit_backoff_base * (2 ** (attempt - 1)),
                            self.rate_limit_backoff_max,
                        )
                        delay *= 1.0 + random.uniform(
                            -self.rate_limit_jitter, self.rate_limit_jitter
                        )
                        delay = max(0.0, delay)
                        self.logger.warning(
                            f"Cohere messages API failed (attempt {attempt}): {e}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
            return outputs
        elif provider == "gemini" and not self.use_proxy:
            # Official google-genai path: emulate messages via single prompt
            for messages in messages_batches:
                text_parts = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    text_parts.append(f"{role}: {content}")
                prompt = "\n\n".join(text_parts)
                outputs.append(self._generate_api(prompt))
            return outputs
        else:
            raise ValueError(f"Unsupported provider for messages: {provider}")


class ModelLoader:
    """Factory class for loading different types of models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ModelLoader with configuration."""
        self.config = config
        self.logger = logging.getLogger("ModelLoader")

        # Extract model configuration
        self.model_type = config.get("type", "whitebox")
        self.model_config = config.get(self.model_type, {})
        self.model_name = self.model_config.get("name")

        if not self.model_name:
            raise ValueError(
                f"Model name not specified in config for type: {self.model_type}"
            )

        self.logger.info(
            f"ModelLoader initialized for {self.model_type} model: {self.model_name}"
        )

        # Defense configuration
        defense_config = config.get("defense", {})
        if defense_config:
            self.model_config["defense"] = defense_config
            self.logger.info(
                f"Defense configuration passed to model: enabled={defense_config.get('enabled', False)}"
            )

    def load_model(self) -> BaseModel:
        """Load a model based on configuration."""
        if self.model_type == "whitebox":
            model = WhiteBoxModel(self.model_name, self.model_config)
        elif self.model_type == "blackbox":
            model = BlackBoxModel(self.model_name, self.model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Load the model
        model.load()
        return model

    @staticmethod
    def load_model_static(
        model_type: str, model_name: str, config: Dict[str, Any]
    ) -> BaseModel:
        """Static method to load a model based on type."""
        if model_type == "whitebox":
            return WhiteBoxModel(model_name, config)
        elif model_type == "blackbox":
            return BlackBoxModel(model_name, config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class JudgeModel(BaseModel):
    """Judge model for evaluating jailbreak results."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        # Judge-specific configuration

        # Hugging Face specific configuration
        self.hf_token = config.get("hf_token")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.torch_dtype = config.get("torch_dtype", "float16")

        self.logger.info(f"Initialized JudgeModel: {model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Use vLLM: {self.use_vllm}")
        self.logger.info(f"Dataset batch size: {self.batch_size}")

    def load(self, hf_token: Optional[str] = None):
        """Load the model and tokenizer."""
        self.logger.info(f"Loading judge model: {self.model_name}")

        # Load tokenizer
        self._load_tokenizer(hf_token)

        # Check whether to use vLLM
        if self.use_vllm:
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is not available. Please install it with: pip install vllm"
                )

            self._load_vllm_model(hf_token)
        else:
            self._load_hf_model(hf_token)

        self.logger.info(f"Successfully loaded judge model: {self.model_name}")

    def _load_tokenizer(self, hf_token: Optional[str]):
        """Load tokenizer."""
        self.logger.info("Loading judge tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=hf_token or self.hf_token, trust_remote_code=True
        )

        # Set padding token and padding direction
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use left padding for decoder-only models
        self.tokenizer.padding_side = "left"
        self.logger.info(
            f"Set tokenizer padding_side to: {self.tokenizer.padding_side}"
        )

        self.logger.info("Judge tokenizer loaded successfully.")

    def _load_vllm_model(self, hf_token: Optional[str]):
        """Load model using vLLM."""
        self.logger.info("Loading judge model with vLLM for accelerated inference")

        # Parse device configuration to determine tensor parallel size
        tensor_parallel_size = self._get_tensor_parallel_size()

        # Set vLLM configuration
        vllm_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "max_logprobs": 10000,
            "max_model_len": self.vllm_kwargs.get("max_model_len", 4096),
            "gpu_memory_utilization": self.vllm_kwargs.get(
                "gpu_memory_utilization", 0.3
            ),
            "enforce_eager": self.vllm_kwargs.get("enforce_eager", True),
            "disable_custom_all_reduce": self.vllm_kwargs.get(
                "disable_custom_all_reduce", True
            ),
            "disable_log_stats": self.vllm_kwargs.get("disable_log_stats", True),
        }

        # Update with user-provided vLLM kwargs
        vllm_config.update(self.vllm_kwargs)

        self.logger.info(
            f'Judge vLLM config: tensor_parallel_size={vllm_config["tensor_parallel_size"]}'
        )
        self.logger.info(
            f'Judge vLLM config: gpu_memory_utilization={vllm_config.get("gpu_memory_utilization", "default")}'
        )

        # Set Hugging Face token
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Load vLLM model
        self.vllm_model = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            download_dir=os.environ.get("TRANSFORMERS_CACHE", None),
            **vllm_config,
        )

        self.logger.info("Judge vLLM model successfully loaded.")

    def _load_hf_model(self, hf_token: Optional[str]):
        """Load model using Hugging Face Transformers."""
        self.logger.info("Loading judge model with Hugging Face Transformers")

        # Set data type
        torch_dtype = (
            getattr(torch, self.torch_dtype)
            if isinstance(self.torch_dtype, str)
            else self.torch_dtype
        )

        # Load model (aligned with WhiteBoxModel)
        if self.device_map is None:
            # Single device loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token or self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
            )
            # Move to configured device if set, otherwise keep default
            if hasattr(self, "device") and isinstance(self.device, str):
                self.model = self.model.to(self.device)
        else:
            # Multi-device loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token or self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
            )

        self.logger.info("Judge Hugging Face model loaded successfully.")

    def _get_tensor_parallel_size(self) -> int:
        """Determine tensor parallel size based on CUDA_VISIBLE_DEVICES."""
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if cuda_devices:
            device_count = len(cuda_devices.split(","))
            return min(device_count, self.vllm_kwargs.get("tensor_parallel_size", 1))
        return 1

    def evaluate(self, jailbreak_results: List[str]) -> List[float]:
        """Evaluate safety of jailbreak results."""
        # Concrete evaluation logic should be implemented here
        # For demonstration, return simulated safety scores
        return [0.8] * len(jailbreak_results)

    def evaluate_batch(self, jailbreak_results: List[str]) -> List[float]:
        """Batch evaluate jailbreak results for safety."""
        scores = []

        # Group by batch size
        for i in range(0, len(jailbreak_results), self.batch_size):
            batch_results = jailbreak_results[i : i + self.batch_size]
            batch_scores = self.evaluate(batch_results)
            scores.extend(batch_scores)

        return scores
