# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured dataclass definitions and factory for stage configurations.

This module provides a robust, type-safe approach to building stage configurations
programmatically, replacing ad-hoc dictionary construction.
"""

import json
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StageRuntimeConfig:
    """Runtime configuration for a stage.
    
    Attributes:
        process: Whether to run this stage in a separate process
        devices: Comma-separated string of device IDs (e.g., "0,1,2")
        max_batch_size: Maximum batch size for this stage
    """
    process: bool
    devices: str
    max_batch_size: int


@dataclass
class StageConfig:
    """Complete configuration for a single stage in the pipeline.
    
    Attributes:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage (e.g., "diffusion", "llm")
        runtime: Runtime configuration for this stage
        engine_args: Dictionary of engine-specific arguments
        final_output: Whether this stage produces final output
        final_output_type: Type of final output (e.g., "image", "text", "audio")
    """
    stage_id: int
    stage_type: str
    runtime: StageRuntimeConfig
    engine_args: dict[str, Any]
    final_output: bool
    final_output_type: str
    
    def to_dict(self) -> dict[str, Any]:
        """Converts the dataclass to a plain dictionary.
        
        Returns:
            Dictionary with stage configuration
        """
        return {
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "runtime": {
                "process": self.runtime.process,
                "devices": self.runtime.devices,
                "max_batch_size": self.runtime.max_batch_size,
            },
            "engine_args": self.engine_args,  # Pass through as-is
            "final_output": self.final_output,
            "final_output_type": self.final_output_type,
        }
    
    def to_omegaconf(self) -> DictConfig:
        """Converts the dataclass to an OmegaConf DictConfig.
        
        Returns:
            OmegaConf DictConfig compatible with OmniStage initialization
        """
        # Convert to dict first, then wrap in OmegaConf
        # This allows engine_args to remain as whatever type it already is
        return OmegaConf.create(self.to_dict())


class StageConfigFactory:
    """Factory for creating stage configurations with validation and normalization.
    
    This factory provides methods to create structured StageConfig objects
    with proper validation, type checking, and normalization of parameters.
    """
    
    @staticmethod
    def _get_device_string(parallel_config: Any | None, num_devices: int | None = None) -> str:
        """Generate device string from parallel configuration or device count.
        
        Args:
            parallel_config: Parallel configuration object with world_size attribute
            num_devices: Number of devices (used if parallel_config is None)
            
        Returns:
            Comma-separated device string (e.g., "0,1,2,3")
        """
        if parallel_config is not None:
            device_count = parallel_config.world_size
        elif num_devices is not None:
            device_count = num_devices
        else:
            device_count = 1
        
        return ",".join(str(i) for i in range(device_count))
    
    @staticmethod
    def _get_default_cache_config(cache_backend: str | None) -> dict[str, Any] | None:
        """Get default cache configuration for a given backend.
        
        Args:
            cache_backend: Name of the cache backend (e.g., "cache_dit", "tea_cache")
            
        Returns:
            Dictionary with default cache configuration, or None if no defaults exist
        """
        if cache_backend == "cache_dit":
            return {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.24,
                "max_continuous_cached_steps": 3,
                "enable_taylorseer": False,
                "taylorseer_order": 1,
                "scm_steps_mask_policy": None,
                "scm_steps_policy": "dynamic",
            }
        if cache_backend == "tea_cache":
            return {
                "rel_l1_thresh": 0.2,
            }
        return None
    
    @staticmethod
    def _normalize_cache_config(cache_backend: str | None, cache_config: Any | None) -> Any | None:
        """Normalize cache configuration, applying defaults if needed.
        
        Args:
            cache_backend: Name of the cache backend
            cache_config: User-provided cache configuration (may be JSON string or dict)
            
        Returns:
            Normalized cache configuration dictionary or None
        """
        if isinstance(cache_config, str):
            try:
                cache_config = json.loads(cache_config)
            except json.JSONDecodeError:
                logger.warning("Invalid cache_config JSON, using defaults.")
                cache_config = None
        
        if cache_config is None and cache_backend not in (None, "", "none"):
            cache_config = StageConfigFactory._get_default_cache_config(cache_backend)
        
        return cache_config
    
    @classmethod
    def create_default_diffusion(cls, kwargs: dict[str, Any]) -> list[DictConfig]:
        """Create default diffusion stage configuration from kwargs.
        
        This method builds a structured StageConfig for a diffusion model
        with proper validation and normalization of all parameters.
        
        Args:
            kwargs: Dictionary of configuration parameters including:
                - dtype: Data type for the model (converted to string)
                - cache_backend: Cache backend name (default: "none")
                - cache_config: Cache configuration (optional)
                - parallel_config: Parallel configuration with world_size attribute
                - Additional engine arguments to pass through
                
        Returns:
            List containing a single OmegaConf DictConfig for the diffusion stage
        """
        # Convert dtype to string to avoid OmegaConf serialization issues
        if "dtype" in kwargs:
            kwargs["dtype"] = str(kwargs["dtype"])
        
        # Normalize cache configuration
        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = cls._normalize_cache_config(
            cache_backend, 
            kwargs.get("cache_config", None)
        )
        
        # Generate device string from parallel configuration
        devices = cls._get_device_string(kwargs.get("parallel_config"))
        
        # Build runtime configuration
        runtime_config = StageRuntimeConfig(
            process=True,
            devices=devices,
            max_batch_size=1,
        )
        
        # Build engine arguments with cache config
        # Wrap in OmegaConf.create to match original behavior
        engine_args = OmegaConf.create({
            **kwargs,
            "cache_backend": cache_backend,
            "cache_config": cache_config,
        })
        # Set model_stage after OmegaConf creation to match original behavior
        engine_args["model_stage"] = "diffusion"
        
        # Create structured stage configuration
        stage_config = StageConfig(
            stage_id=0,
            stage_type="diffusion",
            runtime=runtime_config,
            engine_args=engine_args,
            final_output=True,
            final_output_type="image",
        )
        
        # Convert to OmegaConf for compatibility with existing code
        return [stage_config.to_omegaconf()]
    
    @classmethod
    def create_default_diffusion_async(cls, kwargs: dict[str, Any]) -> list[dict]:
        """Create default diffusion stage configuration for async mode.
        
        This is similar to create_default_diffusion but handles async-specific
        parallel configuration logic where parallel_config may be built from
        individual parameters if not provided.
        
        Args:
            kwargs: Dictionary of configuration parameters
            
        Returns:
            List containing a single dict for the diffusion stage (not wrapped in OmegaConf)
        """
        # Local import to avoid potential circular dependencies with diffusion module
        from vllm_omni.diffusion.data import DiffusionParallelConfig
        
        # Normalize cache configuration
        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = cls._normalize_cache_config(
            cache_backend, 
            kwargs.get("cache_config", None)
        )
        
        # Build or extract parallel_config
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            devices = cls._get_device_string(parallel_config)
        else:
            # Build parallel_config from individual parameters
            ulysses_degree = kwargs.get("ulysses_degree") or 1
            ring_degree = kwargs.get("ring_degree") or 1
            sequence_parallel_size = kwargs.get("sequence_parallel_size")
            tensor_parallel_size = kwargs.get("tensor_parallel_size") or 1
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree
            num_devices = sequence_parallel_size * tensor_parallel_size
            devices = cls._get_device_string(None, num_devices)
            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=tensor_parallel_size,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                cfg_parallel_size=1,
            )
        
        # Build runtime configuration
        runtime_config = StageRuntimeConfig(
            process=True,
            devices=devices,
            max_batch_size=1,
        )
        
        # Build engine arguments specific to async mode
        # Note: We don't wrap in OmegaConf.create here because parallel_config
        # may not be serializable. The original code used a plain dict.
        engine_args = {
            "parallel_config": parallel_config,
            "vae_use_slicing": kwargs.get("vae_use_slicing", False),
            "vae_use_tiling": kwargs.get("vae_use_tiling", False),
            "cache_backend": cache_backend,
            "cache_config": cache_config,
            "enable_cpu_offload": kwargs.get("enable_cpu_offload", False),
            "enforce_eager": kwargs.get("enforce_eager", False),
            "model_stage": "diffusion",
        }
        
        # Create structured stage configuration
        stage_config = StageConfig(
            stage_id=0,
            stage_type="diffusion",
            runtime=runtime_config,
            engine_args=engine_args,
            final_output=True,
            final_output_type="image",
        )
        
        # Return as plain dict for async mode (engine_args contains non-serializable parallel_config)
        return [stage_config.to_dict()]
