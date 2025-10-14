"""
Greater Than Circuit Analysis Package

This package provides tools for reverse-engineering the "greater than" circuit
in transformer language models using mechanistic interpretability techniques.

Based on the TransformerLens library and methodology developed by Neel Nanda.
"""

from .model_setup import ModelSetup
from .prompt_design import PromptGenerator, PromptExample
from .activation_patching import ActivationPatcher, PatchingResult
from .circuit_analysis import CircuitAnalyzer, CircuitComponent
from .circuit_validation import CircuitValidator, ValidationResult
from .visualization import CircuitVisualizer

__version__ = "0.1.0"
__author__ = "Research Implementation"

__all__ = [
    "ModelSetup",
    "PromptGenerator",
    "PromptExample",
    "ActivationPatcher",
    "PatchingResult",
    "CircuitAnalyzer",
    "CircuitComponent",
    "CircuitValidator",
    "ValidationResult",
    "CircuitVisualizer",
]