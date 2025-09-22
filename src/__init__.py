"""
Greater Than Circuit Reverse Engineering Package

This package implements mechanistic interpretability techniques to reverse-engineer
the "greater than" circuit in GPT-2 Small, following methodologies from the field
of transformer mechanistic interpretability.

Acknowledgment: This work builds upon the foundational research and methodologies
developed by Neel Nanda and the broader mechanistic interpretability community.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "contact@example.com"

from .model_setup import ModelSetup
from .prompt_design import PromptGenerator
from .activation_patching import ActivationPatcher
from .circuit_analysis import CircuitAnalyzer
from .visualization import CircuitVisualizer

__all__ = [
    "ModelSetup",
    "PromptGenerator", 
    "ActivationPatcher",
    "CircuitAnalyzer",
    "CircuitVisualizer"
]