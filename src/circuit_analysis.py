"""
Circuit Analysis Module

This module provides tools for analyzing the structure and behavior of the
greater than circuit identified through activation patching.

Acknowledgment: Analysis techniques inspired by circuit-based interpretability
research pioneered by Neel Nanda and the mechanistic interpretability community.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from activation_patching import PatchingResult
import logging

logger = logging.getLogger(__name__)


@dataclass
class CircuitComponent:
    """
    Represents a component in the identified circuit.
    
    Attributes:
        name (str): Name/identifier of the component
        layer (int): Layer number
        head (Optional[int]): Attention head number (if applicable)
        importance_score (float): Importance score from patching experiments
        component_type (str): Type of component (attention, mlp, etc.)
        connections (List[str]): List of components this connects to
    """
    name: str
    layer: int
    head: Optional[int] = None
    importance_score: float = 0.0
    component_type: str = "unknown"
    connections: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []


class CircuitAnalyzer:
    """
    Analyzes the greater than circuit structure and behavior.
    
    This class provides functionality to process patching results, identify
    circuit components, and analyze their roles and interactions.
    """
    
    def __init__(self, model: HookedTransformer):
        """
        Initialize the CircuitAnalyzer.
        
        Args:
            model (HookedTransformer): The model containing the circuit
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.circuit_components = {}
        self.connection_matrix = None
        
        logger.info(f"CircuitAnalyzer initialized for {model.cfg.model_name}")
    
    def identify_circuit_components(
        self,
        patching_results: List[PatchingResult],
        importance_threshold: float = 0.1,
        top_k: int = 20
    ) -> Dict[str, CircuitComponent]:
        """
        Identify key circuit components from patching results.
        
        Args:
            patching_results (List[PatchingResult]): Results from activation patching
            importance_threshold (float): Minimum importance to include component
            top_k (int): Maximum number of components to include
            
        Returns:
            Dict[str, CircuitComponent]: Dictionary of identified components
        """
        # Filter and sort results by importance
        important_results = [
            r for r in patching_results 
            if abs(r.effect_size) >= importance_threshold
        ]
        important_results.sort(key=lambda x: abs(x.effect_size), reverse=True)
        important_results = important_results[:top_k]
        
        components = {}
        
        for result in important_results:
            # Create component identifier
            comp_id = self._create_component_id(result)
            
            # Determine component type
            comp_type = self._determine_component_type(result.hook_name)
            
            # Create component
            component = CircuitComponent(
                name=comp_id,
                layer=result.layer or 0,
                head=result.head,
                importance_score=abs(result.effect_size),
                component_type=comp_type
            )
            
            components[comp_id] = component
        
        self.circuit_components = components
        logger.info(f"Identified {len(components)} circuit components")
        
        return components
    
    def analyze_layer_contributions(
        self,
        patching_results: List[PatchingResult]
    ) -> Dict[int, float]:
        """
        Analyze the contribution of each layer to the circuit.
        
        Args:
            patching_results (List[PatchingResult]): Patching results to analyze
            
        Returns:
            Dict[int, float]: Average importance score by layer
        """
        layer_contributions = {}
        
        for result in patching_results:
            if result.layer is not None:
                if result.layer not in layer_contributions:
                    layer_contributions[result.layer] = []
                layer_contributions[result.layer].append(abs(result.effect_size))
        
        # Calculate average contribution per layer
        avg_contributions = {
            layer: np.mean(scores)
            for layer, scores in layer_contributions.items()
        }
        
        return avg_contributions
    
    def analyze_attention_patterns(
        self,
        tokens: torch.Tensor,
        target_heads: List[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze attention patterns for important heads in the circuit.
        
        Args:
            tokens (torch.Tensor): Input tokens to analyze
            target_heads (List[Tuple[int, int]]): Specific (layer, head) pairs to analyze
            
        Returns:
            Dict[str, torch.Tensor]: Attention patterns for each head
        """
        if target_heads is None:
            # Use heads from identified circuit components
            target_heads = [
                (comp.layer, comp.head)
                for comp in self.circuit_components.values()
                if comp.head is not None
            ]
        
        attention_patterns = {}
        
        # Store attention patterns
        def store_attention_hook(pattern: torch.Tensor, hook_name: str):
            attention_patterns[hook_name] = pattern.clone().detach()
        
        # Add hooks for target heads
        hooks = []
        for layer, head in target_heads:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            hook = self.model.add_hook(
                hook_name,
                lambda pattern, hook=hook_name: store_attention_hook(pattern, hook)
            )
            hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            self.model(tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Extract patterns for specific heads
        head_patterns = {}
        for layer, head in target_heads:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            if hook_name in attention_patterns:
                # Extract pattern for specific head
                pattern = attention_patterns[hook_name][0, head, :, :]  # [seq_len, seq_len]
                head_patterns[f"L{layer}H{head}"] = pattern
        
        return head_patterns
    
    def compute_activation_attribution(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        target_components: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute activation attribution scores for circuit components.
        
        Args:
            clean_tokens (torch.Tensor): Clean input tokens
            corrupted_tokens (torch.Tensor): Corrupted input tokens
            target_components (List[str]): Specific components to analyze
            
        Returns:
            Dict[str, torch.Tensor]: Attribution scores for each component
        """
        if target_components is None:
            target_components = list(self.circuit_components.keys())
        
        attributions = {}
        
        # Get clean and corrupted activations
        clean_acts = self._get_activations(clean_tokens, target_components)
        corrupted_acts = self._get_activations(corrupted_tokens, target_components)
        
        # Compute attribution as difference in activations
        for comp_name in target_components:
            if comp_name in clean_acts and comp_name in corrupted_acts:
                attribution = clean_acts[comp_name] - corrupted_acts[comp_name]
                attributions[comp_name] = attribution
        
        return attributions
    
    def find_information_flow(
        self,
        tokens: torch.Tensor,
        source_components: List[str],
        target_components: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze information flow between circuit components.
        
        Args:
            tokens (torch.Tensor): Input tokens
            source_components (List[str]): Source components
            target_components (List[str]): Target components
            
        Returns:
            Dict[str, Dict[str, float]]: Information flow matrix
        """
        # This is a simplified version - full implementation would use
        # more sophisticated techniques like gradient-based analysis
        
        flow_matrix = {}
        
        # Get activations for all components
        all_components = source_components + target_components
        activations = self._get_activations(tokens, all_components)
        
        # Compute correlations as a proxy for information flow
        for source in source_components:
            flow_matrix[source] = {}
            if source not in activations:
                continue
                
            source_act = activations[source].flatten()
            
            for target in target_components:
                if target not in activations:
                    flow_matrix[source][target] = 0.0
                    continue
                
                target_act = activations[target].flatten()
                
                # Compute correlation
                try:
                    correlation = np.corrcoef(
                        source_act.cpu().numpy(),
                        target_act.cpu().numpy()
                    )[0, 1]
                    flow_matrix[source][target] = float(correlation)
                except:
                    flow_matrix[source][target] = 0.0
        
        return flow_matrix
    
    def analyze_positional_effects(
        self,
        patching_results: List[PatchingResult]
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze how patching effects vary by token position.
        
        Args:
            patching_results (List[PatchingResult]): Patching results to analyze
            
        Returns:
            Dict[int, Dict[str, float]]: Effects organized by position and component
        """
        positional_effects = {}
        
        for result in patching_results:
            if result.position is not None:
                pos = result.position
                comp_id = self._create_component_id(result)
                
                if pos not in positional_effects:
                    positional_effects[pos] = {}
                
                positional_effects[pos][comp_id] = abs(result.effect_size)
        
        return positional_effects
    
    def create_circuit_summary(
        self,
        patching_results: List[PatchingResult]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive summary of the identified circuit.
        
        Args:
            patching_results (List[PatchingResult]): All patching results
            
        Returns:
            Dict[str, Any]: Comprehensive circuit summary
        """
        # Identify components
        components = self.identify_circuit_components(patching_results)
        
        # Analyze layers
        layer_contributions = self.analyze_layer_contributions(patching_results)
        
        # Analyze positions
        positional_effects = self.analyze_positional_effects(patching_results)
        
        # Create summary
        summary = {
            "circuit_overview": {
                "total_components": len(components),
                "layers_involved": list(layer_contributions.keys()),
                "most_important_layer": max(layer_contributions.items(), key=lambda x: x[1])[0],
                "attention_heads": [
                    f"L{comp.layer}H{comp.head}" 
                    for comp in components.values() 
                    if comp.head is not None
                ],
                "mlp_components": [
                    comp.name 
                    for comp in components.values() 
                    if comp.component_type == "mlp"
                ]
            },
            "component_importance": {
                comp.name: comp.importance_score
                for comp in sorted(components.values(), 
                                 key=lambda x: x.importance_score, 
                                 reverse=True)
            },
            "layer_analysis": layer_contributions,
            "positional_analysis": {
                pos: {
                    "avg_effect": np.mean(list(effects.values())),
                    "max_effect": max(effects.values()) if effects else 0,
                    "active_components": len(effects)
                }
                for pos, effects in positional_effects.items()
            },
            "circuit_depth": len(set(comp.layer for comp in components.values())),
            "attention_vs_mlp": {
                "attention_components": len([c for c in components.values() if c.component_type == "attention"]),
                "mlp_components": len([c for c in components.values() if c.component_type == "mlp"]),
                "other_components": len([c for c in components.values() if c.component_type not in ["attention", "mlp"]])
            }
        }
        
        return summary
    
    def _create_component_id(self, result: PatchingResult) -> str:
        """Create a unique identifier for a circuit component."""
        if result.head is not None:
            return f"L{result.layer}H{result.head}"
        elif result.layer is not None:
            comp_type = self._determine_component_type(result.hook_name)
            return f"L{result.layer}_{comp_type}"
        else:
            return result.hook_name.replace(".", "_")
    
    def _determine_component_type(self, hook_name: str) -> str:
        """Determine the component type from hook name."""
        if "attn" in hook_name:
            return "attention"
        elif "mlp" in hook_name:
            return "mlp"
        elif "resid" in hook_name:
            return "residual"
        elif "ln" in hook_name or "norm" in hook_name:
            return "layernorm"
        else:
            return "other"
    
    def _get_activations(
        self,
        tokens: torch.Tensor,
        component_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Get activations for specified components."""
        activations = {}
        
        def store_hook(activation: torch.Tensor, name: str):
            activations[name] = activation.clone().detach()
        
        # Map component names to hook names
        hook_names = []
        for comp_name in component_names:
            if comp_name in self.circuit_components:
                comp = self.circuit_components[comp_name]
                if comp.head is not None:
                    hook_name = f"blocks.{comp.layer}.attn.hook_result"
                else:
                    hook_name = f"blocks.{comp.layer}.{comp.component_type}.hook_post"
                hook_names.append((hook_name, comp_name))
        
        # Add hooks
        hooks = []
        for hook_name, comp_name in hook_names:
            try:
                hook = self.model.add_hook(
                    hook_name,
                    lambda act, name=comp_name: store_hook(act, name)
                )
                hooks.append(hook)
            except:
                logger.warning(f"Could not add hook for {hook_name}")
        
        # Run forward pass
        with torch.no_grad():
            self.model(tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations


def main():
    """Example usage of the CircuitAnalyzer class."""
    from model_setup import ModelSetup
    from prompt_design import PromptGenerator
    from activation_patching import ActivationPatcher
    
    # Setup model
    setup = ModelSetup()
    model = setup.load_model()
    
    # Create test data
    generator = PromptGenerator(seed=42)
    pairs = generator.create_prompt_pairs(n_pairs=3)
    
    # Run activation patching
    patcher = ActivationPatcher(model)
    clean_tokens = model.to_tokens(pairs[0][0].prompt_text)
    corrupted_tokens = model.to_tokens(pairs[0][1].prompt_text)
    
    # Get some patching results (simplified for example)
    results = patcher.patch_attention_heads(
        corrupted_tokens=corrupted_tokens,
        clean_tokens=clean_tokens,
        positions=[-1]
    )
    
    # Analyze circuit
    analyzer = CircuitAnalyzer(model)
    components = analyzer.identify_circuit_components(results)
    
    print(f"Identified {len(components)} circuit components:")
    for name, comp in components.items():
        print(f"  {name}: {comp.importance_score:.3f} ({comp.component_type})")
    
    # Create circuit summary
    summary = analyzer.create_circuit_summary(results)
    print(f"\nCircuit involves {summary['circuit_depth']} layers")
    print(f"Most important layer: {summary['circuit_overview']['most_important_layer']}")


if __name__ == "__main__":
    main()