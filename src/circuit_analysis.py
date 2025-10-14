"""
Circuit Analysis Module

This module provides tools for analyzing and identifying circuits in transformer models
using activation patching results. It includes functionality for component identification,
layer contribution analysis, and circuit structure mapping.

Implements methods described in:
- "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
- "Interpretability in the Wild" (Nanda et al., 2023)

Author: Research implementation based on Neel Nanda's TransformerLens methodology
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from .activation_patching import PatchingResult


@dataclass
class CircuitComponent:
    """
    Represents a component in the identified circuit.
    
    Attributes:
        name: Unique identifier for the component
        layer: Layer number in the model
        head: Attention head number (None for MLP components)
        importance_score: Quantitative measure of component importance
        component_type: Type of component ('attention', 'mlp', 'residual')
        connections: List of connected component names
        metadata: Additional information about the component
    """
    name: str
    layer: int
    head: Optional[int] = None
    importance_score: float = 0.0
    component_type: str = "attention"
    connections: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate component data after initialization."""
        if self.component_type not in ["attention", "mlp", "residual"]:
            raise ValueError(f"Invalid component type: {self.component_type}")
        if self.component_type == "attention" and self.head is None:
            raise ValueError("Attention components must specify a head number")


class CircuitAnalyzer:
    """
    Analyzer for identifying and characterizing circuits in transformer models.
    
    This class processes activation patching results to identify the key components
    and connections that implement specific model capabilities.
    """
    
    def __init__(self, model, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the circuit analyzer.
        
        Args:
            model: The transformer model to analyze
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        
    def identify_circuit_components(
        self,
        patching_results: List[PatchingResult],
        importance_threshold: float = 0.1,
        top_k: Optional[int] = None
    ) -> Dict[str, CircuitComponent]:
        """
        Identify circuit components from patching results.
        
        Args:
            patching_results: List of patching results from experiments
            importance_threshold: Minimum importance score to include component
            top_k: If specified, return only top k components
            
        Returns:
            Dictionary mapping component names to CircuitComponent objects
        """
        components = {}
        
        # Sort results by effect size (descending)
        sorted_results = sorted(
            patching_results,
            key=lambda x: abs(x.effect_size),
            reverse=True
        )
        
        # Apply top_k filter if specified
        if top_k is not None:
            sorted_results = sorted_results[:top_k]
        
        for result in sorted_results:
            # Skip if below importance threshold
            if abs(result.effect_size) < importance_threshold:
                continue
            
            # Determine component type and create name
            if "attn" in result.hook_name:
                component_type = "attention"
                component_name = f"L{result.layer}H{result.head}"
            elif "mlp" in result.hook_name:
                component_type = "mlp"
                component_name = f"L{result.layer}_mlp"
            else:
                component_type = "residual"
                component_name = f"L{result.layer}_resid"
            
            # Create circuit component
            component = CircuitComponent(
                name=component_name,
                layer=result.layer,
                head=result.head,
                importance_score=abs(result.effect_size),
                component_type=component_type,
                metadata={
                    "hook_name": result.hook_name,
                    "position": result.position,
                    "metric_diff": result.metric_diff,
                    "original_metric": result.original_metric,
                    "patched_metric": result.patched_metric
                }
            )
            
            components[component_name] = component
        
        return components
    
    def analyze_layer_contributions(
        self,
        patching_results: List[PatchingResult]
    ) -> Dict[int, float]:
        """
        Analyze the contribution of each layer to the circuit.
        
        Args:
            patching_results: List of patching results
            
        Returns:
            Dictionary mapping layer numbers to aggregate importance scores
        """
        layer_contributions = {i: 0.0 for i in range(self.n_layers)}
        
        for result in patching_results:
            layer_contributions[result.layer] = max(
                layer_contributions[result.layer],
                abs(result.effect_size)
            )
        
        return layer_contributions
    
    def analyze_head_contributions(
        self,
        patching_results: List[PatchingResult],
        layer: Optional[int] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Analyze the contribution of each attention head.
        
        Args:
            patching_results: List of patching results
            layer: If specified, only analyze heads in this layer
            
        Returns:
            Dictionary mapping (layer, head) tuples to importance scores
        """
        head_contributions = {}
        
        for result in patching_results:
            # Skip non-attention components
            if result.head is None:
                continue
            
            # Filter by layer if specified
            if layer is not None and result.layer != layer:
                continue
            
            key = (result.layer, result.head)
            head_contributions[key] = abs(result.effect_size)
        
        return head_contributions
    
    def infer_circuit_connections(
        self,
        components: Dict[str, CircuitComponent],
        attention_patterns: Optional[torch.Tensor] = None
    ) -> Dict[str, CircuitComponent]:
        """
        Infer connections between circuit components.
        
        This uses attention patterns and layer ordering to determine likely
        information flow paths through the circuit.
        
        Args:
            components: Dictionary of identified circuit components
            attention_patterns: Optional attention pattern tensor [layer, head, seq, seq]
            
        Returns:
            Updated components dictionary with connection information
        """
        # Sort components by layer
        sorted_components = sorted(
            components.items(),
            key=lambda x: x[1].layer
        )
        
        # For each component, identify potential downstream connections
        for i, (name1, comp1) in enumerate(sorted_components):
            for name2, comp2 in sorted_components[i+1:]:
                # Components in later layers can receive from earlier layers
                if comp2.layer > comp1.layer:
                    # Add connection if component 2 is important
                    if comp2.importance_score > 0.1:
                        comp1.connections.append(name2)
        
        return components
    
    def get_circuit_summary(
        self,
        components: Dict[str, CircuitComponent]
    ) -> Dict[str, any]:
        """
        Generate a summary of the identified circuit.
        
        Args:
            components: Dictionary of circuit components
            
        Returns:
            Dictionary containing circuit summary statistics
        """
        # Count components by type
        type_counts = {}
        for comp in components.values():
            type_counts[comp.component_type] = type_counts.get(comp.component_type, 0) + 1
        
        # Find layer distribution
        layer_distribution = {}
        for comp in components.values():
            layer_distribution[comp.layer] = layer_distribution.get(comp.layer, 0) + 1
        
        # Calculate total importance
        total_importance = sum(comp.importance_score for comp in components.values())
        
        # Find most important component
        most_important = max(
            components.values(),
            key=lambda x: x.importance_score
        )
        
        return {
            "total_components": len(components),
            "component_types": type_counts,
            "layer_distribution": layer_distribution,
            "total_importance": total_importance,
            "average_importance": total_importance / len(components) if components else 0,
            "most_important_component": {
                "name": most_important.name,
                "layer": most_important.layer,
                "importance": most_important.importance_score
            }
        }
    
    def rank_components_by_importance(
        self,
        components: Dict[str, CircuitComponent],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, CircuitComponent]]:
        """
        Rank circuit components by importance score.
        
        Args:
            components: Dictionary of circuit components
            top_k: If specified, return only top k components
            
        Returns:
            List of (name, component) tuples sorted by importance
        """
        ranked = sorted(
            components.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def filter_components_by_layer(
        self,
        components: Dict[str, CircuitComponent],
        min_layer: int = 0,
        max_layer: Optional[int] = None
    ) -> Dict[str, CircuitComponent]:
        """
        Filter components by layer range.
        
        Args:
            components: Dictionary of circuit components
            min_layer: Minimum layer number (inclusive)
            max_layer: Maximum layer number (inclusive), None for no upper limit
            
        Returns:
            Filtered dictionary of components
        """
        if max_layer is None:
            max_layer = self.n_layers - 1
        
        return {
            name: comp
            for name, comp in components.items()
            if min_layer <= comp.layer <= max_layer
        }
    
    def export_circuit_graph(
        self,
        components: Dict[str, CircuitComponent],
        output_format: str = "dict"
    ) -> Dict:
        """
        Export circuit structure in a format suitable for visualization.
        
        Args:
            components: Dictionary of circuit components
            output_format: Format for export ('dict', 'networkx', etc.)
            
        Returns:
            Circuit graph in specified format
        """
        if output_format != "dict":
            raise NotImplementedError(f"Format {output_format} not yet supported")
        
        # Create nodes list
        nodes = []
        for name, comp in components.items():
            nodes.append({
                "id": name,
                "label": name,
                "layer": comp.layer,
                "head": comp.head,
                "type": comp.component_type,
                "importance": comp.importance_score
            })
        
        # Create edges list
        edges = []
        for name, comp in components.items():
            for target in comp.connections:
                edges.append({
                    "source": name,
                    "target": target,
                    "type": "information_flow"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": self.get_circuit_summary(components)
        }