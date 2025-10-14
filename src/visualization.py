"""
Visualization Module

This module provides visualization tools for the greater than circuit analysis,
including circuit diagrams, attention patterns, and patching results.

Acknowledgment: Visualization approaches inspired by interpretability research
tools developed by Neel Nanda and the broader mechanistic interpretability community.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
from .circuit_analysis import CircuitComponent, CircuitAnalyzer
from .activation_patching import PatchingResult
import logging

logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")


class CircuitVisualizer:
    """
    Provides comprehensive visualization tools for circuit analysis.
    
    This class creates various plots and diagrams to help understand
    the structure and behavior of the identified greater than circuit.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        logger.info(f"CircuitVisualizer initialized with output_dir: {output_dir}")
    
    def plot_patching_results(
        self,
        results: List[PatchingResult],
        title: str = "Activation Patching Results",
        save_path: Optional[str] = None,
        top_k: int = 20
    ) -> plt.Figure:
        """
        Create a bar plot of patching results sorted by effect size.
        
        Args:
            results: List of PatchingResult objects
            title: Plot title
            save_path: Path to save the figure
            top_k: Number of top results to display
            
        Returns:
            Matplotlib figure object
        """
        # Sort results by effect size
        sorted_results = sorted(
            results,
            key=lambda x: abs(x.effect_size),
            reverse=True
        )[:top_k]
        
        # Extract data for plotting
        labels = []
        effect_sizes = []
        colors = []
        
        for result in sorted_results:
            # Create label
            if result.head is not None:
                label = f"L{result.layer}H{result.head}"
                color = 'steelblue'
            elif "mlp" in result.hook_name:
                label = f"L{result.layer}_MLP"
                color = 'coral'
            else:
                label = f"L{result.layer}_Resid"
                color = 'lightgreen'
            
            labels.append(label)
            effect_sizes.append(result.effect_size)
            colors.append(color)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.barh(range(len(labels)), effect_sizes, color=colors)
        
        # Customize plot
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Effect Size', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, effect_sizes)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label='Attention Head'),
            Patch(facecolor='coral', label='MLP'),
            Patch(facecolor='lightgreen', label='Residual')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Patching results plot saved to {save_path}")
        
        return fig
    
    def plot_circuit_diagram(
        self,
        components: Dict[str, CircuitComponent],
        title: str = "Greater Than Circuit Diagram",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive circuit diagram using Plotly.
        
        Args:
            components: Dictionary of circuit components
            title: Diagram title
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure object
        """
        # Organize components by layer
        layers = {}
        for name, comp in components.items():
            if comp.layer not in layers:
                layers[comp.layer] = []
            layers[comp.layer].append((name, comp))
        
        # Create node positions
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        layer_spacing = 2.0
        
        for layer_idx, (layer, comps) in enumerate(sorted(layers.items())):
            x = layer_idx * layer_spacing
            n_comps = len(comps)
            
            for i, (name, comp) in enumerate(comps):
                y = (i - n_comps/2) * 0.5
                
                node_x.append(x)
                node_y.append(y)
                node_text.append(
                    f"{name}<br>"
                    f"Layer: {comp.layer}<br>"
                    f"Type: {comp.component_type}<br>"
                    f"Importance: {comp.importance_score:.3f}"
                )
                
                # Color by component type
                if comp.component_type == "attention":
                    node_color.append('lightblue')
                elif comp.component_type == "mlp":
                    node_color.append('lightcoral')
                else:
                    node_color.append('lightgreen')
                
                # Size by importance
                node_size.append(20 + comp.importance_score * 40)
        
        # Create edges
        edge_x = []
        edge_y = []
        
        for name, comp in components.items():
            source_idx = list(components.keys()).index(name)
            for target_name in comp.connections:
                if target_name in components:
                    target_idx = list(components.keys()).index(target_name)
                    
                    edge_x.extend([node_x[source_idx], node_x[target_idx], None])
                    edge_y.extend([node_y[source_idx], node_y[target_idx], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(color='darkgray', width=2)
            ),
            text=[name for name in components.keys()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Layer"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Circuit diagram saved to {save_path}")
        
        return fig
    
    def plot_attention_patterns(
        self,
        attention_patterns: Dict[str, torch.Tensor],
        token_labels: List[str],
        title: str = "Attention Patterns",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention patterns as heatmaps.
        
        Args:
            attention_patterns: Dictionary mapping component names to attention tensors
            token_labels: Labels for tokens in the sequence
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n_patterns = len(attention_patterns)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_patterns, figsize=(6*n_patterns, 5))
        
        if n_patterns == 1:
            axes = [axes]
        
        for ax, (name, pattern) in zip(axes, attention_patterns.items()):
            # Convert to numpy if tensor
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(pattern, cmap='viridis', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha='right')
            ax.set_yticklabels(token_labels)
            
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention patterns plot saved to {save_path}")
        
        return fig
    
    def plot_layer_importance(
        self,
        layer_contributions: Dict[int, float],
        title: str = "Layer Importance",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a bar plot showing importance of each layer.
        
        Args:
            layer_contributions: Dictionary mapping layer numbers to importance scores
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Sort by layer number
        layers = sorted(layer_contributions.keys())
        importances = [layer_contributions[l] for l in layers]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = ax.bar(layers, importances, color='steelblue', alpha=0.7)
        
        # Highlight most important layers
        max_importance = max(importances)
        for bar, importance in zip(bars, importances):
            if importance >= max_importance * 0.7:
                bar.set_color('coral')
        
        # Customize plot
        ax.set_xlabel('Layer Number', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for layer, importance in zip(layers, importances):
            if importance > 0.01:  # Only label significant bars
                ax.text(layer, importance, f'{importance:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Layer importance plot saved to {save_path}")
        
        return fig
    
    def create_summary_dashboard(
        self,
        circuit_summary: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive dashboard summarizing circuit analysis.
        
        Args:
            circuit_summary: Dictionary containing circuit summary statistics
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Component Type Distribution',
                'Layer Distribution',
                'Top Components by Importance',
                'Circuit Statistics'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ]
        )
        
        # 1. Component type distribution (pie chart)
        if 'component_types' in circuit_summary:
            types = list(circuit_summary['component_types'].keys())
            counts = list(circuit_summary['component_types'].values())
            
            fig.add_trace(
                go.Pie(labels=types, values=counts, name='Component Types'),
                row=1, col=1
            )
        
        # 2. Layer distribution (bar chart)
        if 'layer_distribution' in circuit_summary:
            layers = sorted(circuit_summary['layer_distribution'].keys())
            counts = [circuit_summary['layer_distribution'][l] for l in layers]
            
            fig.add_trace(
                go.Bar(x=layers, y=counts, name='Layer Distribution'),
                row=1, col=2
            )
        
        # 3. Statistics table
        stats_data = [
            ['Total Components', circuit_summary.get('total_components', 0)],
            ['Total Importance', f"{circuit_summary.get('total_importance', 0):.3f}"],
            ['Average Importance', f"{circuit_summary.get('average_importance', 0):.3f}"],
            ['Circuit Depth', circuit_summary.get('circuit_depth', 'N/A')],
        ]
        
        if 'most_important_component' in circuit_summary:
            mic = circuit_summary['most_important_component']
            stats_data.append([
                'Most Important',
                f"{mic.get('name', 'N/A')} ({mic.get('importance', 0):.3f})"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data],
                                  [row[1] for row in stats_data]])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Circuit Analysis Dashboard",
            showlegend=False,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def plot_validation_results(
        self,
        validation_results: Dict[str, Any],
        title: str = "Validation Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization of validation test results.
        
        Args:
            validation_results: Dictionary of validation results
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract test names and accuracies
        test_names = []
        accuracies = []
        
        for test_name, result in validation_results.items():
            test_names.append(test_name.replace('_', ' ').title())
            accuracies.append(result.accuracy)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = ax.barh(range(len(test_names)), accuracies, color='steelblue', alpha=0.7)
        
        # Color code by performance
        for bar, acc in zip(bars, accuracies):
            if acc >= 0.8:
                bar.set_color('green')
            elif acc >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Customize plot
        ax.set_yticks(range(len(test_names)))
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc, i, f' {acc:.3f}', va='center', fontsize=10)
        
        # Add reference line at 0.5
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation results plot saved to {save_path}")
        
        return fig


def main():
    """Example usage of the CircuitVisualizer class."""
    # Create sample data for demonstration
    visualizer = CircuitVisualizer()
    
    # Sample patching results
    from .activation_patching import PatchingResult
    
    sample_results = [
        PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
        PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
        PatchingResult("blocks.7.attn.hook_result", 7, 2, -1, 0.28, 0.25, 0.53, 0.28),
    ]
    
    # Create plots
    fig1 = visualizer.plot_patching_results(sample_results, save_path="results/patching_results.png")
    
    # Sample circuit components
    from .circuit_analysis import CircuitComponent
    
    sample_components = {
        "L5H7": CircuitComponent("L5H7", 5, 7, 0.45, "attention"),
        "L3_mlp": CircuitComponent("L3_mlp", 3, None, 0.32, "mlp"),
        "L7H2": CircuitComponent("L7H2", 7, 2, 0.28, "attention"),
    }
    
    fig2 = visualizer.plot_circuit_diagram(sample_components, save_path="results/circuit_diagram.html")
    
    print("Visualizations created successfully!")


if __name__ == "__main__":
    main()