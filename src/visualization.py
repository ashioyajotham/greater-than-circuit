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
from circuit_analysis import CircuitComponent, CircuitAnalyzer
from activation_patching import PatchingResult
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
        Initialize the CircuitVisualizer.
        
        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.colors = {
            "attention": "#FF6B6B",
            "mlp": "#4ECDC4", 
            "residual": "#45B7D1",
            "layernorm": "#FFA726",
            "other": "#9575CD"
        }
        
        logger.info(f"CircuitVisualizer initialized, output dir: {output_dir}")
    
    def plot_patching_results(
        self,
        results: List[PatchingResult],
        title: str = "Activation Patching Results",
        save_path: Optional[str] = None,
        top_k: int = 20
    ) -> plt.Figure:
        """
        Create a comprehensive plot of activation patching results.
        
        Args:
            results (List[PatchingResult]): Patching results to plot
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            top_k (int): Number of top results to show
            
        Returns:
            plt.Figure: The created figure
        """
        # Sort results by absolute effect size
        sorted_results = sorted(results, key=lambda x: abs(x.effect_size), reverse=True)
        top_results = sorted_results[:top_k]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Top components bar plot
        component_names = [self._format_component_name(r) for r in top_results]
        effect_sizes = [r.effect_size for r in top_results]
        colors = [self._get_component_color(r) for r in top_results]
        
        bars = ax1.barh(range(len(component_names)), effect_sizes, color=colors)
        ax1.set_yticks(range(len(component_names)))
        ax1.set_yticklabels(component_names, fontsize=8)
        ax1.set_xlabel('Effect Size')
        ax1.set_title(f'Top {top_k} Components by Effect Size')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, effect_sizes)):
            ax1.text(value + 0.001, i, f'{value:.3f}', 
                    va='center', fontsize=8)
        
        # 2. Effect size distribution
        all_effect_sizes = [abs(r.effect_size) for r in results]
        ax2.hist(all_effect_sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_effect_sizes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_effect_sizes):.3f}')
        ax2.set_xlabel('Absolute Effect Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Effect Sizes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Layer-wise effects
        layer_effects = {}
        for r in results:
            if r.layer is not None:
                if r.layer not in layer_effects:
                    layer_effects[r.layer] = []
                layer_effects[r.layer].append(abs(r.effect_size))
        
        if layer_effects:
            layers = sorted(layer_effects.keys())
            avg_effects = [np.mean(layer_effects[layer]) for layer in layers]
            max_effects = [np.max(layer_effects[layer]) for layer in layers]
            
            x = np.arange(len(layers))
            width = 0.35
            
            ax3.bar(x - width/2, avg_effects, width, label='Average', alpha=0.8)
            ax3.bar(x + width/2, max_effects, width, label='Maximum', alpha=0.8)
            
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Effect Size')
            ax3.set_title('Layer-wise Effect Analysis')
            ax3.set_xticks(x)
            ax3.set_xticklabels(layers)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Position-wise effects
        position_effects = {}
        for r in results:
            if r.position is not None:
                if r.position not in position_effects:
                    position_effects[r.position] = []
                position_effects[r.position].append(abs(r.effect_size))
        
        if position_effects:
            positions = sorted(position_effects.keys())
            avg_pos_effects = [np.mean(position_effects[pos]) for pos in positions]
            
            ax4.plot(positions, avg_pos_effects, 'o-', linewidth=2, markersize=6)
            ax4.set_xlabel('Token Position')
            ax4.set_ylabel('Average Effect Size')
            ax4.set_title('Position-wise Effect Analysis')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved patching results plot to {save_path}")
        
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
            components (Dict[str, CircuitComponent]): Circuit components to visualize
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        # Organize components by layer
        layers = {}
        for comp in components.values():
            layer = comp.layer
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(comp)
        
        # Create node positions
        node_positions = {}
        y_spacing = 1.0
        x_spacing = 2.0
        
        for layer_idx, (layer_num, layer_comps) in enumerate(sorted(layers.items())):
            x = layer_idx * x_spacing
            
            # Arrange components vertically within layer
            n_comps = len(layer_comps)
            for comp_idx, comp in enumerate(layer_comps):
                y = (comp_idx - n_comps/2) * y_spacing
                node_positions[comp.name] = (x, y)
        
        # Create edges (simplified - in practice would need more sophisticated connection analysis)
        edges = []
        for layer_idx in range(len(layers) - 1):
            current_layer = list(layers.values())[layer_idx]
            next_layer = list(layers.values())[layer_idx + 1]
            
            for curr_comp in current_layer:
                for next_comp in next_layer:
                    # Add edge with weight based on importance
                    weight = (curr_comp.importance_score + next_comp.importance_score) / 2
                    edges.append((curr_comp.name, next_comp.name, weight))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        for source, target, weight in edges:
            if source in node_positions and target in node_positions:
                x0, y0 = node_positions[source]
                x1, y1 = node_positions[target]
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=weight*10, color='gray'),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Add nodes
        for comp in components.values():
            if comp.name in node_positions:
                x, y = node_positions[comp.name]
                color = self.colors.get(comp.component_type, self.colors["other"])
                
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=comp.importance_score * 100 + 10,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    text=[comp.name],
                    textposition="middle center",
                    textfont=dict(size=10, color='white'),
                    hovertemplate=f"<b>{comp.name}</b><br>" +
                                f"Layer: {comp.layer}<br>" +
                                f"Type: {comp.component_type}<br>" +
                                f"Importance: {comp.importance_score:.3f}<extra></extra>",
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Layer", showgrid=True),
            yaxis=dict(title="Component Position", showgrid=True),
            showlegend=False,
            hovermode='closest',
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved circuit diagram to {save_path}")
        
        return fig
    
    def plot_attention_patterns(
        self,
        attention_patterns: Dict[str, torch.Tensor],
        token_labels: List[str],
        title: str = "Attention Patterns",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention patterns for important heads.
        
        Args:
            attention_patterns (Dict[str, torch.Tensor]): Attention patterns by head
            token_labels (List[str]): Labels for tokens
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        n_heads = len(attention_patterns)
        if n_heads == 0:
            logger.warning("No attention patterns to plot")
            return None
        
        # Calculate grid dimensions
        n_cols = min(3, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_heads == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for idx, (head_name, pattern) in enumerate(attention_patterns.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Convert to numpy and handle different tensor shapes
            if isinstance(pattern, torch.Tensor):
                pattern_np = pattern.detach().cpu().numpy()
            else:
                pattern_np = pattern
            
            # Create heatmap
            im = ax.imshow(pattern_np, cmap='Blues', aspect='auto')
            
            # Set labels
            ax.set_title(f'{head_name}', fontweight='bold')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Set tick labels if provided
            if token_labels and len(token_labels) == pattern_np.shape[0]:
                ax.set_xticks(range(len(token_labels)))
                ax.set_yticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=45, ha='right')
                ax.set_yticklabels(token_labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention patterns to {save_path}")
        
        return fig
    
    def plot_layer_importance(
        self,
        layer_contributions: Dict[int, float],
        title: str = "Layer Importance Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the importance of different layers in the circuit.
        
        Args:
            layer_contributions (Dict[int, float]): Layer importance scores
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if not layer_contributions:
            logger.warning("No layer contributions to plot")
            return None
        
        layers = sorted(layer_contributions.keys())
        contributions = [layer_contributions[layer] for layer in layers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Bar plot
        bars = ax1.bar(layers, contributions, color='lightcoral', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('Average Importance Score')
        ax1.set_title('Layer Contributions to Circuit')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, contributions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Cumulative plot
        cumulative = np.cumsum(contributions) / np.sum(contributions)
        ax2.plot(layers, cumulative, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Cumulative Importance (Normalized)')
        ax2.set_title('Cumulative Circuit Importance')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add horizontal lines for reference
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%')
        ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='80%')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved layer importance plot to {save_path}")
        
        return fig
    
    def create_summary_dashboard(
        self,
        circuit_summary: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive summary dashboard of the circuit analysis.
        
        Args:
            circuit_summary (Dict[str, Any]): Circuit analysis summary
            save_path (str, optional): Path to save the dashboard
            
        Returns:
            go.Figure: Interactive Plotly dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Component Types Distribution',
                'Layer Contributions',
                'Positional Effects',
                'Circuit Statistics'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Component types pie chart
        attn_vs_mlp = circuit_summary.get('attention_vs_mlp', {})
        labels = list(attn_vs_mlp.keys())
        values = list(attn_vs_mlp.values())
        
        fig.add_trace(
            go.Pie(labels=labels, values=values, name="Component Types"),
            row=1, col=1
        )
        
        # 2. Layer contributions bar chart
        layer_analysis = circuit_summary.get('layer_analysis', {})
        if layer_analysis:
            layers = list(layer_analysis.keys())
            contributions = list(layer_analysis.values())
            
            fig.add_trace(
                go.Bar(x=layers, y=contributions, name="Layer Contributions"),
                row=1, col=2
            )
        
        # 3. Positional effects scatter plot
        pos_analysis = circuit_summary.get('positional_analysis', {})
        if pos_analysis:
            positions = list(pos_analysis.keys())
            avg_effects = [data.get('avg_effect', 0) for data in pos_analysis.values()]
            
            fig.add_trace(
                go.Scatter(x=positions, y=avg_effects, mode='lines+markers',
                          name="Positional Effects"),
                row=2, col=1
            )
        
        # 4. Statistics table
        overview = circuit_summary.get('circuit_overview', {})
        stats_data = [
            ['Total Components', overview.get('total_components', 'N/A')],
            ['Circuit Depth', circuit_summary.get('circuit_depth', 'N/A')],
            ['Most Important Layer', overview.get('most_important_layer', 'N/A')],
            ['Attention Heads', len(overview.get('attention_heads', []))],
            ['MLP Components', len(overview.get('mlp_components', []))]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*stats_data)))
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Greater Than Circuit Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved summary dashboard to {save_path}")
        
        return fig
    
    def _format_component_name(self, result: PatchingResult) -> str:
        """Format component name for display."""
        if result.head is not None:
            return f"L{result.layer}H{result.head}"
        elif result.layer is not None:
            comp_type = "MLP" if "mlp" in result.hook_name else "ATN"
            return f"L{result.layer}_{comp_type}"
        else:
            return result.hook_name.split('.')[-1]
    
    def _get_component_color(self, result: PatchingResult) -> str:
        """Get color for component based on type."""
        if "attn" in result.hook_name:
            return self.colors["attention"]
        elif "mlp" in result.hook_name:
            return self.colors["mlp"]
        elif "resid" in result.hook_name:
            return self.colors["residual"]
        else:
            return self.colors["other"]


def main():
    """Example usage of the CircuitVisualizer class."""
    # Create sample data for demonstration
    visualizer = CircuitVisualizer()
    
    # Sample patching results
    from activation_patching import PatchingResult
    
    sample_results = [
        PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
        PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
        PatchingResult("blocks.7.attn.hook_result", 7, 2, -1, 0.28, 0.25, 0.53, 0.28),
    ]
    
    # Create plots
    fig1 = visualizer.plot_patching_results(sample_results, save_path="results/patching_results.png")
    
    # Sample circuit components
    from circuit_analysis import CircuitComponent
    
    sample_components = {
        "L5H7": CircuitComponent("L5H7", 5, 7, 0.45, "attention"),
        "L3_mlp": CircuitComponent("L3_mlp", 3, None, 0.32, "mlp"),
        "L7H2": CircuitComponent("L7H2", 7, 2, 0.28, "attention"),
    }
    
    fig2 = visualizer.plot_circuit_diagram(sample_components, save_path="results/circuit_diagram.html")
    
    print("Visualization examples completed!")
    plt.show()


if __name__ == "__main__":
    main()