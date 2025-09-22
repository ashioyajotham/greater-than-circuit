"""
Activation Patching Module

* Activation patching refers to the technique of intervening in the activations of a neural network
to understand the role of specific components in the network's behavior.
* While activation itself means the output of a neuron or layer after applying an activation function,
activation patching involves replacing or modifying these activations during inference to observe their impact on the model's behavior.

This module implements activation patching techniques to isolate and analyze
the components of the greater than circuit in GPT-2 Small.

Acknowledgment: Activation patching methodology developed and refined by
Neel Nanda and the mechanistic interpretability research community.
"""

import torch
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import logging

logger = logging.getLogger(__name__)


@dataclass 
class PatchingResult:
    """
    Data class for storing activation patching results.
    
    Attributes:
        hook_name (str): Name of the hook that was patched
        layer (int): Layer number (if applicable)
        head (int): Attention head number (if applicable)  
        position (int): Token position that was patched
        metric_diff (float): Difference in metric after patching
        original_metric (float): Original metric value
        patched_metric (float): Metric value after patching
        effect_size (float): Normalized effect size
    """
    hook_name: str
    layer: Optional[int] = None
    head: Optional[int] = None
    position: Optional[int] = None
    metric_diff: float = 0.0
    original_metric: float = 0.0
    patched_metric: float = 0.0
    effect_size: float = 0.0


class ActivationPatcher:
    """
    Implements activation patching for mechanistic interpretability analysis.
    
    This class provides functionality to patch activations at different layers
    and positions to identify which components are crucial for the greater than circuit.
    """
    
    def __init__(self, model: HookedTransformer):
        """
        Initialize the ActivationPatcher.
        
        Args:
            model (HookedTransformer): The model to perform patching on
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.stored_activations = {}
        
        logger.info(f"ActivationPatcher initialized for {model.cfg.model_name}")
    
    def store_activations(
        self,
        tokens: torch.Tensor,
        hook_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Store activations from a forward pass for later patching.
        
        Args:
            tokens (torch.Tensor): Input tokens
            hook_names (List[str]): Names of hooks to store activations from
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping hook names to activations
        """
        stored_acts = {}
        
        def store_hook(activation: torch.Tensor, hook_name: str):
            stored_acts[hook_name] = activation.clone().detach()
        
        # Add hooks to store activations
        hooks = []
        for hook_name in hook_names:
            hook = self.model.add_hook(
                hook_name,
                lambda act, hook=hook_name: store_hook(act, hook)
            )
            hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            self.model(tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        self.stored_activations.update(stored_acts)
        logger.info(f"Stored activations for {len(hook_names)} hooks")
        
        return stored_acts
    
    def patch_activation(
        self,
        corrupted_tokens: torch.Tensor,
        clean_tokens: torch.Tensor,
        hook_name: str,
        position: Optional[int] = None,
        metric_fn: Optional[Callable] = None
    ) -> PatchingResult:
        """
        Patch a single activation and measure the effect.
        
        Args:
            corrupted_tokens (torch.Tensor): Corrupted input tokens
            clean_tokens (torch.Tensor): Clean input tokens  
            hook_name (str): Name of the hook to patch
            position (int, optional): Token position to patch (if None, patch all)
            metric_fn (Callable, optional): Function to compute metric
            
        Returns:
            PatchingResult: Results of the patching experiment
        """
        if metric_fn is None:
            metric_fn = self._default_metric_fn
        
        # Store clean activations
        clean_acts = self.store_activations(clean_tokens, [hook_name])
        clean_activation = clean_acts[hook_name]
        
        # Get original (corrupted) metric
        with torch.no_grad():
            original_logits = self.model(corrupted_tokens)
            original_metric = metric_fn(original_logits, corrupted_tokens)
        
        # Define patching hook
        def patching_hook(activation: torch.Tensor, hook_name: str = hook_name):
            if position is not None:
                # Patch specific position
                activation[:, position, :] = clean_activation[:, position, :].to(activation.device)
            else:
                # Patch all positions
                activation[:] = clean_activation.to(activation.device)
            return activation
        
        # Apply patch and measure effect
        hook = self.model.add_hook(hook_name, patching_hook)
        
        with torch.no_grad():
            patched_logits = self.model(corrupted_tokens)
            patched_metric = metric_fn(patched_logits, corrupted_tokens)
        
        hook.remove()
        
        # Calculate effect metrics
        metric_diff = patched_metric - original_metric
        effect_size = metric_diff / (abs(original_metric) + 1e-8)
        
        # Parse layer and head from hook name if applicable
        layer, head = self._parse_hook_name(hook_name)
        
        result = PatchingResult(
            hook_name=hook_name,
            layer=layer,
            head=head,
            position=position,
            metric_diff=metric_diff,
            original_metric=original_metric,
            patched_metric=patched_metric,
            effect_size=effect_size
        )
        
        return result
    
    def comprehensive_patching(
        self,
        corrupted_tokens: torch.Tensor,
        clean_tokens: torch.Tensor,
        component_types: List[str] = ["attn", "mlp", "resid"],
        positions: Optional[List[int]] = None,
        metric_fn: Optional[Callable] = None
    ) -> List[PatchingResult]:
        """
        Perform comprehensive activation patching across multiple components.
        
        Args:
            corrupted_tokens (torch.Tensor): Corrupted input tokens
            clean_tokens (torch.Tensor): Clean input tokens
            component_types (List[str]): Types of components to patch
            positions (List[int], optional): Specific positions to test
            metric_fn (Callable, optional): Function to compute metric
            
        Returns:
            List[PatchingResult]: Results from all patching experiments
        """
        results = []
        
        # Get all hook names for specified components
        hook_names = self._get_hook_names(component_types)
        
        if positions is None:
            # Test all token positions
            seq_len = corrupted_tokens.shape[1]
            positions = list(range(seq_len))
        
        logger.info(f"Starting comprehensive patching: {len(hook_names)} hooks × {len(positions)} positions")
        
        # Store all clean activations at once
        logger.info("Storing clean activations...")
        self.store_activations(clean_tokens, hook_names)
        
        # Patch each hook at each position
        for i, hook_name in enumerate(hook_names):
            if i % 10 == 0:
                logger.info(f"Processing hook {i+1}/{len(hook_names)}: {hook_name}")
            
            for position in positions:
                try:
                    result = self.patch_activation(
                        corrupted_tokens=corrupted_tokens,
                        clean_tokens=clean_tokens,
                        hook_name=hook_name,
                        position=position,
                        metric_fn=metric_fn
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to patch {hook_name} at position {position}: {e}")
                    continue
        
        logger.info(f"Completed comprehensive patching: {len(results)} results")
        return results
    
    def patch_attention_heads(
        self,
        corrupted_tokens: torch.Tensor,
        clean_tokens: torch.Tensor,
        positions: Optional[List[int]] = None,
        metric_fn: Optional[Callable] = None
    ) -> List[PatchingResult]:
        """
        Specifically patch attention head outputs.
        
        Args:
            corrupted_tokens (torch.Tensor): Corrupted input tokens
            clean_tokens (torch.Tensor): Clean input tokens
            positions (List[int], optional): Specific positions to test
            metric_fn (Callable, optional): Function to compute metric
            
        Returns:
            List[PatchingResult]: Results from attention head patching
        """
        results = []
        
        if positions is None:
            positions = list(range(corrupted_tokens.shape[1]))
        
        logger.info(f"Patching attention heads across {self.model.cfg.n_layers} layers")
        
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                hook_name = get_act_name("result", layer, head)
                
                for position in positions:
                    try:
                        result = self.patch_activation(
                            corrupted_tokens=corrupted_tokens,
                            clean_tokens=clean_tokens,
                            hook_name=hook_name,
                            position=position,
                            metric_fn=metric_fn
                        )
                        result.layer = layer
                        result.head = head
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Failed to patch head {layer}.{head} at position {position}: {e}")
                        continue
        
        return results
    
    def find_critical_components(
        self,
        results: List[PatchingResult],
        threshold: float = 0.1,
        top_k: int = 10
    ) -> List[PatchingResult]:
        """
        Identify the most critical components based on patching results.
        
        Args:
            results (List[PatchingResult]): Patching results to analyze
            threshold (float): Minimum effect size to consider critical
            top_k (int): Number of top components to return
            
        Returns:
            List[PatchingResult]: Top critical components
        """
        # Filter by threshold
        critical = [r for r in results if abs(r.effect_size) >= threshold]
        
        # Sort by absolute effect size
        critical.sort(key=lambda x: abs(x.effect_size), reverse=True)
        
        # Return top k
        top_critical = critical[:top_k]
        
        logger.info(f"Found {len(critical)} components above threshold, returning top {len(top_critical)}")
        
        return top_critical
    
    def _default_metric_fn(self, logits: torch.Tensor, tokens: torch.Tensor) -> float:
        """
        Default metric function: probability of correct answer.
        
        Args:
            logits (torch.Tensor): Model logits
            tokens (torch.Tensor): Input tokens
            
        Returns:
            float: Probability of correct answer
        """
        # Get logits for the last token position
        last_logits = logits[0, -1, :]
        
        # Get probabilities
        probs = torch.softmax(last_logits, dim=-1)
        
        # Find tokens for "True" and "False"
        true_token = self.model.to_single_token(" True")
        false_token = self.model.to_single_token(" False")
        
        # Return probability of "True" token (assuming that's what we want)
        return probs[true_token].item()
    
    def _get_hook_names(self, component_types: List[str]) -> List[str]:
        """Get hook names for specified component types."""
        hook_names = []
        
        for layer in range(self.model.cfg.n_layers):
            for comp_type in component_types:
                if comp_type == "attn":
                    # Attention output
                    hook_names.append(get_act_name("attn_out", layer))
                elif comp_type == "mlp":
                    # MLP output  
                    hook_names.append(get_act_name("mlp_out", layer))
                elif comp_type == "resid":
                    # Residual stream
                    hook_names.extend([
                        get_act_name("resid_pre", layer),
                        get_act_name("resid_mid", layer),
                        get_act_name("resid_post", layer)
                    ])
                elif comp_type == "attn_heads":
                    # Individual attention heads
                    for head in range(self.model.cfg.n_heads):
                        hook_names.append(get_act_name("result", layer, head))
        
        return hook_names
    
    def _parse_hook_name(self, hook_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse layer and head numbers from hook name."""
        layer = None
        head = None
        
        # Extract layer number
        if "blocks." in hook_name:
            try:
                layer_str = hook_name.split("blocks.")[1].split(".")[0]
                layer = int(layer_str)
            except (IndexError, ValueError):
                pass
        
        # Extract head number (for attention heads)
        if "head_" in hook_name or ".head." in hook_name:
            try:
                # Different patterns for head specification
                if ".head." in hook_name:
                    head_str = hook_name.split(".head.")[1].split(".")[0]
                elif "head_" in hook_name:
                    head_str = hook_name.split("head_")[1].split(".")[0]
                head = int(head_str)
            except (IndexError, ValueError):
                pass
        
        return layer, head
    
    def analyze_position_effects(
        self,
        results: List[PatchingResult]
    ) -> Dict[int, float]:
        """
        Analyze the effect of patching different token positions.
        
        Args:
            results (List[PatchingResult]): Patching results to analyze
            
        Returns:
            Dict[int, float]: Average effect size by position
        """
        position_effects = {}
        
        for result in results:
            if result.position is not None:
                if result.position not in position_effects:
                    position_effects[result.position] = []
                position_effects[result.position].append(abs(result.effect_size))
        
        # Calculate averages
        avg_effects = {
            pos: np.mean(effects) 
            for pos, effects in position_effects.items()
        }
        
        return avg_effects
    
    def summarize_results(self, results: List[PatchingResult]) -> Dict[str, Any]:
        """
        Generate a summary of patching results.
        
        Args:
            results (List[PatchingResult]): Results to summarize
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}
        
        effect_sizes = [abs(r.effect_size) for r in results]
        
        summary = {
            "total_experiments": len(results),
            "mean_effect_size": np.mean(effect_sizes),
            "max_effect_size": np.max(effect_sizes),
            "min_effect_size": np.min(effect_sizes),
            "std_effect_size": np.std(effect_sizes),
            "significant_effects": len([e for e in effect_sizes if e > 0.1]),
            "position_effects": self.analyze_position_effects(results),
            "top_components": [
                {
                    "hook_name": r.hook_name,
                    "layer": r.layer,
                    "head": r.head,
                    "position": r.position,
                    "effect_size": r.effect_size
                }
                for r in sorted(results, key=lambda x: abs(x.effect_size), reverse=True)[:5]
            ]
        }
        
        return summary


def main():
    """Example usage of the ActivationPatcher class."""
    from model_setup import ModelSetup
    from prompt_design import PromptGenerator
    
    # Setup model
    setup = ModelSetup()
    model = setup.load_model()
    
    # Create prompt examples
    generator = PromptGenerator(seed=42)
    pairs = generator.create_prompt_pairs(n_pairs=5)
    
    # Get clean and corrupted examples
    clean_example, corrupted_example = pairs[0]
    
    # Tokenize
    clean_tokens = model.to_tokens(clean_example.prompt_text)
    corrupted_tokens = model.to_tokens(corrupted_example.prompt_text)
    
    # Initialize patcher
    patcher = ActivationPatcher(model)
    
    # Test single patch
    result = patcher.patch_activation(
        corrupted_tokens=corrupted_tokens,
        clean_tokens=clean_tokens,
        hook_name="blocks.0.attn.hook_result",
        position=-1  # Last token position
    )
    
    print(f"Patching result: {result.hook_name} -> {result.effect_size:.3f}")
    
    # Test attention head patching
    head_results = patcher.patch_attention_heads(
        corrupted_tokens=corrupted_tokens,
        clean_tokens=clean_tokens,
        positions=[-1]  # Only last position
    )
    
    # Find critical components
    critical = patcher.find_critical_components(head_results, threshold=0.05, top_k=5)
    
    print(f"\nTop {len(critical)} critical attention heads:")
    for result in critical:
        print(f"  Layer {result.layer}, Head {result.head}: {result.effect_size:.3f}")


if __name__ == "__main__":
    main()