"""
Run Greater Than Circuit Analysis - Hanna et al. Methodology

This script implements the exact methodology from:
"How does GPT-2 compute greater-than?: Interpreting mathematical abilities 
in a pre-trained language model" (Hanna et al., NeurIPS 2023)

Reference: https://arxiv.org/abs/2305.00586
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prompt_design_hanna import (
    YearPromptGenerator,
    get_year_token_ids,
    compute_probability_difference,
    create_pd_metric_fn,
    YearPromptExample
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_name: str = "gpt2-small", device: str = "cpu"):
    """Load GPT-2 Small with TransformerLens."""
    logger.info(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.set_use_attn_result(True)
    logger.info(f"Model loaded on {device}")
    return model


def compute_baseline_pd(model, examples, year_token_ids):
    """Compute baseline Probability Difference across examples."""
    pds = []
    
    for example in examples:
        tokens = model.to_tokens(example.prompt_text)
        with torch.no_grad():
            logits = model(tokens)
        
        final_logits = logits[0, -1, :]
        pd = compute_probability_difference(final_logits, example.start_year, year_token_ids)
        pds.append(pd)
    
    return np.mean(pds), np.std(pds), pds


def patch_layer_attention(
    model,
    clean_tokens,
    corrupted_tokens,
    layer: int,
    metric_fn,
    position: int = -1
):
    """
    Patch entire attention layer output from corrupted to clean run.
    
    Args:
        model: HookedTransformer
        clean_tokens: Tokens for clean prompt
        corrupted_tokens: Tokens for corrupted prompt
        layer: Layer to patch
        metric_fn: Metric function (returns PD)
        position: Position to patch (-1 for last)
        
    Returns:
        Effect size and metric values
    """
    # Run clean forward pass to get baseline metric
    with torch.no_grad():
        clean_logits = model(clean_tokens)
    clean_metric = metric_fn(clean_logits).item()
    
    # Cache activations from clean run
    clean_cache = {}
    
    def cache_hook(activation, hook):
        clean_cache[hook.name] = activation.clone()
        return activation
    
    hook_name = get_act_name("attn_out", layer)
    
    # Run clean with caching
    with torch.no_grad():
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, cache_hook)]
        )
    
    # Run corrupted and patch in clean activations
    def patch_hook(activation, hook):
        # Patch at specified position
        if position == -1:
            activation[:, -1, :] = clean_cache[hook.name][:, -1, :]
        else:
            activation[:, position, :] = clean_cache[hook.name][:, position, :]
        return activation
    
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )
    
    patched_metric = metric_fn(patched_logits).item()
    
    # Run corrupted baseline
    with torch.no_grad():
        corrupted_logits = model(corrupted_tokens)
    corrupted_metric = metric_fn(corrupted_logits).item()
    
    # Effect size: how much patching moves toward clean behavior
    total_effect = clean_metric - corrupted_metric
    recovered_effect = patched_metric - corrupted_metric
    
    effect_size = recovered_effect / total_effect if abs(total_effect) > 1e-6 else 0.0
    
    return {
        'layer': layer,
        'component': f'attn_L{layer}',
        'effect_size': effect_size,
        'clean_metric': clean_metric,
        'corrupted_metric': corrupted_metric,
        'patched_metric': patched_metric,
        'total_effect': total_effect,
        'recovered_effect': recovered_effect
    }


def patch_layer_mlp(
    model,
    clean_tokens,
    corrupted_tokens,
    layer: int,
    metric_fn,
    position: int = -1
):
    """
    Patch MLP layer output from corrupted to clean run.
    """
    # Run clean forward pass to get baseline metric
    with torch.no_grad():
        clean_logits = model(clean_tokens)
    clean_metric = metric_fn(clean_logits).item()
    
    # Cache activations from clean run
    clean_cache = {}
    
    def cache_hook(activation, hook):
        clean_cache[hook.name] = activation.clone()
        return activation
    
    hook_name = get_act_name("mlp_out", layer)
    
    # Run clean with caching
    with torch.no_grad():
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, cache_hook)]
        )
    
    # Run corrupted and patch in clean activations
    def patch_hook(activation, hook):
        if position == -1:
            activation[:, -1, :] = clean_cache[hook.name][:, -1, :]
        else:
            activation[:, position, :] = clean_cache[hook.name][:, position, :]
        return activation
    
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )
    
    patched_metric = metric_fn(patched_logits).item()
    
    # Run corrupted baseline
    with torch.no_grad():
        corrupted_logits = model(corrupted_tokens)
    corrupted_metric = metric_fn(corrupted_logits).item()
    
    # Effect size calculation
    total_effect = clean_metric - corrupted_metric
    recovered_effect = patched_metric - corrupted_metric
    
    effect_size = recovered_effect / total_effect if abs(total_effect) > 1e-6 else 0.0
    
    return {
        'layer': layer,
        'component': f'MLP{layer}',
        'effect_size': effect_size,
        'clean_metric': clean_metric,
        'corrupted_metric': corrupted_metric,
        'patched_metric': patched_metric,
        'total_effect': total_effect,
        'recovered_effect': recovered_effect
    }


def run_circuit_analysis(
    n_examples: int = 50,
    seed: int = 42,
    device: str = "cpu"
):
    """
    Run the full circuit analysis using Hanna et al. methodology.
    
    Args:
        n_examples: Number of examples to use
        seed: Random seed
        device: Device to run on
        
    Returns:
        Dictionary with all results
    """
    print("=" * 70)
    print("GREATER THAN CIRCUIT ANALYSIS - Hanna et al. Methodology")
    print("=" * 70)
    
    # Load model
    print("\n[STEP 1] Loading GPT-2 Small...")
    model = load_model("gpt2-small", device)
    
    # Get year token IDs
    print("\n[STEP 2] Getting year token IDs...")
    year_token_ids = get_year_token_ids(model)
    print(f"  Found {len(year_token_ids)} single-token years")
    
    # Generate prompts
    print("\n[STEP 3] Generating year completion prompts...")
    generator = YearPromptGenerator(seed=seed)
    
    # Generate balanced examples
    examples = generator.generate_balanced_year_dataset(n_examples, template_idx=0)
    
    print(f"  Generated {len(examples)} examples")
    print(f"  Example prompt: '{examples[0].prompt_text}'")
    print(f"  Start year: {examples[0].start_year:02d}")
    
    # Compute baseline PD
    print("\n[STEP 4] Computing baseline Probability Difference...")
    avg_pd, std_pd, all_pds = compute_baseline_pd(model, examples, year_token_ids)
    print(f"  Average PD: {avg_pd:.4f} (+/- {std_pd:.4f})")
    print(f"  Min PD: {min(all_pds):.4f}, Max PD: {max(all_pds):.4f}")
    
    if avg_pd > 0.5:
        print("  >>> GPT-2 performs WELL on year completion task!")
    elif avg_pd > 0:
        print("  >>> GPT-2 shows positive bias toward correct years")
    else:
        print("  >>> GPT-2 does NOT perform greater-than on this task")
    
    # Generate prompt pairs
    print("\n[STEP 5] Creating prompt pairs for activation patching...")
    pairs = generator.create_prompt_pairs(min(n_examples, 20), template_idx=0)
    print(f"  Created {len(pairs)} prompt pairs")
    
    # Run activation patching
    print("\n[STEP 6] Running activation patching experiment...")
    
    n_layers = model.cfg.n_layers
    attention_results = []
    mlp_results = []
    
    for pair_idx, (clean_ex, corrupt_ex) in enumerate(pairs):
        print(f"  Processing pair {pair_idx + 1}/{len(pairs)}...")
        
        clean_tokens = model.to_tokens(clean_ex.prompt_text)
        corrupt_tokens = model.to_tokens(corrupt_ex.prompt_text)
        
        # Create metric function for this example
        metric_fn = create_pd_metric_fn(model, year_token_ids, clean_ex.start_year)
        
        # Patch each attention layer
        for layer in range(n_layers):
            result = patch_layer_attention(
                model, clean_tokens, corrupt_tokens, layer, metric_fn
            )
            result['pair_idx'] = pair_idx
            attention_results.append(result)
        
        # Patch each MLP layer
        for layer in range(n_layers):
            result = patch_layer_mlp(
                model, clean_tokens, corrupt_tokens, layer, metric_fn
            )
            result['pair_idx'] = pair_idx
            mlp_results.append(result)
    
    # Aggregate results by layer
    print("\n[STEP 7] Analyzing results...")
    
    # Average attention effects by layer
    attn_by_layer = {}
    for r in attention_results:
        layer = r['layer']
        if layer not in attn_by_layer:
            attn_by_layer[layer] = []
        attn_by_layer[layer].append(r['effect_size'])
    
    attn_avg = {l: np.mean(effects) for l, effects in attn_by_layer.items()}
    
    # Average MLP effects by layer
    mlp_by_layer = {}
    for r in mlp_results:
        layer = r['layer']
        if layer not in mlp_by_layer:
            mlp_by_layer[layer] = []
        mlp_by_layer[layer].append(r['effect_size'])
    
    mlp_avg = {l: np.mean(effects) for l, effects in mlp_by_layer.items()}
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n--- Attention Layer Effects (avg across examples) ---")
    for layer in sorted(attn_avg.keys()):
        effect = attn_avg[layer]
        bar = "█" * int(abs(effect) * 50)
        sign = "+" if effect >= 0 else "-"
        print(f"  Layer {layer:2d}: {sign}{abs(effect):.4f} {bar}")
    
    print("\n--- MLP Layer Effects (avg across examples) ---")
    for layer in sorted(mlp_avg.keys()):
        effect = mlp_avg[layer]
        bar = "█" * int(abs(effect) * 50)
        sign = "+" if effect >= 0 else "-"
        print(f"  MLP {layer:2d}:   {sign}{abs(effect):.4f} {bar}")
    
    # Identify key components (Hanna et al. found MLPs 8-11 critical)
    print("\n--- Key Findings ---")
    
    # Top attention layers
    top_attn = sorted(attn_avg.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"Top attention layers: {[f'L{l} ({e:.3f})' for l, e in top_attn]}")
    
    # Top MLP layers
    top_mlp = sorted(mlp_avg.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    print(f"Top MLP layers: {[f'MLP{l} ({e:.3f})' for l, e in top_mlp]}")
    
    # Check for MLPs 9-10 (Hanna et al. key finding)
    mlp9_effect = mlp_avg.get(9, 0)
    mlp10_effect = mlp_avg.get(10, 0)
    mlp11_effect = mlp_avg.get(11, 0)
    
    print(f"\nHanna et al. Key MLPs:")
    print(f"  MLP 9:  {mlp9_effect:.4f}")
    print(f"  MLP 10: {mlp10_effect:.4f}")
    print(f"  MLP 11: {mlp11_effect:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline Probability Difference: {avg_pd:.4f}")
    print(f"Number of examples: {n_examples}")
    print(f"Number of prompt pairs: {len(pairs)}")
    
    # Compare to paper findings
    print("\nComparison to Hanna et al.:")
    if avg_pd > 0.7:
        print("  [MATCH] High baseline PD (paper: ~0.81)")
    elif avg_pd > 0.3:
        print("  [PARTIAL] Moderate baseline PD")
    else:
        print("  [DIFFER] Low baseline PD")
    
    important_mlps = [l for l, e in top_mlp if e > 0.05]
    if 9 in important_mlps or 10 in important_mlps:
        print("  [MATCH] MLPs 9/10 among top components (paper finding)")
    else:
        print(f"  [DIFFER] Top MLPs are {important_mlps}, not 9/10")
    
    return {
        'baseline_pd': avg_pd,
        'baseline_pd_std': std_pd,
        'attention_effects': attn_avg,
        'mlp_effects': mlp_avg,
        'n_examples': n_examples,
        'all_attention_results': attention_results,
        'all_mlp_results': mlp_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Greater Than Circuit Analysis (Hanna et al.)")
    parser.add_argument("--n_examples", type=int, default=50, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    results = run_circuit_analysis(
        n_examples=args.n_examples,
        seed=args.seed,
        device=args.device
    )
    
    print("\nAnalysis complete!")
