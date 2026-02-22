"""Debug script to understand patching results."""

import torch
import logging
from src.model_setup import ModelSetup
from src.prompt_design import PromptGenerator
from src.activation_patching import ActivationPatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Setup
    setup = ModelSetup("gpt2-small", device="cpu")
    model = setup.load_model()
    
    generator = PromptGenerator(seed=42)
    prompt_pairs = generator.create_prompt_pairs(n_pairs=5, corruption_type="swap_numbers")
    
    patcher = ActivationPatcher(model)
    
    # Get one prompt pair
    clean_example, corrupted_example = prompt_pairs[0]
    
    logger.info(f"Clean prompt: {clean_example.prompt_text}")
    logger.info(f"Corrupted prompt: {corrupted_example.prompt_text}")
    
    # Tokenize
    clean_tokens = model.to_tokens(clean_example.prompt_text + " ")
    corrupted_tokens = model.to_tokens(corrupted_example.prompt_text + " ")
    
    # Get baseline predictions
    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    
    # Get probability of completion tokens
    clean_probs = torch.softmax(clean_logits[0, -1, :], dim=-1)
    corrupt_probs = torch.softmax(corrupted_logits[0, -1, :], dim=-1)
    
    # Get top 10 tokens for each
    logger.info("\nClean prompt top tokens:")
    top_clean = torch.topk(clean_probs, 10)
    for i, (prob, idx) in enumerate(zip(top_clean.values, top_clean.indices)):
        token = model.to_string(idx.item())
        logger.info(f"  {i+1}. {repr(token)} ({prob.item():.4f})")
    
    logger.info("\nCorrupted prompt top tokens:")
    top_corrupt = torch.topk(corrupt_probs, 10)
    for i, (prob, idx) in enumerate(zip(top_corrupt.values, top_corrupt.indices)):
        token = model.to_string(idx.item())
        logger.info(f"  {i+1}. {repr(token)} ({prob.item():.4f})")
    
    # Run patching on a few key layers
    logger.info("\n=== Patching Results ===")
    
    results = patcher.patch_attention_heads(
        corrupted_tokens=corrupted_tokens,
        clean_tokens=clean_tokens,
        positions=[-1]
    )
    
    # Sort by effect size
    sorted_results = sorted(results, key=lambda r: abs(r.effect_size), reverse=True)
    
    logger.info(f"\nTotal results: {len(sorted_results)}")
    logger.info("\nTop 20 by effect size:")
    for r in sorted_results[:20]:
        logger.info(f"  L{r.layer}H{r.head}: effect={r.effect_size:.4f}, "
                   f"metric_diff={r.metric_diff:.4f}, "
                   f"orig={r.original_metric:.4f}, patched={r.patched_metric:.4f}")
    
    # Also show MLP patching
    logger.info("\n=== MLP Patching Results ===")
    mlp_results = patcher.patch_mlp_layers(
        corrupted_tokens=corrupted_tokens,
        clean_tokens=clean_tokens,
        positions=[-1]
    )
    
    mlp_sorted = sorted(mlp_results, key=lambda r: abs(r.effect_size), reverse=True)
    logger.info(f"\nMLP results (top 12):")
    for r in mlp_sorted[:12]:
        logger.info(f"  MLP{r.layer}: effect={r.effect_size:.4f}, "
                   f"metric_diff={r.metric_diff:.4f}, "
                   f"orig={r.original_metric:.4f}, patched={r.patched_metric:.4f}")

if __name__ == "__main__":
    main()
