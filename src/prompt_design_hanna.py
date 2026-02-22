"""
Prompt Design Module - Hanna et al. Paper Implementation

This module implements the exact methodology from:
"How does GPT-2 compute greater-than?: Interpreting mathematical abilities 
in a pre-trained language model" (Hanna et al., NeurIPS 2023)

Key methodology:
- Prompt format: "The war lasted from the year 17{YY} to the year 17"
- Task: Predict valid 2-digit end years (years > starting year YY)
- Metric: Probability Difference = Σp(y > YY) - Σp(y ≤ YY)
- Corruption: "01" dataset where all starting years end in 01

Reference: https://arxiv.org/abs/2305.00586
"""

import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Two-digit years that are single tokens in GPT-2's tokenizer
# Excludes 00 which tokenizes differently
VALID_YEARS = list(range(1, 100))  # 01-99

# Template variations from the paper
YEAR_TEMPLATES = [
    "The war lasted from the year 17{yy} to the year 17",
    "The event occurred from the year 17{yy} to the year 17",
    "The period spanned from 17{yy} to 17",
    "From 17{yy} to 17",
    "The reign was from 17{yy} until 17",
]


@dataclass
class YearPromptExample:
    """
    Data class for year completion prompts (Hanna et al. methodology).
    
    Attributes:
        start_year: The starting 2-digit year (e.g., 32 for 1732)
        prompt_text: The formatted prompt
        valid_years: List of valid completion years (> start_year)
        is_corrupted: Whether this uses the 01-corruption baseline
    """
    start_year: int
    prompt_text: str
    valid_years: List[int] = field(default_factory=list)
    is_corrupted: bool = False
    template_idx: int = 0
    
    def __post_init__(self):
        """Compute valid years after initialization."""
        if not self.valid_years:
            self.valid_years = [y for y in VALID_YEARS if y > self.start_year]


class YearPromptGenerator:
    """
    Generates year completion prompts following Hanna et al. methodology.
    
    The task: Given "The war lasted from the year 1732 to the year 17",
    the model should assign higher probability to years > 32 (e.g., 33-99)
    than to years ≤ 32 (e.g., 01-32).
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.templates = YEAR_TEMPLATES
        logger.info(f"YearPromptGenerator initialized with seed {seed}")
    
    def generate_year_examples(
        self,
        n_examples: int = 100,
        year_range: Tuple[int, int] = (1, 98),
        template_idx: int = 0
    ) -> List[YearPromptExample]:
        """
        Generate year completion examples.
        
        Args:
            n_examples: Number of examples to generate
            year_range: Range for starting years (min, max)
            template_idx: Which template to use (0 = default war template)
            
        Returns:
            List of YearPromptExample objects
        """
        examples = []
        template = self.templates[template_idx]
        
        for _ in range(n_examples):
            # Sample a starting year
            start_year = random.randint(year_range[0], year_range[1])
            
            # Format as 2-digit string
            yy_str = f"{start_year:02d}"
            
            # Create prompt
            prompt_text = template.format(yy=yy_str)
            
            example = YearPromptExample(
                start_year=start_year,
                prompt_text=prompt_text,
                template_idx=template_idx
            )
            examples.append(example)
        
        logger.info(f"Generated {n_examples} year completion examples")
        return examples
    
    def generate_balanced_year_dataset(
        self,
        n_examples: int = 100,
        template_idx: int = 0
    ) -> List[YearPromptExample]:
        """
        Generate a balanced dataset across different starting years.
        
        Samples uniformly across the year range to ensure good coverage.
        
        Args:
            n_examples: Number of examples
            template_idx: Template to use
            
        Returns:
            List of YearPromptExample objects
        """
        examples = []
        template = self.templates[template_idx]
        
        # Sample starting years uniformly from 01-98
        # (99 has no valid completions > 99)
        year_range = list(range(1, 99))
        
        if n_examples >= len(year_range):
            # Use all years, then repeat as needed
            sampled_years = year_range * (n_examples // len(year_range) + 1)
            sampled_years = sampled_years[:n_examples]
        else:
            sampled_years = random.sample(year_range, n_examples)
        
        for start_year in sampled_years:
            yy_str = f"{start_year:02d}"
            prompt_text = template.format(yy=yy_str)
            
            example = YearPromptExample(
                start_year=start_year,
                prompt_text=prompt_text,
                template_idx=template_idx
            )
            examples.append(example)
        
        random.shuffle(examples)
        logger.info(f"Generated {len(examples)} balanced year examples")
        return examples
    
    def generate_01_corrupted_examples(
        self,
        clean_examples: List[YearPromptExample]
    ) -> List[YearPromptExample]:
        """
        Generate 01-corrupted versions of clean examples.
        
        Following Hanna et al., the corruption dataset uses YY=01 for all
        examples. This creates a baseline where almost all years (02-99) are
        valid, making the task trivially easy.
        
        Args:
            clean_examples: Clean examples to create corrupted versions for
            
        Returns:
            List of corrupted YearPromptExample objects
        """
        corrupted = []
        
        for example in clean_examples:
            template = self.templates[example.template_idx]
            
            # Use 01 as the starting year (corruption baseline)
            corrupted_prompt = template.format(yy="01")
            
            corrupted_example = YearPromptExample(
                start_year=1,  # 01
                prompt_text=corrupted_prompt,
                is_corrupted=True,
                template_idx=example.template_idx
            )
            corrupted.append(corrupted_example)
        
        logger.info(f"Generated {len(corrupted)} 01-corrupted examples")
        return corrupted
    
    def create_prompt_pairs(
        self,
        n_pairs: int = 50,
        template_idx: int = 0
    ) -> List[Tuple[YearPromptExample, YearPromptExample]]:
        """
        Create clean/corrupted prompt pairs for activation patching.
        
        Args:
            n_pairs: Number of pairs to create
            template_idx: Template to use
            
        Returns:
            List of (clean, corrupted) tuples
        """
        clean_examples = self.generate_balanced_year_dataset(n_pairs, template_idx)
        corrupted_examples = self.generate_01_corrupted_examples(clean_examples)
        
        pairs = list(zip(clean_examples, corrupted_examples))
        logger.info(f"Created {len(pairs)} prompt pairs for activation patching")
        return pairs


def get_year_token_ids(tokenizer) -> Dict[str, int]:
    """
    Get token IDs for all 2-digit years.
    
    In GPT-2, most 2-digit numbers are single tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer or TransformerLens model tokenizer
        
    Returns:
        Dict mapping year strings to token IDs
    """
    year_tokens = {}
    
    for year in range(0, 100):
        year_str = f"{year:02d}"
        
        # Try to get token ID
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(year_str, add_special_tokens=False)
            if len(tokens) == 1:
                year_tokens[year_str] = tokens[0]
        else:
            # TransformerLens model
            tokens = tokenizer.to_tokens(year_str, prepend_bos=False)
            if tokens.shape[-1] == 1:
                year_tokens[year_str] = tokens[0, 0].item()
    
    return year_tokens


def compute_probability_difference(
    logits: torch.Tensor,
    start_year: int,
    year_token_ids: Dict[str, int]
) -> float:
    """
    Compute the Probability Difference (PD) metric from Hanna et al.
    
    PD = Σp(y > YY) - Σp(y ≤ YY)
    
    where YY is the starting year and y ranges over all valid year tokens.
    
    Args:
        logits: Model output logits at the final position [vocab_size]
        start_year: The starting 2-digit year (e.g., 32)
        year_token_ids: Dict mapping year strings to token IDs
        
    Returns:
        Probability difference as float in range [-1, 1]
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    prob_greater = 0.0
    prob_less_equal = 0.0
    
    for year in range(1, 100):  # 01-99
        year_str = f"{year:02d}"
        if year_str not in year_token_ids:
            continue
            
        token_id = year_token_ids[year_str]
        prob = probs[token_id].item()
        
        if year > start_year:
            prob_greater += prob
        else:
            prob_less_equal += prob
    
    pd = prob_greater - prob_less_equal
    return pd


def compute_batch_probability_difference(
    model,
    examples: List[YearPromptExample],
    year_token_ids: Dict[str, int]
) -> Tuple[float, List[float]]:
    """
    Compute average and individual PD scores for a batch of examples.
    
    Args:
        model: TransformerLens HookedTransformer model
        examples: List of YearPromptExample objects
        year_token_ids: Dict mapping year strings to token IDs
        
    Returns:
        Tuple of (average_pd, list_of_individual_pds)
    """
    pds = []
    
    for example in examples:
        # Tokenize and run model
        tokens = model.to_tokens(example.prompt_text)
        
        with torch.no_grad():
            logits = model(tokens)
        
        # Get logits for last position
        final_logits = logits[0, -1, :]
        
        # Compute PD
        pd = compute_probability_difference(
            final_logits, 
            example.start_year, 
            year_token_ids
        )
        pds.append(pd)
    
    avg_pd = sum(pds) / len(pds) if pds else 0.0
    return avg_pd, pds


def create_pd_metric_fn(model, year_token_ids: Dict[str, int], start_year: int):
    """
    Create a metric function for activation patching that returns PD.
    
    Args:
        model: TransformerLens model (not used in metric, just for interface)
        year_token_ids: Dict mapping year strings to token IDs
        start_year: The starting 2-digit year for this example
        
    Returns:
        Metric function that takes logits and returns PD
    """
    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        """Compute PD from logits."""
        # logits shape: [batch, seq, vocab] or [seq, vocab]
        if logits.dim() == 3:
            final_logits = logits[0, -1, :]
        else:
            final_logits = logits[-1, :]
        
        probs = torch.softmax(final_logits, dim=-1)
        
        prob_greater = torch.tensor(0.0, device=logits.device)
        prob_less_equal = torch.tensor(0.0, device=logits.device)
        
        for year in range(1, 100):
            year_str = f"{year:02d}"
            if year_str not in year_token_ids:
                continue
            
            token_id = year_token_ids[year_str]
            prob = probs[token_id]
            
            if year > start_year:
                prob_greater = prob_greater + prob
            else:
                prob_less_equal = prob_less_equal + prob
        
        return prob_greater - prob_less_equal
    
    return metric_fn


# Example usage and test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate examples
    generator = YearPromptGenerator(seed=42)
    
    # Basic examples
    examples = generator.generate_year_examples(5)
    print("\n=== Year Completion Examples ===")
    for ex in examples:
        print(f"Prompt: '{ex.prompt_text}'")
        print(f"  Start year: {ex.start_year:02d}")
        print(f"  Valid completions: {len(ex.valid_years)} years (>{ex.start_year:02d})")
    
    # Prompt pairs
    pairs = generator.create_prompt_pairs(3)
    print("\n=== Prompt Pairs for Patching ===")
    for clean, corrupt in pairs:
        print(f"Clean:    '{clean.prompt_text}' (valid: >{clean.start_year:02d})")
        print(f"Corrupt:  '{corrupt.prompt_text}' (valid: >{corrupt.start_year:02d})")
        print()
