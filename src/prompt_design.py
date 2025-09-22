"""
Prompt Design Module

This module handles the generation of prompts for testing the "greater than" capability
in language models. It creates structured test cases with number pairs and True/False labels.

Acknowledgment: Prompt design methodologies inspired by mechanistic interpretability
research practices developed by Neel Nanda and the research community.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptExample:
    """
    Data class representing a single prompt example for greater than testing.
    
    Attributes:
        num1 (int): First number in the comparison
        num2 (int): Second number in the comparison  
        correct_answer (bool): Whether num1 > num2
        prompt_text (str): The formatted prompt string
        answer_text (str): The expected answer ("True" or "False")
    """
    num1: int
    num2: int
    correct_answer: bool
    prompt_text: str
    answer_text: str
    
    def __post_init__(self):
        """Validate the example after initialization."""
        assert (self.num1 > self.num2) == self.correct_answer, \
            f"Inconsistent example: {self.num1} > {self.num2} = {self.correct_answer}"


class PromptGenerator:
    """
    Generates structured prompts for testing greater than capability in language models.
    
    This class creates various types of test cases including clean examples, corrupted
    examples for activation patching, and edge cases for comprehensive testing.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the prompt generator.
        
        Args:
            seed (int, optional): Random seed for reproducible results
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Template patterns for prompts
        self.prompt_templates = [
            "{num1} > {num2}",
            "Is {num1} greater than {num2}?",
            "{num1} is greater than {num2}",
            "The number {num1} is greater than {num2}",
            "Compare: {num1} > {num2}",
        ]
        
        self.answer_options = ["True", "False"]
        
        logger.info(f"PromptGenerator initialized with seed {seed}")
    
    def generate_basic_examples(
        self,
        n_examples: int = 100,
        num_range: Tuple[int, int] = (1, 100),
        template_idx: int = 0
    ) -> List[PromptExample]:
        """
        Generate basic greater than examples.
        
        Args:
            n_examples (int): Number of examples to generate
            num_range (Tuple[int, int]): Range for random numbers (min, max)
            template_idx (int): Index of prompt template to use
            
        Returns:
            List[PromptExample]: List of generated examples
        """
        examples = []
        template = self.prompt_templates[template_idx]
        
        for _ in range(n_examples):
            # Generate two random numbers
            num1 = random.randint(num_range[0], num_range[1])
            num2 = random.randint(num_range[0], num_range[1])
            
            # Ensure numbers are different for clearer comparison
            while num1 == num2:
                num2 = random.randint(num_range[0], num_range[1])
            
            # Determine correct answer
            correct_answer = num1 > num2
            answer_text = "True" if correct_answer else "False"
            
            # Format prompt
            prompt_text = template.format(num1=num1, num2=num2)
            
            example = PromptExample(
                num1=num1,
                num2=num2,
                correct_answer=correct_answer,
                prompt_text=prompt_text,
                answer_text=answer_text
            )
            examples.append(example)
        
        logger.info(f"Generated {len(examples)} basic examples")
        return examples
    
    def generate_balanced_dataset(
        self,
        n_examples: int = 200,
        num_range: Tuple[int, int] = (1, 50),
        template_idx: int = 0
    ) -> List[PromptExample]:
        """
        Generate a balanced dataset with equal numbers of True/False examples.
        
        Args:
            n_examples (int): Total number of examples (should be even)
            num_range (Tuple[int, int]): Range for random numbers
            template_idx (int): Index of prompt template to use
            
        Returns:
            List[PromptExample]: Balanced list of examples
        """
        if n_examples % 2 != 0:
            logger.warning(f"n_examples ({n_examples}) is odd, adding 1 to make it even")
            n_examples += 1
        
        examples = []
        template = self.prompt_templates[template_idx]
        
        # Generate True examples (num1 > num2)
        for _ in range(n_examples // 2):
            num2 = random.randint(num_range[0], num_range[1] - 1)
            num1 = random.randint(num2 + 1, num_range[1])
            
            prompt_text = template.format(num1=num1, num2=num2)
            
            example = PromptExample(
                num1=num1,
                num2=num2,
                correct_answer=True,
                prompt_text=prompt_text,
                answer_text="True"
            )
            examples.append(example)
        
        # Generate False examples (num1 <= num2)
        for _ in range(n_examples // 2):
            num1 = random.randint(num_range[0], num_range[1] - 1)
            num2 = random.randint(num1, num_range[1])  # num2 >= num1
            
            prompt_text = template.format(num1=num1, num2=num2)
            
            example = PromptExample(
                num1=num1,
                num2=num2,
                correct_answer=False,
                prompt_text=prompt_text,
                answer_text="False"
            )
            examples.append(example)
        
        # Shuffle the examples
        random.shuffle(examples)
        
        logger.info(f"Generated {len(examples)} balanced examples")
        return examples
    
    def generate_edge_cases(self, template_idx: int = 0) -> List[PromptExample]:
        """
        Generate edge case examples for thorough testing.
        
        Args:
            template_idx (int): Index of prompt template to use
            
        Returns:
            List[PromptExample]: List of edge case examples
        """
        examples = []
        template = self.prompt_templates[template_idx]
        
        # Edge cases to test
        edge_cases = [
            (1, 0),    # Small numbers
            (0, 1),
            (10, 9),   # Close numbers
            (9, 10),
            (100, 1),  # Large vs small
            (1, 100),
            (50, 50),  # Equal numbers (should be False for >)
            (0, 0),    # Both zero
            (99, 100), # Large close numbers
            (100, 99),
        ]
        
        for num1, num2 in edge_cases:
            correct_answer = num1 > num2
            answer_text = "True" if correct_answer else "False"
            prompt_text = template.format(num1=num1, num2=num2)
            
            example = PromptExample(
                num1=num1,
                num2=num2,
                correct_answer=correct_answer,
                prompt_text=prompt_text,
                answer_text=answer_text
            )
            examples.append(example)
        
        logger.info(f"Generated {len(examples)} edge case examples")
        return examples
    
    def generate_corrupted_examples(
        self,
        clean_examples: List[PromptExample],
        corruption_type: str = "flip_answer"
    ) -> List[PromptExample]:
        """
        Generate corrupted versions of clean examples for activation patching.
        
        Args:
            clean_examples (List[PromptExample]): Clean examples to corrupt
            corruption_type (str): Type of corruption to apply
            
        Returns:
            List[PromptExample]: Corrupted examples
        """
        corrupted = []
        
        for example in clean_examples:
            if corruption_type == "flip_answer":
                # Keep the same prompt but flip the expected answer
                corrupted_example = PromptExample(
                    num1=example.num1,
                    num2=example.num2,
                    correct_answer=not example.correct_answer,
                    prompt_text=example.prompt_text,
                    answer_text="False" if example.answer_text == "True" else "True"
                )
            elif corruption_type == "swap_numbers":
                # Swap the numbers in the prompt
                corrupted_prompt = example.prompt_text.replace(
                    str(example.num1), "TEMP"
                ).replace(
                    str(example.num2), str(example.num1)
                ).replace(
                    "TEMP", str(example.num2)
                )
                
                corrupted_example = PromptExample(
                    num1=example.num2,  # Swapped
                    num2=example.num1,  # Swapped
                    correct_answer=example.num2 > example.num1,
                    prompt_text=corrupted_prompt,
                    answer_text="True" if example.num2 > example.num1 else "False"
                )
            else:
                raise ValueError(f"Unknown corruption type: {corruption_type}")
            
            corrupted.append(corrupted_example)
        
        logger.info(f"Generated {len(corrupted)} corrupted examples using {corruption_type}")
        return corrupted
    
    def create_prompt_pairs(
        self,
        n_pairs: int = 50,
        num_range: Tuple[int, int] = (1, 50)
    ) -> List[Tuple[PromptExample, PromptExample]]:
        """
        Create pairs of clean and corrupted examples for activation patching.
        
        Args:
            n_pairs (int): Number of prompt pairs to create
            num_range (Tuple[int, int]): Range for random numbers
            
        Returns:
            List[Tuple[PromptExample, PromptExample]]: List of (clean, corrupted) pairs
        """
        # Generate clean examples
        clean_examples = self.generate_basic_examples(n_pairs, num_range)
        
        # Generate corrupted examples
        corrupted_examples = self.generate_corrupted_examples(clean_examples)
        
        # Pair them up
        pairs = list(zip(clean_examples, corrupted_examples))
        
        logger.info(f"Created {len(pairs)} prompt pairs for activation patching")
        return pairs
    
    def format_for_model(
        self,
        examples: List[PromptExample],
        include_answer: bool = False,
        separator: str = " "
    ) -> List[str]:
        """
        Format examples for model input.
        
        Args:
            examples (List[PromptExample]): Examples to format
            include_answer (bool): Whether to include the answer in the prompt
            separator (str): Separator between prompt and answer
            
        Returns:
            List[str]: Formatted prompts ready for model input
        """
        formatted = []
        
        for example in examples:
            prompt = example.prompt_text
            if include_answer:
                prompt += f"{separator}{example.answer_text}"
            formatted.append(prompt)
        
        return formatted
    
    def get_statistics(self, examples: List[PromptExample]) -> Dict[str, Any]:
        """
        Get statistics about a set of examples.
        
        Args:
            examples (List[PromptExample]): Examples to analyze
            
        Returns:
            Dict[str, Any]: Statistics about the examples
        """
        true_count = sum(1 for ex in examples if ex.correct_answer)
        false_count = len(examples) - true_count
        
        num1_values = [ex.num1 for ex in examples]
        num2_values = [ex.num2 for ex in examples]
        
        stats = {
            "total_examples": len(examples),
            "true_examples": true_count,
            "false_examples": false_count,
            "balance_ratio": true_count / len(examples) if examples else 0,
            "num1_range": (min(num1_values), max(num1_values)) if num1_values else (0, 0),
            "num2_range": (min(num2_values), max(num2_values)) if num2_values else (0, 0),
            "num1_mean": np.mean(num1_values) if num1_values else 0,
            "num2_mean": np.mean(num2_values) if num2_values else 0,
        }
        
        return stats
    
    def print_statistics(self, examples: List[PromptExample]):
        """Print formatted statistics about examples."""
        stats = self.get_statistics(examples)
        
        print(f"\n{'='*40}")
        print("DATASET STATISTICS")
        print(f"{'='*40}")
        print(f"Total Examples: {stats['total_examples']:,}")
        print(f"True Examples: {stats['true_examples']:,} ({stats['balance_ratio']:.1%})")
        print(f"False Examples: {stats['false_examples']:,} ({1-stats['balance_ratio']:.1%})")
        print(f"Number 1 Range: {stats['num1_range'][0]} - {stats['num1_range'][1]} (mean: {stats['num1_mean']:.1f})")
        print(f"Number 2 Range: {stats['num2_range'][0]} - {stats['num2_range'][1]} (mean: {stats['num2_mean']:.1f})")
        print(f"{'='*40}\n")
    
    def save_examples(self, examples: List[PromptExample], filepath: str):
        """
        Save examples to a CSV file.
        
        Args:
            examples (List[PromptExample]): Examples to save
            filepath (str): Path to save the file
        """
        import pandas as pd
        
        data = []
        for example in examples:
            data.append({
                "num1": example.num1,
                "num2": example.num2,
                "correct_answer": example.correct_answer,
                "prompt_text": example.prompt_text,
                "answer_text": example.answer_text
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(examples)} examples to {filepath}")


def main():
    """Example usage of the PromptGenerator class."""
    # Initialize generator
    generator = PromptGenerator(seed=42)
    
    # Generate basic examples
    basic_examples = generator.generate_basic_examples(n_examples=20)
    print("Sample basic examples:")
    for i, ex in enumerate(basic_examples[:5]):
        print(f"{i+1}. {ex.prompt_text} -> {ex.answer_text}")
    
    # Generate balanced dataset
    balanced_examples = generator.generate_balanced_dataset(n_examples=100)
    generator.print_statistics(balanced_examples)
    
    # Generate edge cases
    edge_cases = generator.generate_edge_cases()
    print("\nEdge case examples:")
    for ex in edge_cases[:5]:
        print(f"   {ex.prompt_text} -> {ex.answer_text}")
    
    # Create prompt pairs for activation patching
    pairs = generator.create_prompt_pairs(n_pairs=10)
    print(f"\nCreated {len(pairs)} prompt pairs for activation patching")
    
    # Show example pair
    clean, corrupted = pairs[0]
    print(f"Clean: {clean.prompt_text} -> {clean.answer_text}")
    print(f"Corrupted: {corrupted.prompt_text} -> {corrupted.answer_text}")


if __name__ == "__main__":
    main()