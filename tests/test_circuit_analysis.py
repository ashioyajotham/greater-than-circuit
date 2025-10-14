"""
Test suite for the Greater Than Circuit analysis modules.

This file provides unit tests for the core functionality of the
circuit analysis implementation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from src.prompt_design import PromptGenerator, PromptExample
from src.activation_patching import PatchingResult
from src.circuit_analysis import CircuitComponent, CircuitAnalyzer
from src.circuit_validation import ValidationResult


class TestPromptGenerator:
    """Test cases for the PromptGenerator class."""
    
    def test_initialization(self):
        """Test proper initialization of PromptGenerator."""
        generator = PromptGenerator(seed=42)
        assert generator.seed == 42
        assert len(generator.prompt_templates) > 0
        assert "True" in generator.answer_options
        assert "False" in generator.answer_options
    
    def test_generate_basic_examples(self):
        """Test generation of basic examples."""
        generator = PromptGenerator(seed=42)
        examples = generator.generate_basic_examples(n_examples=10)
        
        assert len(examples) == 10
        assert all(isinstance(ex, PromptExample) for ex in examples)
        assert all(ex.num1 != ex.num2 for ex in examples)  # Numbers should be different
        
        # Check consistency
        for ex in examples:
            assert (ex.num1 > ex.num2) == ex.correct_answer
            assert ex.answer_text == ("True" if ex.correct_answer else "False")
    
    def test_generate_balanced_dataset(self):
        """Test generation of balanced dataset."""
        generator = PromptGenerator(seed=42)
        examples = generator.generate_balanced_dataset(n_examples=20)
        
        assert len(examples) == 20
        
        # Check balance
        true_count = sum(1 for ex in examples if ex.correct_answer)
        false_count = len(examples) - true_count
        assert abs(true_count - false_count) <= 1  # Should be balanced
    
    def test_generate_edge_cases(self):
        """Test generation of edge cases."""
        generator = PromptGenerator(seed=42)
        edge_cases = generator.generate_edge_cases()
        
        assert len(edge_cases) > 0
        assert all(isinstance(ex, PromptExample) for ex in edge_cases)
        
        # Should include some specific edge cases
        nums = [(ex.num1, ex.num2) for ex in edge_cases]
        assert (0, 0) in nums or (1, 0) in nums  # Some small number cases
    
    def test_generate_corrupted_examples(self):
        """Test generation of corrupted examples."""
        generator = PromptGenerator(seed=42)
        clean_examples = generator.generate_basic_examples(n_examples=5)
        
        # Test flip_answer corruption
        corrupted = generator.generate_corrupted_examples(clean_examples, "flip_answer")
        assert len(corrupted) == len(clean_examples)
        
        for clean, corrupt in zip(clean_examples, corrupted):
            assert clean.correct_answer != corrupt.correct_answer
            assert clean.prompt_text == corrupt.prompt_text  # Prompt stays same
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        generator = PromptGenerator(seed=42)
        examples = generator.generate_balanced_dataset(n_examples=100)
        
        stats = generator.get_statistics(examples)
        assert stats["total_examples"] == 100
        assert 0.4 <= stats["balance_ratio"] <= 0.6  # Should be roughly balanced
        assert stats["num1_range"][0] <= stats["num1_range"][1]
        assert stats["num2_range"][0] <= stats["num2_range"][1]


class TestPatchingResult:
    """Test cases for PatchingResult dataclass."""
    
    def test_initialization(self):
        """Test proper initialization of PatchingResult."""
        result = PatchingResult(
            hook_name="blocks.5.attn.hook_result",
            layer=5,
            head=7,
            position=-1,
            metric_diff=0.3,
            original_metric=0.2,
            patched_metric=0.5,
            effect_size=0.3
        )
        
        assert result.hook_name == "blocks.5.attn.hook_result"
        assert result.layer == 5
        assert result.head == 7
        assert result.position == -1
        assert result.metric_diff == 0.3
        assert result.effect_size == 0.3


class TestCircuitComponent:
    """Test cases for CircuitComponent dataclass."""
    
    def test_initialization(self):
        """Test proper initialization of CircuitComponent."""
        component = CircuitComponent(
            name="L5H7",
            layer=5,
            head=7,
            importance_score=0.45,
            component_type="attention"
        )
        
        assert component.name == "L5H7"
        assert component.layer == 5
        assert component.head == 7
        assert component.importance_score == 0.45
        assert component.component_type == "attention"
        assert component.connections == []  # Default empty list
    
    def test_connections_list(self):
        """Test connections list functionality."""
        component = CircuitComponent(
            name="L3_mlp",
            layer=3,
            importance_score=0.32,
            component_type="mlp",
            connections=["L5H7", "L7H2"]
        )
        
        assert len(component.connections) == 2
        assert "L5H7" in component.connections


class TestValidationResult:
    """Test cases for ValidationResult dataclass."""
    
    def test_initialization(self):
        """Test proper initialization of ValidationResult."""
        result = ValidationResult(
            test_name="baseline_accuracy",
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            total_examples=100,
            correct_predictions=85
        )
        
        assert result.test_name == "baseline_accuracy"
        assert result.accuracy == 0.85
        assert result.precision == 0.87
        assert result.recall == 0.83
        assert result.f1_score == 0.85
        assert result.total_examples == 100
        assert result.correct_predictions == 85
        assert result.details == {}  # Default empty dict


class TestMockCircuitAnalyzer:
    """Test cases for CircuitAnalyzer with mocked model."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.cfg.model_name = "gpt2-small"
        model.cfg.n_layers = 12
        model.cfg.n_heads = 12
        return model
    
    @pytest.fixture
    def mock_results(self):
        """Create mock patching results."""
        return [
            PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
            PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
            PatchingResult("blocks.7.attn.hook_result", 7, 2, -1, 0.28, 0.25, 0.53, 0.28),
            PatchingResult("blocks.1.attn.hook_result", 1, 3, -1, 0.05, 0.4, 0.45, 0.05),
        ]
    
    def test_identify_circuit_components(self, mock_model, mock_results):
        """Test circuit component identification."""
        analyzer = CircuitAnalyzer(mock_model)
        
        components = analyzer.identify_circuit_components(
            mock_results,
            importance_threshold=0.1,
            top_k=3
        )
        
        # Should return top 3 components above threshold
        assert len(components) == 3
        
        # Check that components are sorted by importance
        component_list = list(components.values())
        assert component_list[0].importance_score >= component_list[1].importance_score
        assert component_list[1].importance_score >= component_list[2].importance_score
        
        # Check that low-importance component is filtered out
        component_names = list(components.keys())
        assert "L1H3" not in component_names  # Should be filtered by threshold
    
    def test_analyze_layer_contributions(self, mock_model, mock_results):
        """Test layer contribution analysis."""
        analyzer = CircuitAnalyzer(mock_model)
        
        contributions = analyzer.analyze_layer_contributions(mock_results)
        
        # Should have contributions for layers that appear in results
        assert 5 in contributions
        assert 3 in contributions  
        assert 7 in contributions
        assert 1 in contributions
        
        # Layer 5 should have highest contribution (0.45)
        assert contributions[5] == 0.45
        assert contributions[3] == 0.32


def test_integration_prompt_to_analysis():
    """Integration test from prompt generation to basic analysis."""
    # Generate test data
    generator = PromptGenerator(seed=42)
    examples = generator.generate_basic_examples(n_examples=5)
    
    # Create mock patching results
    mock_results = [
        PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
        PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
    ]
    
    # Mock model
    mock_model = Mock()
    mock_model.cfg.model_name = "gpt2-small"
    mock_model.cfg.n_layers = 12
    
    # Analyze components
    analyzer = CircuitAnalyzer(mock_model)
    components = analyzer.identify_circuit_components(mock_results)
    
    # Verify integration
    assert len(examples) == 5
    assert len(components) == 2
    assert "L5H7" in components
    assert "L3_mlp" in components


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])