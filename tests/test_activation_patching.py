"""
Test suite for the Activation Patching module.

This file provides unit tests for activation patching functionality,
including patching operations, metric calculations, and hook management.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Tuple

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from src.activation_patching import (
    ActivationPatcher,
    PatchingResult,
    compute_logit_diff,
    compute_probability_diff
)
from src.prompt_design import PromptGenerator, PromptExample


class TestPatchingResult:
    """Test cases for the PatchingResult dataclass."""
    
    def test_initialization_with_all_fields(self):
        """Test initialization with all fields specified."""
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
        assert result.original_metric == 0.2
        assert result.patched_metric == 0.5
        assert result.effect_size == 0.3
    
    def test_initialization_mlp_component(self):
        """Test initialization for MLP component (no head)."""
        result = PatchingResult(
            hook_name="blocks.3.mlp.hook_post",
            layer=3,
            head=None,
            position=-1,
            metric_diff=0.25,
            original_metric=0.1,
            patched_metric=0.35,
            effect_size=0.25
        )
        
        assert result.head is None
        assert "mlp" in result.hook_name
        assert result.layer == 3
    
    def test_effect_size_calculation(self):
        """Test that effect size matches metric diff."""
        result = PatchingResult(
            hook_name="blocks.2.attn.hook_result",
            layer=2,
            head=4,
            position=-1,
            metric_diff=0.42,
            original_metric=0.15,
            patched_metric=0.57,
            effect_size=0.42
        )
        
        # Effect size should equal metric diff
        assert result.effect_size == result.metric_diff
        assert abs(result.patched_metric - result.original_metric - result.metric_diff) < 1e-6


class TestMetricFunctions:
    """Test cases for metric computation functions."""
    
    def test_compute_logit_diff_basic(self):
        """Test basic logit difference computation."""
        # Create mock logits with proper vocab size [batch, seq, vocab]
        vocab_size = 50257  # GPT-2 vocab size
        logits = torch.zeros(1, 5, vocab_size)
        
        # Set specific logits for True (token 6407) and False (token 10352)
        true_token_id = 6407
        false_token_id = 10352
        
        logits[0, -1, true_token_id] = 2.0
        logits[0, -1, false_token_id] = 1.0
        
        diff = compute_logit_diff(logits, true_token_id, false_token_id)
        
        assert isinstance(diff, float)
        assert diff == 1.0  # 2.0 - 1.0
    
    def test_compute_logit_diff_negative(self):
        """Test logit diff when False logit is higher."""
        vocab_size = 50257
        logits = torch.zeros(1, 5, vocab_size)
        
        true_token_id = 6407
        false_token_id = 10352
        
        logits[0, -1, true_token_id] = 1.0
        logits[0, -1, false_token_id] = 3.0
        
        diff = compute_logit_diff(logits, true_token_id, false_token_id)
        
        assert diff == -2.0  # 1.0 - 3.0
    
    def test_compute_probability_diff_basic(self):
        """Test basic probability difference computation."""
        vocab_size = 50257
        logits = torch.zeros(1, 5, vocab_size)
        
        true_token_id = 6407
        false_token_id = 10352
        
        # Set logits that will give clear probabilities
        logits[0, -1, true_token_id] = 10.0
        logits[0, -1, false_token_id] = 0.0
        
        diff = compute_probability_diff(logits, true_token_id, false_token_id)
        
        assert isinstance(diff, float)
        assert diff > 0  # True should have higher probability
        assert -1.0 <= diff <= 1.0  # Probability diff is bounded
    
    def test_compute_probability_diff_normalized(self):
        """Test that probability differences are properly normalized."""
        vocab_size = 50257
        logits = torch.randn(1, 5, vocab_size)
        
        true_token_id = 6407
        false_token_id = 10352
        
        diff = compute_probability_diff(logits, true_token_id, false_token_id)
        
        # Should be bounded between -1 and 1
        assert -1.0 <= diff <= 1.0


class TestActivationPatcher:
    """Test cases for the ActivationPatcher class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock transformer model."""
        model = Mock()
        model.cfg.model_name = "gpt2-small"
        model.cfg.n_layers = 12
        model.cfg.n_heads = 12
        model.cfg.d_model = 768
        
        # Mock tokenizer methods
        model.to_tokens = Mock(return_value=torch.randint(0, 1000, (1, 10)))
        model.to_single_token = Mock(side_effect=lambda x: 6407 if "True" in x else 10352)
        
        # Mock forward pass
        model.run_with_cache = Mock(return_value=(
            torch.randn(1, 10, 50257),  # logits
            {}  # cache
        ))
        
        # Mock device - create a proper mock for parameters()
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        model.parameters = Mock(return_value=iter([mock_param]))
        
        # Mock the model call to return logits
        model.return_value = torch.randn(1, 10, 50257)
        
        # Mock add_hook and remove methods
        mock_hook = Mock()
        mock_hook.remove = Mock()
        model.add_hook = Mock(return_value=mock_hook)
        
        return model
    
    @pytest.fixture
    def patcher(self, mock_model):
        """Create an ActivationPatcher instance."""
        return ActivationPatcher(mock_model)
    
    def test_initialization(self, mock_model):
        """Test proper initialization of ActivationPatcher."""
        patcher = ActivationPatcher(mock_model)
        
        assert patcher.model == mock_model
        assert patcher.n_layers == 12
        assert patcher.n_heads == 12
        assert hasattr(patcher, 'true_token_id')
        assert hasattr(patcher, 'false_token_id')
    
    def test_get_hook_names_attention(self, patcher):
        """Test generation of attention hook names."""
        hook_names = patcher.get_hook_names(component_type="attention")
        
        assert isinstance(hook_names, list)
        assert len(hook_names) > 0
        
        # Should contain attention hooks
        assert any("attn" in name for name in hook_names)
        
        # Should have hooks for all layers
        layers_found = set()
        for name in hook_names:
            if "blocks." in name:
                layer_num = int(name.split(".")[1])
                layers_found.add(layer_num)
        
        assert len(layers_found) == 12  # All 12 layers
    
    def test_get_hook_names_mlp(self, patcher):
        """Test generation of MLP hook names."""
        hook_names = patcher.get_hook_names(component_type="mlp")
        
        assert isinstance(hook_names, list)
        assert len(hook_names) > 0
        
        # Should contain MLP hooks
        assert any("mlp" in name for name in hook_names)
    
    def test_get_hook_names_all(self, patcher):
        """Test generation of all hook names."""
        hook_names = patcher.get_hook_names(component_type="all")
        
        assert isinstance(hook_names, list)
        assert len(hook_names) > 0
        
        # Should contain both attention and MLP hooks
        assert any("attn" in name for name in hook_names)
        assert any("mlp" in name for name in hook_names)
    
    def test_get_hook_names_specific_layers(self, patcher):
        """Test filtering hook names by specific layers."""
        hook_names = patcher.get_hook_names(
            component_type="attention",
            layers=[3, 5, 7]
        )
        
        # Extract layer numbers from hook names
        layers_found = set()
        for name in hook_names:
            if "blocks." in name:
                layer_num = int(name.split(".")[1])
                layers_found.add(layer_num)
        
        # Should only have specified layers
        assert layers_found == {3, 5, 7}
    
    @patch.object(ActivationPatcher, 'store_activations')
    def test_patch_activation_basic(self, mock_store, patcher, mock_model):
        """Test basic activation patching."""
        # Mock stored activations
        mock_activation = torch.randn(1, 10, 768)
        mock_store.return_value = {"blocks.5.attn.hook_result": mock_activation}
        
        # Create sample tokens
        clean_tokens = torch.randint(0, 1000, (1, 10))
        corrupted_tokens = torch.randint(0, 1000, (1, 10))
        
        # Mock the model's forward pass to return consistent logits
        mock_model.return_value = torch.randn(1, 10, 50257)
        
        result = patcher.patch_activation(
            corrupted_tokens=corrupted_tokens,
            clean_tokens=clean_tokens,
            hook_name="blocks.5.attn.hook_result",
            position=-1
        )
        
        assert isinstance(result, PatchingResult)
        assert result.hook_name == "blocks.5.attn.hook_result"
        assert result.layer == 5
        assert result.position == -1
        assert isinstance(result.metric_diff, float)
    
    @patch.object(ActivationPatcher, 'store_activations')
    def test_patch_activation_with_head(self, mock_store, patcher, mock_model):
        """Test patching specific attention head."""
        # Mock stored activations
        mock_activation = torch.randn(1, 10, 768)
        mock_store.return_value = {"blocks.5.attn.hook_result": mock_activation}
        
        clean_tokens = torch.randint(0, 1000, (1, 10))
        corrupted_tokens = torch.randint(0, 1000, (1, 10))
        
        mock_model.return_value = torch.randn(1, 10, 50257)
        
        result = patcher.patch_activation(
            corrupted_tokens=corrupted_tokens,
            clean_tokens=clean_tokens,
            hook_name="blocks.5.attn.hook_result",
            position=-1,
            head_idx=7
        )
        
        assert result.head == 7
    
    @patch.object(ActivationPatcher, 'patch_activation')
    def test_run_patching_experiment_single_hook(self, mock_patch, patcher, mock_model):
        """Test running experiment on single hook."""
        # Mock patch_activation to return a result
        mock_patch.return_value = PatchingResult(
            "blocks.5.attn.hook_result", 5, None, -1, 0.3, 0.2, 0.5, 0.3
        )
        
        # Create sample examples
        generator = PromptGenerator(seed=42)
        clean_examples = generator.generate_basic_examples(n_examples=3)
        corrupted_examples = generator.generate_corrupted_examples(
            clean_examples, 
            "flip_answer"
        )
        
        results = patcher.run_patching_experiment(
            clean_examples=clean_examples,
            corrupted_examples=corrupted_examples,
            hook_names=["blocks.5.attn.hook_result"],
            positions=[-1]
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, PatchingResult) for r in results)
    
    @patch.object(ActivationPatcher, 'patch_activation')
    def test_run_patching_experiment_multiple_hooks(self, mock_patch, patcher, mock_model):
        """Test running experiment on multiple hooks."""
        # Mock patch_activation to return a result
        mock_patch.return_value = PatchingResult(
            "blocks.5.attn.hook_result", 5, None, -1, 0.3, 0.2, 0.5, 0.3
        )
        
        generator = PromptGenerator(seed=42)
        clean_examples = generator.generate_basic_examples(n_examples=2)
        corrupted_examples = generator.generate_corrupted_examples(
            clean_examples,
            "flip_answer"
        )
        
        hook_names = [
            "blocks.3.attn.hook_result",
            "blocks.5.attn.hook_result",
            "blocks.7.mlp.hook_post"
        ]
        
        results = patcher.run_patching_experiment(
            clean_examples=clean_examples,
            corrupted_examples=corrupted_examples,
            hook_names=hook_names,
            positions=[-1]
        )
        
        assert len(results) > 0
        
        # Check that we got results for different hooks
        assert all(isinstance(r, PatchingResult) for r in results)
    
    @patch.object(ActivationPatcher, 'run_patching_experiment')
    def test_run_comprehensive_patching(self, mock_run_patching, patcher, mock_model):
        """Test comprehensive patching across layers."""
        # Mock run_patching_experiment
        mock_run_patching.return_value = [
            PatchingResult("blocks.3.attn.hook_result", 3, None, -1, 0.3, 0.2, 0.5, 0.3),
            PatchingResult("blocks.5.attn.hook_result", 5, None, -1, 0.3, 0.2, 0.5, 0.3),
        ]
        
        generator = PromptGenerator(seed=42)
        clean_examples = generator.generate_basic_examples(n_examples=2)
        corrupted_examples = generator.generate_corrupted_examples(
            clean_examples,
            "flip_answer"
        )
        
        results = patcher.run_comprehensive_patching(
            clean_examples=clean_examples,
            corrupted_examples=corrupted_examples,
            component_types=["attention"],
            layers=[3, 5, 7]
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_get_top_components(self, patcher):
        """Test extraction of top components by effect size."""
        # Create sample results
        results = [
            PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
            PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
            PatchingResult("blocks.7.attn.hook_result", 7, 2, -1, 0.28, 0.25, 0.53, 0.28),
            PatchingResult("blocks.1.attn.hook_result", 1, 3, -1, 0.05, 0.4, 0.45, 0.05),
        ]
        
        top_3 = patcher.get_top_components(results, top_k=3)
        
        assert len(top_3) == 3
        
        # Check that results are sorted by effect size (descending)
        assert top_3[0].effect_size >= top_3[1].effect_size
        assert top_3[1].effect_size >= top_3[2].effect_size
        
        # Check that the highest is first
        assert top_3[0].effect_size == 0.45
    
    def test_filter_by_threshold(self, patcher):
        """Test filtering results by threshold."""
        results = [
            PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
            PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
            PatchingResult("blocks.7.attn.hook_result", 7, 2, -1, 0.28, 0.25, 0.53, 0.28),
            PatchingResult("blocks.1.attn.hook_result", 1, 3, -1, 0.05, 0.4, 0.45, 0.05),
        ]
        
        filtered = patcher.filter_by_threshold(results, threshold=0.25)
        
        # Should only include results with effect_size >= 0.25
        assert len(filtered) == 3
        assert all(r.effect_size >= 0.25 for r in filtered)
        
        # The low-scoring result should be filtered out
        assert all(r.effect_size != 0.05 for r in filtered)
    
    def test_summarize_results_by_layer(self, patcher):
        """Test summarization of results by layer."""
        results = [
            PatchingResult("blocks.5.attn.hook_result", 5, 7, -1, 0.45, 0.2, 0.65, 0.45),
            PatchingResult("blocks.5.attn.hook_result", 5, 3, -1, 0.30, 0.2, 0.50, 0.30),
            PatchingResult("blocks.3.mlp.hook_post", 3, None, -1, 0.32, 0.3, 0.62, 0.32),
            PatchingResult("blocks.7.attn.hook_result", 7, 2, -1, 0.28, 0.25, 0.53, 0.28),
        ]
        
        summary = patcher.summarize_results_by_layer(results)
        
        assert isinstance(summary, dict)
        
        # Should have entries for layers 3, 5, 7
        assert 3 in summary
        assert 5 in summary
        assert 7 in summary
        
        # Layer 5 should have max effect size of 0.45
        assert summary[5]['max_effect'] == 0.45
        assert summary[5]['count'] == 2  # Two results for layer 5


class TestIntegrationPatching:
    """Integration tests for activation patching workflow."""
    
    @patch.object(ActivationPatcher, 'patch_activation')
    def test_end_to_end_patching_workflow(self, mock_patch):
        """Test complete patching workflow from data to results."""
        # Mock patch_activation to return a result
        mock_patch.return_value = PatchingResult(
            "blocks.5.attn.hook_result", 5, None, -1, 0.3, 0.2, 0.5, 0.3
        )
        
        # 1. Generate data
        generator = PromptGenerator(seed=42)
        clean_examples = generator.generate_basic_examples(n_examples=3)
        corrupted_examples = generator.generate_corrupted_examples(
            clean_examples,
            "flip_answer"
        )
        
        # 2. Create mock model with proper parameter setup
        mock_model = Mock()
        mock_model.cfg.model_name = "gpt2-small"
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        mock_model.cfg.d_model = 768
        mock_model.to_tokens = Mock(return_value=torch.randint(0, 1000, (1, 10)))
        mock_model.to_single_token = Mock(side_effect=lambda x: 6407 if "True" in x else 10352)
        
        # Mock parameters() properly
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        # 3. Run patching
        patcher = ActivationPatcher(mock_model)
        results = patcher.run_patching_experiment(
            clean_examples=clean_examples,
            corrupted_examples=corrupted_examples,
            hook_names=["blocks.5.attn.hook_result"],
            positions=[-1]
        )
        
        # 4. Verify results
        assert len(results) > 0
        assert all(isinstance(r, PatchingResult) for r in results)
        
        # 5. Get top components
        top_components = patcher.get_top_components(results, top_k=5)
        assert len(top_components) <= 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])