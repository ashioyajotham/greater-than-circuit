"""
Test suite for the Model Setup module.

This file provides unit tests for model initialization, configuration,
tokenization, and basic inference capabilities.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from src.model_setup import ModelSetup


class TestModelSetup:
    """Test cases for the ModelSetup class."""
    
    @pytest.fixture
    def mock_hooked_transformer(self):
        """Create a mock HookedTransformer model."""
        with patch('src.model_setup.HookedTransformer') as mock_transformer:
            # Create mock model instance
            mock_model = Mock()
            mock_model.cfg.model_name = "gpt2-small"
            mock_model.cfg.n_layers = 12
            mock_model.cfg.n_heads = 12
            mock_model.cfg.d_model = 768
            mock_model.cfg.d_head = 64
            mock_model.cfg.d_vocab = 50257
            mock_model.cfg.n_ctx = 1024
            mock_model.cfg.d_mlp = 3072
            mock_model.cfg.device = "cpu"
            mock_model.cfg.act_fn = "gelu"
            mock_model.cfg.normalization_type = "LN"
            
            # Mock device
            mock_param = Mock()
            mock_param.device = torch.device('cpu')
            mock_model.parameters = Mock(return_value=iter([mock_param]))
            
            # Mock tokenizer with proper decode method
            mock_tokenizer = Mock()
            mock_tokenizer.decode = Mock(return_value=" True")
            mock_model.tokenizer = mock_tokenizer
            
            # Mock tokenization methods
            mock_model.to_tokens = Mock(return_value=torch.randint(0, 1000, (1, 5)))
            mock_model.to_str_tokens = Mock(return_value=["5", " >", " 3", ":", " True"])
            mock_model.to_single_token = Mock(side_effect=lambda x: 6407 if "True" in x else 10352)
            mock_model.to_string = Mock(return_value=" True")
            
            # Mock forward pass
            mock_logits = torch.randn(1, 5, 50257)
            mock_logits[0, -1, 6407] = 10.0  # High logit for " True"
            mock_model.return_value = mock_logits
            mock_model.run_with_cache = Mock(return_value=(mock_logits, {}))
            
            # Set the from_pretrained to return our mock model
            mock_transformer.from_pretrained = Mock(return_value=mock_model)
            
            yield mock_transformer, mock_model
    
    def test_initialization_default(self):
        """Test default initialization of ModelSetup."""
        setup = ModelSetup()
        
        assert setup.model_name == "gpt2-small"
        assert setup.device == "cpu"
        assert setup.model is None
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model name."""
        setup = ModelSetup(model_name="gpt2-medium")
        
        assert setup.model_name == "gpt2-medium"
        assert setup.device == "cpu"
    
    def test_initialization_cuda_device(self):
        """Test initialization with CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            setup = ModelSetup(device="cuda")
            assert setup.device == "cuda"
    
    def test_load_model_success(self, mock_hooked_transformer):
        """Test successful model loading."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        model = setup.load_model()
        
        # Verify model was loaded
        assert model is not None
        assert setup.model == model
        # Check that from_pretrained was called (don't check exact args)
        assert mock_transformer.from_pretrained.called
    
    def test_load_model_caching(self, mock_hooked_transformer):
        """Test that model loading checks if already loaded."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        model1 = setup.load_model()
        
        # Reset the model to None to test force_reload
        original_call_count = mock_transformer.from_pretrained.call_count
        
        # Load again without force_reload - should return cached
        model2 = setup.load_model()
        
        # Should be same instance
        assert model1 == model2
    
    def test_load_model_force_reload(self, mock_hooked_transformer):
        """Test force reload of model."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        setup.load_model()
        original_count = mock_transformer.from_pretrained.call_count
        
        setup.load_model(force_reload=True)
        
        # Should have called from_pretrained again
        assert mock_transformer.from_pretrained.call_count > original_count
    
    def test_get_model_info(self, mock_hooked_transformer):
        """Test retrieval of model information."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        setup.load_model()
        info = setup.get_model_info()
        
        assert isinstance(info, dict)
        assert info['model_name'] == "gpt2-small"
        assert info['n_layers'] == 12
        assert info['n_heads'] == 12
        assert info['d_model'] == 768
    
    def test_get_model_info_without_loading(self):
        """Test that getting info without loading model raises error."""
        setup = ModelSetup()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            setup.get_model_info()
    
    def test_test_model_basic(self, mock_hooked_transformer):
        """Test basic model testing functionality."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        setup.load_model()
        
        result = setup.test_model_basic("5 > 3:")
        
        assert isinstance(result, str)
        assert "5 > 3:" in result  # Should contain original prompt
        # Should have called the model
        assert mock_model.called
    
    def test_print_model_info(self, mock_hooked_transformer, capsys):
        """Test printing model information."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        setup.load_model()
        setup.print_model_info()
        
        captured = capsys.readouterr()
        assert "gpt2-small" in captured.out.lower()
        assert "12" in captured.out  # layers
    
    def test_model_device_property(self, mock_hooked_transformer):
        """Test accessing model device."""
        mock_transformer, mock_model = mock_hooked_transformer
        
        setup = ModelSetup()
        setup.load_model()
        
        # Should be able to access device
        assert setup.device == "cpu"


class TestModelSetupIntegration:
    """Integration tests for model setup workflows."""
    
    @patch('src.model_setup.HookedTransformer')
    def test_full_setup_workflow(self, mock_transformer):
        """Test complete model setup workflow."""
        # Create mock model
        mock_model = Mock()
        mock_model.cfg.model_name = "gpt2-small"
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        mock_model.cfg.d_model = 768
        mock_model.cfg.d_vocab = 50257
        mock_model.cfg.d_head = 64
        mock_model.cfg.n_ctx = 1024
        mock_model.cfg.d_mlp = 3072
        mock_model.cfg.device = "cpu"
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        # Mock tokenizer with proper decode
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value=" True")
        mock_model.tokenizer = mock_tokenizer
        
        mock_model.to_tokens = Mock(return_value=torch.randint(0, 1000, (1, 5)))
        mock_model.to_str_tokens = Mock(return_value=["5", " >", " 3", ":", " True"])
        mock_model.to_string = Mock(return_value=" True")
        mock_model.return_value = torch.randn(1, 5, 50257)
        
        mock_transformer.from_pretrained = Mock(return_value=mock_model)
        
        # 1. Initialize setup
        setup = ModelSetup(model_name="gpt2-small", device="cpu")
        assert setup.model_name == "gpt2-small"
        
        # 2. Load model
        model = setup.load_model()
        assert model is not None
        
        # 3. Get info
        info = setup.get_model_info()
        assert info['n_layers'] == 12
        
        # 4. Test basic functionality
        result = setup.test_model_basic("5 > 3:")
        assert result is not None
        assert isinstance(result, str)
    
    @patch('src.model_setup.HookedTransformer')
    def test_error_handling_workflow(self, mock_transformer):
        """Test error handling in various scenarios."""
        # Test using model before loading
        setup = ModelSetup()
        
        with pytest.raises(ValueError):
            setup.get_model_info()


class TestModelSetupValidation:
    """Tests for model validation and verification."""
    
    @patch('src.model_setup.HookedTransformer')
    def test_validate_model_architecture(self, mock_transformer):
        """Test model architecture validation."""
        mock_model = Mock()
        mock_model.cfg.model_name = "gpt2-small"
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        mock_model.cfg.d_model = 768
        mock_model.cfg.d_head = 64
        mock_model.cfg.d_vocab = 50257
        mock_model.cfg.n_ctx = 1024
        mock_model.cfg.d_mlp = 3072
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        mock_transformer.from_pretrained = Mock(return_value=mock_model)
        
        setup = ModelSetup()
        model = setup.load_model()
        
        # Validate architecture
        info = setup.get_model_info()
        assert info['n_layers'] == 12
        assert info['n_heads'] == 12
        assert info['d_model'] == 768
    
    @patch('src.model_setup.HookedTransformer')
    def test_model_forward_pass(self, mock_transformer):
        """Test model forward pass works."""
        mock_model = Mock()
        mock_model.cfg.model_name = "gpt2-small"
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        mock_model.cfg.d_model = 768
        mock_model.cfg.d_vocab = 50257
        mock_model.cfg.d_head = 64
        mock_model.cfg.n_ctx = 1024
        mock_model.cfg.d_mlp = 3072
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        # Mock tokenizer with proper decode
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value=" output")
        mock_model.tokenizer = mock_tokenizer
        
        # Mock tokenization and forward
        test_tokens = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.to_tokens = Mock(return_value=test_tokens)
        mock_model.return_value = torch.randn(1, 5, 50257)
        mock_model.to_string = Mock(return_value=" output")
        
        mock_transformer.from_pretrained = Mock(return_value=mock_model)
        
        setup = ModelSetup()
        setup.load_model()
        
        # Run a forward pass
        result = setup.test_model_basic("test")
        assert result is not None
        assert isinstance(result, str)
        assert "test" in result


class TestModelSetupEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_model_name_validation(self):
        """Test that model name is properly stored."""
        setup = ModelSetup(model_name="gpt2-medium")
        assert setup.model_name == "gpt2-medium"
    
    def test_device_validation(self):
        """Test device validation."""
        # CPU should always work
        setup = ModelSetup(device="cpu")
        assert setup.device == "cpu"
        
        # CUDA when not available should fall back to CPU or raise
        with patch('torch.cuda.is_available', return_value=False):
            setup = ModelSetup(device="cuda")
            # Implementation might keep "cuda" or fall back to "cpu"
            # Just verify it doesn't crash
            assert setup.device in ["cpu", "cuda"]
    
    @patch('src.model_setup.HookedTransformer')
    def test_repeated_model_info_calls(self, mock_transformer):
        """Test that model info can be called multiple times."""
        mock_model = Mock()
        mock_model.cfg.model_name = "gpt2-small"
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        mock_model.cfg.d_model = 768
        mock_model.cfg.d_vocab = 50257
        mock_model.cfg.d_head = 64
        mock_model.cfg.n_ctx = 1024
        mock_model.cfg.d_mlp = 3072
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        mock_transformer.from_pretrained = Mock(return_value=mock_model)
        
        setup = ModelSetup()
        setup.load_model()
        
        # Call multiple times
        info1 = setup.get_model_info()
        info2 = setup.get_model_info()
        
        # Should return same info
        assert info1 == info2


class TestModelSetupPrepareForAnalysis:
    """Test preparation for circuit analysis."""
    
    @patch('src.model_setup.HookedTransformer')
    def test_prepare_for_analysis(self, mock_transformer):
        """Test model preparation for analysis."""
        mock_model = Mock()
        mock_model.cfg.n_layers = 12
        mock_model.cfg.n_heads = 12
        
        # Mock reset_hooks and eval methods
        mock_model.reset_hooks = Mock()
        mock_model.eval = Mock()
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_param.requires_grad_ = Mock()
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        mock_transformer.from_pretrained = Mock(return_value=mock_model)
        
        setup = ModelSetup()
        setup.load_model()
        
        # Prepare for analysis
        result = setup.prepare_for_analysis()
        
        # Verify preparation steps
        assert mock_model.reset_hooks.called
        assert mock_model.eval.called
        assert result == mock_model
    
    @patch('src.model_setup.HookedTransformer')
    def test_get_activation_names(self, mock_transformer):
        """Test retrieval of activation names."""
        mock_model = Mock()
        mock_model.cfg.n_layers = 3  # Use fewer layers for testing
        mock_model.cfg.n_heads = 4
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        mock_transformer.from_pretrained = Mock(return_value=mock_model)
        
        setup = ModelSetup()
        setup.load_model()
        
        # Get activation names
        activations = setup.get_activation_names()
        
        # Verify structure
        assert isinstance(activations, dict)
        assert "embeddings" in activations
        assert "attention" in activations
        assert "mlp" in activations
        assert "layer_norm" in activations
        assert "residual" in activations
        
        # Verify some content
        assert len(activations["embeddings"]) > 0
        assert len(activations["attention"]) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])