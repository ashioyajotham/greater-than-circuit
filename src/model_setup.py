"""
Model Setup Module

This module handles the loading and configuration of GPT-2 Small using TransformerLens
for mechanistic interpretability analysis of the greater than circuit.

Acknowledgment: Built using TransformerLens library developed by Neel Nanda and 
the mechanistic interpretability research community.
"""

import torch
from transformer_lens import HookedTransformer, utils
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSetup:
    """
    Handles the setup and configuration of GPT-2 Small for circuit analysis.
    
    This class provides functionality to load the model, configure it for
    mechanistic interpretability, and prepare it for activation patching experiments.
    """
    
    def __init__(self, model_name: str = "gpt2-small", device: Optional[str] = None):
        """
        Initialize the ModelSetup class.
        
        Args:
            model_name (str): Name of the model to load (default: "gpt2-small")
            device (str, optional): Device to run the model on. If None, auto-detect.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing ModelSetup for {model_name} on {self.device}")
    
    def load_model(self, **kwargs) -> HookedTransformer:
        """
        Load the GPT-2 Small model using TransformerLens.
        
        Returns:
            HookedTransformer: The loaded model ready for mechanistic interpretability
        """
        try:
            logger.info(f"Loading {self.model_name}...")
            
            # Load model with TransformerLens
            self.model = HookedTransformer.from_pretrained(
                self.model_name,
                center_unembed=True,  # Important for interpretability
                center_writing_weights=True,  # Centers the writing weights
                fold_ln=True,  # Fold layer norm into weights for cleaner analysis
                refactor_factored_attn_matrices=False,  # Keep attention matrices separate
                device=self.device,
                **kwargs
            )
            
            # Enable attention result hooks for head-level patching
            self.model.set_use_attn_result(True)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Store tokenizer for convenience
            self.tokenizer = self.model.tokenizer
            
            logger.info(f"Successfully loaded {self.model_name}")
            logger.info(f"Model configuration: {self.model.cfg}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model configuration and statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        info = {
            "model_name": self.model_name,
            "device": str(self.model.cfg.device),
            "n_layers": self.model.cfg.n_layers,
            "n_heads": self.model.cfg.n_heads,
            "d_model": self.model.cfg.d_model,
            "d_head": self.model.cfg.d_head,
            "d_mlp": self.model.cfg.d_mlp,
            "d_vocab": self.model.cfg.d_vocab,
            "n_ctx": self.model.cfg.n_ctx,
            "act_fn": self.model.cfg.act_fn,
            "normalization_type": self.model.cfg.normalization_type,
        }
        
        return info
    
    def print_model_info(self):
        """Print formatted model information."""
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return
        
        info = self.get_model_info()
        
        print(f"\n{'='*50}")
        print(f"MODEL INFORMATION: {info['model_name'].upper()}")
        print(f"{'='*50}")
        print(f"Device: {info['device']}")
        print(f"Layers: {info['n_layers']}")
        print(f"Attention Heads: {info['n_heads']}")
        print(f"Model Dimension: {info['d_model']}")
        print(f"Head Dimension: {info['d_head']}")
        print(f"MLP Dimension: {info['d_mlp']}")
        print(f"Vocabulary Size: {info['d_vocab']:,}")
        print(f"Context Length: {info['n_ctx']:,}")
        print(f"Activation Function: {info['act_fn']}")
        print(f"Normalization: {info['normalization_type']}")
        print(f"{'='*50}\n")
    
    def test_model_basic(self, test_prompt: str = "Hello, world!") -> str:
        """
        Test basic model functionality with a simple prompt.
        
        Args:
            test_prompt (str): Test prompt to use
            
        Returns:
            str: Model's completion of the prompt
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Testing model with prompt: '{test_prompt}'")
        
        # Tokenize the prompt
        tokens = self.model.to_tokens(test_prompt)
        
        # Generate completion
        with torch.no_grad():
            logits = self.model(tokens)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            next_word = self.tokenizer.decode([next_token])
        
        completion = test_prompt + next_word
        logger.info(f"Model completion: '{completion}'")
        
        return completion
    
    def prepare_for_analysis(self) -> HookedTransformer:
        """
        Prepare the model for circuit analysis by enabling various hooks and settings.
        
        Returns:
            HookedTransformer: The model prepared for analysis
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Clear any existing hooks
        self.model.reset_hooks()
        
        # Set model to evaluation mode (important for consistent behavior)
        self.model.eval()
        
        # Disable gradient computation for faster inference
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        logger.info("Model prepared for circuit analysis")
        return self.model
    
    def get_activation_names(self) -> Dict[str, list]:
        """
        Get all available activation names organized by component type.
        
        Returns:
            Dict[str, list]: Dictionary mapping component types to activation names
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        activation_names = {
            "embeddings": ["embed", "pos_embed"],
            "attention": [],
            "mlp": [],
            "layer_norm": [],
            "residual": []
        }
        
        for layer in range(self.model.cfg.n_layers):
            # Attention components
            for head in range(self.model.cfg.n_heads):
                activation_names["attention"].extend([
                    f"blocks.{layer}.attn.hook_q",
                    f"blocks.{layer}.attn.hook_k", 
                    f"blocks.{layer}.attn.hook_v",
                    f"blocks.{layer}.attn.hook_pattern",
                    f"blocks.{layer}.attn.hook_z",
                    f"blocks.{layer}.attn.hook_result"
                ])
            
            # MLP components
            activation_names["mlp"].extend([
                f"blocks.{layer}.mlp.hook_pre",
                f"blocks.{layer}.mlp.hook_post"
            ])
            
            # Layer norm components
            activation_names["layer_norm"].extend([
                f"blocks.{layer}.ln1.hook_scale",
                f"blocks.{layer}.ln1.hook_normalized", 
                f"blocks.{layer}.ln2.hook_scale",
                f"blocks.{layer}.ln2.hook_normalized"
            ])
            
            # Residual stream
            activation_names["residual"].extend([
                f"blocks.{layer}.hook_resid_pre",
                f"blocks.{layer}.hook_resid_mid", 
                f"blocks.{layer}.hook_resid_post"
            ])
        
        return activation_names


def main():
    """Example usage of the ModelSetup class."""
    # Initialize model setup
    setup = ModelSetup()
    
    # Load the model
    model = setup.load_model()
    
    # Print model information
    setup.print_model_info()
    
    # Test basic functionality
    test_result = setup.test_model_basic("The number 5 is greater than 3:")
    print(f"Test result: {test_result}")
    
    # Prepare for analysis
    model = setup.prepare_for_analysis()
    
    # Get activation names
    activations = setup.get_activation_names()
    print(f"Available activation types: {list(activations.keys())}")


if __name__ == "__main__":
    main()