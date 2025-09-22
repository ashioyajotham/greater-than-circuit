"""
Main execution script for the Greater Than Circuit analysis.

This script runs the complete pipeline from model setup to circuit validation,
providing a comprehensive analysis of the greater than capability in GPT-2 Small.

Usage:
    python main.py [--n_examples 200] [--output_dir results] [--seed 42]

Acknowledgment: This implementation builds upon the foundational work of
Neel Nanda and the mechanistic interpretability research community.
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_setup import ModelSetup
from src.prompt_design import PromptGenerator
from src.activation_patching import ActivationPatcher
from src.circuit_analysis import CircuitAnalyzer
from src.visualization import CircuitVisualizer
from src.circuit_validation import CircuitValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('circuit_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_output_directory(output_dir: str) -> Path:
    """
    Create and setup the output directory structure.
    
    Args:
        output_dir (str): Path to output directory
        
    Returns:
        Path: Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)
    
    logger.info(f"Output directory setup complete: {output_path.absolute()}")
    return output_path


def run_circuit_analysis(
    n_examples: int = 200,
    output_dir: str = "results", 
    seed: int = 42,
    quick_mode: bool = False
) -> dict:
    """
    Run the complete greater than circuit analysis pipeline.
    
    Args:
        n_examples (int): Number of test examples to generate
        output_dir (str): Directory to save results
        seed (int): Random seed for reproducibility
        quick_mode (bool): If True, run faster analysis with fewer components
        
    Returns:
        dict: Dictionary containing all analysis results
    """
    logger.info("=" * 80)
    logger.info("GREATER THAN CIRCUIT ANALYSIS - STARTING")
    logger.info("=" * 80)
    logger.info(f"Parameters: n_examples={n_examples}, seed={seed}, quick_mode={quick_mode}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir)
    
    results = {}
    
    try:
        # 1. Model Setup
        logger.info("\n🔧 STEP 1: Model Setup")
        start_time = time.time()
        
        setup = ModelSetup()
        model = setup.load_model()
        setup.print_model_info()
        
        setup_time = time.time() - start_time
        logger.info(f"Model setup completed in {setup_time:.2f} seconds")
        results['model_info'] = setup.get_model_info()
        
        # 2. Prompt Generation
        logger.info("\n📝 STEP 2: Prompt Generation")
        start_time = time.time()
        
        generator = PromptGenerator(seed=seed)
        
        # Generate test examples
        test_examples = generator.generate_balanced_dataset(n_examples=n_examples)
        edge_cases = generator.generate_edge_cases()
        
        # Create prompt pairs for patching
        n_pairs = 20 if quick_mode else 50
        prompt_pairs = generator.create_prompt_pairs(n_pairs=n_pairs)
        
        # Save examples
        generator.save_examples(test_examples, output_path / "data" / "test_examples.csv")
        generator.save_examples(edge_cases, output_path / "data" / "edge_cases.csv")
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {len(test_examples)} test examples and {len(prompt_pairs)} prompt pairs in {generation_time:.2f} seconds")
        
        results['test_examples'] = test_examples
        results['prompt_pairs'] = prompt_pairs
        
        # 3. Initialize Analysis Components
        logger.info("\n🔍 STEP 3: Initialize Analysis Components")
        
        patcher = ActivationPatcher(model)
        analyzer = CircuitAnalyzer(model)
        visualizer = CircuitVisualizer(output_dir=str(output_path))
        validator = CircuitValidator(model, generator, patcher, analyzer)
        
        # 4. Baseline Validation
        logger.info("\n📊 STEP 4: Baseline Validation")
        start_time = time.time()
        
        baseline_result = validator.validate_baseline_accuracy(test_examples[:100])
        logger.info(f"Baseline accuracy: {baseline_result.accuracy:.3f} ({baseline_result.correct_predictions}/{baseline_result.total_examples})")
        
        baseline_time = time.time() - start_time
        results['baseline_result'] = baseline_result
        
        # 5. Activation Patching
        logger.info("\n🧠 STEP 5: Activation Patching")
        start_time = time.time()
        
        # Use first few prompt pairs for comprehensive analysis
        patching_results = []
        
        n_analysis_pairs = 3 if quick_mode else 10
        for i, (clean_example, corrupted_example) in enumerate(prompt_pairs[:n_analysis_pairs]):
            logger.info(f"  Processing pair {i+1}/{n_analysis_pairs}")
            
            clean_tokens = model.to_tokens(clean_example.prompt_text + " ")
            corrupted_tokens = model.to_tokens(corrupted_example.prompt_text + " ")
            
            # Patch attention heads (most informative for this task)
            pair_results = patcher.patch_attention_heads(
                corrupted_tokens=corrupted_tokens,
                clean_tokens=clean_tokens,
                positions=[-1]  # Focus on last position
            )
            patching_results.extend(pair_results)
        
        patching_time = time.time() - start_time
        logger.info(f"Completed {len(patching_results)} patching experiments in {patching_time:.2f} seconds")
        
        results['patching_results'] = patching_results
        
        # 6. Circuit Analysis
        logger.info("\n🔬 STEP 6: Circuit Analysis")
        start_time = time.time()
        
        # Identify circuit components
        circuit_components = analyzer.identify_circuit_components(
            patching_results,
            importance_threshold=0.05,
            top_k=20
        )
        
        # Analyze layer contributions
        layer_contributions = analyzer.analyze_layer_contributions(patching_results)
        
        # Create comprehensive summary
        circuit_summary = analyzer.create_circuit_summary(patching_results)
        
        analysis_time = time.time() - start_time
        logger.info(f"Identified {len(circuit_components)} circuit components across {circuit_summary['circuit_depth']} layers in {analysis_time:.2f} seconds")
        
        results['circuit_components'] = circuit_components
        results['layer_contributions'] = layer_contributions
        results['circuit_summary'] = circuit_summary
        
        # 7. Visualization
        logger.info("\n📈 STEP 7: Visualization")
        start_time = time.time()
        
        # Create visualizations
        fig1 = visualizer.plot_patching_results(
            patching_results,
            title="Greater Than Circuit - Patching Results",
            save_path=output_path / "plots" / "patching_results.png",
            top_k=20
        )
        
        fig2 = visualizer.plot_layer_importance(
            layer_contributions,
            title="Layer Contributions to Greater Than Circuit",
            save_path=output_path / "plots" / "layer_importance.png"
        )
        
        fig3 = visualizer.plot_circuit_diagram(
            circuit_components,
            title="Greater Than Circuit Structure",
            save_path=output_path / "plots" / "circuit_diagram.html"
        )
        
        fig4 = visualizer.create_summary_dashboard(
            circuit_summary,
            save_path=output_path / "plots" / "dashboard.html"
        )
        
        viz_time = time.time() - start_time
        logger.info(f"Generated visualizations in {viz_time:.2f} seconds")
        
        # 8. Comprehensive Validation
        logger.info("\n✅ STEP 8: Circuit Validation")
        start_time = time.time()
        
        if not quick_mode:
            # Run comprehensive validation
            validation_results = validator.run_comprehensive_validation(n_test_examples=min(200, n_examples))
        else:
            # Quick validation
            validation_results = {
                'baseline': baseline_result,
                'necessity': validator.validate_circuit_necessity(
                    test_examples[:30], circuit_components
                ),
                'robustness_edge_cases': validator.validate_robustness(
                    test_examples[:50], ['edge_cases']
                )[0]
            }
        
        # Generate validation report
        report = validator.generate_validation_report(
            validation_results,
            save_path=output_path / "reports" / "validation_report.txt"
        )
        
        validation_time = time.time() - start_time
        logger.info(f"Completed validation in {validation_time:.2f} seconds")
        
        results['validation_results'] = validation_results
        results['validation_report'] = report
        
        # 9. Final Summary
        logger.info("\n🎯 FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✓ Baseline Accuracy: {baseline_result.accuracy:.1%}")
        logger.info(f"✓ Circuit Components Identified: {len(circuit_components)}")
        logger.info(f"✓ Circuit Depth: {circuit_summary['circuit_depth']} layers")
        logger.info(f"✓ Most Important Layer: {circuit_summary['circuit_overview']['most_important_layer']}")
        
        # Top components
        logger.info("✓ Top Circuit Components:")
        for i, (name, comp) in enumerate(list(circuit_components.items())[:5]):
            logger.info(f"   {i+1}. {name}: Layer {comp.layer}, Importance {comp.importance_score:.3f}")
        
        # Validation summary
        if 'necessity' in validation_results:
            necessity_score = validation_results['necessity'].details.get('necessity_score', 0)
            logger.info(f"✓ Circuit Necessity Score: {necessity_score:.3f}")
        
        logger.info(f"✓ Results saved to: {output_path.absolute()}")
        logger.info("=" * 60)
        
        # Save final results summary
        summary_data = {
            'baseline_accuracy': baseline_result.accuracy,
            'n_components': len(circuit_components),
            'circuit_depth': circuit_summary['circuit_depth'],
            'most_important_layer': circuit_summary['circuit_overview']['most_important_layer'],
            'top_components': [
                {
                    'name': name,
                    'layer': comp.layer,
                    'importance': comp.importance_score,
                    'type': comp.component_type
                }
                for name, comp in list(circuit_components.items())[:10]
            ]
        }
        
        import json
        with open(output_path / "reports" / "analysis_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info("\n🙏 Acknowledgment: This analysis builds upon the foundational work")
        logger.info("   of Neel Nanda and the mechanistic interpretability community.")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """Main entry point for the circuit analysis."""
    parser = argparse.ArgumentParser(
        description="Reverse-engineer the greater than circuit in GPT-2 Small",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n_examples', 
        type=int, 
        default=200,
        help='Number of test examples to generate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save analysis results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick mode (faster but less comprehensive)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    try:
        results = run_circuit_analysis(
            n_examples=args.n_examples,
            output_dir=args.output_dir,
            seed=args.seed,
            quick_mode=args.quick
        )
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {os.path.abspath(args.output_dir)}")
        print("📊 Check the generated plots and reports for detailed findings.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()