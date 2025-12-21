"""
Chapter 22: Complete Super Resolution Learning Path
===================================================

This script runs through all sections of Chapter 22 in order,
demonstrating the progressive learning path from basics to advanced methods.

Usage:
    python run_chapter22.py [--section SECTION] [--all]
    
    --section: Run specific section (1-6)
    --all: Run all sections in sequence
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import all section modules using relative imports
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import modules
degradation_module = import_module_from_path(
    'image_degradation', 
    os.path.join(current_dir, '01_basics', 'image_degradation.py')
)
interpolation_module = import_module_from_path(
    'traditional_methods',
    os.path.join(current_dir, '02_interpolation', 'traditional_methods.py')
)
sisr_module = import_module_from_path(
    'simple_sr_network',
    os.path.join(current_dir, '03_single_image_sr', 'simple_sr_network.py')
)
vsr_module = import_module_from_path(
    'temporal_fusion',
    os.path.join(current_dir, '04_video_sr', 'temporal_fusion.py')
)
rbpn_module = import_module_from_path(
    'rbpn_network',
    os.path.join(current_dir, '05_rbpn', 'rbpn_network.py')
)
metrics_module = import_module_from_path(
    'metrics',
    os.path.join(current_dir, '06_evaluation', 'metrics.py')
)

# Get functions
demonstrate_degradation = degradation_module.demonstrate_degradation
demonstrate_interpolation = interpolation_module.demonstrate_interpolation
demonstrate_sisr = sisr_module.demonstrate_sisr
demonstrate_video_sr = vsr_module.demonstrate_video_sr
demonstrate_rbpn = rbpn_module.demonstrate_rbpn
evaluate_sr_result = metrics_module.evaluate_sr_result
print_evaluation_results = metrics_module.print_evaluation_results


def run_section_1():
    """Section 1: Understanding Image Degradation"""
    print("\n" + "="*70)
    print("SECTION 1: Understanding Image Degradation")
    print("="*70)
    demonstrate_degradation()


def run_section_2():
    """Section 2: Traditional Interpolation Methods"""
    print("\n" + "="*70)
    print("SECTION 2: Traditional Interpolation Methods")
    print("="*70)
    demonstrate_interpolation()


def run_section_3():
    """Section 3: Single Image Super Resolution"""
    print("\n" + "="*70)
    print("SECTION 3: Single Image Super Resolution (SISR)")
    print("="*70)
    demonstrate_sisr()


def run_section_4():
    """Section 4: Video Super Resolution"""
    print("\n" + "="*70)
    print("SECTION 4: Video Super Resolution (VSR)")
    print("="*70)
    demonstrate_video_sr()


def run_section_5():
    """Section 5: Recurrent Back-Projection Network"""
    print("\n" + "="*70)
    print("SECTION 5: Recurrent Back-Projection Network (RBPN)")
    print("="*70)
    demonstrate_rbpn()


def run_section_6():
    """Section 6: Evaluation Metrics"""
    print("\n" + "="*70)
    print("SECTION 6: Evaluation Metrics")
    print("="*70)
    print("\nThis section is integrated into the demonstrations above.")
    print("Metrics (PSNR, SSIM) are calculated when comparing methods.")
    print("\n✅ Evaluation metrics demonstration complete!")
    print()


def run_all_sections():
    """Run all sections in sequence"""
    print("\n" + "="*70)
    print("CHAPTER 22: COMPLETE SUPER RESOLUTION LEARNING PATH")
    print("="*70)
    print("\nThis will run all sections in order:")
    print("1. Understanding Image Degradation")
    print("2. Traditional Interpolation Methods")
    print("3. Single Image Super Resolution")
    print("4. Video Super Resolution")
    print("5. Recurrent Back-Projection Network")
    print("6. Evaluation Metrics")
    print("\n" + "="*70)
    
    run_section_1()
    run_section_2()
    run_section_3()
    run_section_4()
    run_section_5()
    run_section_6()
    
    print("\n" + "="*70)
    print("✅ CHAPTER 22 COMPLETE!")
    print("="*70)
    print("\nYou have learned:")
    print("✓ How images are degraded")
    print("✓ Traditional upsampling methods")
    print("✓ Deep learning for single image SR")
    print("✓ Using temporal information in video SR")
    print("✓ Advanced RBPN architecture")
    print("✓ How to evaluate SR quality")
    print("\nNext steps:")
    print("- Experiment with different scale factors")
    print("- Try training models on real datasets")
    print("- Explore advanced architectures (EDVR, BasicVSR)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run Chapter 22 Super Resolution demonstrations'
    )
    parser.add_argument(
        '--section', type=int, choices=[1, 2, 3, 4, 5, 6],
        help='Run specific section (1-6)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all sections in sequence'
    )
    
    args = parser.parse_args()
    
    if args.all:
        run_all_sections()
    elif args.section:
        section_map = {
            1: run_section_1,
            2: run_section_2,
            3: run_section_3,
            4: run_section_4,
            5: run_section_5,
            6: run_section_6
        }
        section_map[args.section]()
    else:
        print("Please specify --section <1-6> or --all")
        print("\nAvailable sections:")
        print("  1 - Understanding Image Degradation")
        print("  2 - Traditional Interpolation Methods")
        print("  3 - Single Image Super Resolution")
        print("  4 - Video Super Resolution")
        print("  5 - Recurrent Back-Projection Network")
        print("  6 - Evaluation Metrics")
        print("\nExample: python run_chapter22.py --section 1")
        print("         python run_chapter22.py --all")


if __name__ == "__main__":
    main()

