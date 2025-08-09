#!/usr/bin/env python3
"""
EMD Testing Suite Runner
========================

This script runs all the EMD testing and benchmarking scripts in sequence.
It provides a comprehensive test of EMD packages for HFT applications.

Author: Trading System
Date: 2025-07-30
"""

import subprocess
import sys
import os
from pathlib import Path
import time


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📁 Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=False, text=True, cwd=Path.cwd())
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"\n❌ {description} failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_name}")
        return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False


def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Main execution function"""
    print("🧪 EMD Testing Suite Runner")
    print("=" * 40)
    print("This will run a comprehensive test of EMD packages for HFT applications")
    print()
    
    # Check basic requirements
    if not check_requirements():
        print("❌ Basic requirements not met. Please install missing packages.")
        return
    
    # List of scripts to run
    test_scripts = [
        {
            'script': 'install_and_test_emd_packages.py',
            'description': 'Package Installation and Basic Testing',
            'required': True
        },
        {
            'script': 'emd_performance_benchmark.py', 
            'description': 'Comprehensive Performance Benchmark',
            'required': False
        },
        {
            'script': 'hft_microstructure_emd_example.py',
            'description': 'HFT Microstructure Analysis Example',
            'required': False
        }
    ]
    
    results = {}
    
    for test_config in test_scripts:
        script_name = test_config['script']
        description = test_config['description']
        required = test_config['required']
        
        # Check if script exists
        if not Path(script_name).exists():
            print(f"\n⚠️  Script not found: {script_name}")
            results[script_name] = False
            continue
        
        # Run the script
        success = run_script(script_name, description)
        results[script_name] = success
        
        # If this is a required script and it failed, ask user if they want to continue
        if required and not success:
            response = input(f"\n❓ Required script '{script_name}' failed. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("🛑 Testing stopped by user")
                break
        
        # Brief pause between scripts
        time.sleep(2)
    
    # Summary report
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_scripts = len(results)
    successful_scripts = sum(results.values())
    
    print(f"Total scripts: {total_scripts}")
    print(f"Successful: {successful_scripts}")
    print(f"Failed: {total_scripts - successful_scripts}")
    print(f"Success rate: {(successful_scripts/total_scripts)*100:.1f}%")
    
    print(f"\nDetailed results:")
    for script, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {script}")
    
    if successful_scripts == total_scripts:
        print(f"\n🎉 All tests completed successfully!")
        print(f"\n📁 Check the following directories for results:")
        print(f"  • benchmark_results/ - Performance benchmark data")
        print(f"  • *.png files - Visualization plots")
        print(f"  • *.csv files - Analysis data")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n🏁 EMD testing suite complete!")


if __name__ == "__main__":
    main()