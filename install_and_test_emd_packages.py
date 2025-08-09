#!/usr/bin/env python3
"""
EMD Package Installation and Quick Test Script
=============================================

This script handles installation and basic testing of EMD packages
for high-frequency trading applications.

Packages tested:
- PyEMD (already in requirements.txt)
- PyHHT (will attempt to install)
- SciPy (for custom implementation)

Author: Trading System
Date: 2025-07-30
"""

import subprocess
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time


def install_package(package_name: str, pip_name: Optional[str] = None) -> bool:
    """Install a package using pip"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", pip_name
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ {package_name} installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}")
        print(f"Error: {e.stderr}")
        return False


def test_package_import(package_name: str, import_statement: str) -> Tuple[bool, Optional[str]]:
    """Test if a package can be imported"""
    try:
        exec(import_statement)
        print(f"✅ {package_name} import successful")
        return True, None
    except ImportError as e:
        print(f"❌ {package_name} import failed: {e}")
        return False, str(e)
    except Exception as e:
        print(f"❌ {package_name} unexpected error: {e}")
        return False, str(e)


def generate_test_signal(length: int = 1000, sampling_rate: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a test signal similar to HFT price data"""
    dt = 1.0 / sampling_rate  # 100ms intervals for HFT
    t = np.arange(length) * dt
    
    # Create a complex signal with multiple components
    # Base trend
    trend = 50000 + 100 * t
    
    # Multiple oscillatory components (like market cycles)
    cycle1 = 200 * np.sin(2 * np.pi * 0.1 * t)  # Slow cycle
    cycle2 = 50 * np.sin(2 * np.pi * 0.5 * t)   # Medium cycle
    cycle3 = 10 * np.sin(2 * np.pi * 2.0 * t)   # Fast cycle
    
    # Add noise (microstructure effects)
    noise = 20 * np.random.normal(0, 1, length)
    
    # Combine all components
    signal = trend + cycle1 + cycle2 + cycle3 + noise
    
    return t, signal


def test_pyemd():
    """Test PyEMD functionality"""
    print("\n🧪 Testing PyEMD...")
    
    try:
        from PyEMD import EMD, EEMD, CEEMDAN
        
        # Generate test signal
        t, signal = generate_test_signal(500)  # Shorter for faster testing
        
        print(f"📊 Test signal: {len(signal)} samples")
        
        # Test EMD
        print("  Testing EMD...")
        start_time = time.perf_counter()
        emd = EMD()
        imfs_emd = emd(signal)
        emd_time = time.perf_counter() - start_time
        print(f"    ✅ EMD: {len(imfs_emd)} IMFs in {emd_time*1000:.1f}ms")
        
        # Test EEMD (with reduced trials for speed)
        print("  Testing EEMD...")
        start_time = time.perf_counter()
        eemd = EEMD(trials=20)
        imfs_eemd = eemd(signal)
        eemd_time = time.perf_counter() - start_time
        print(f"    ✅ EEMD: {len(imfs_eemd)} IMFs in {eemd_time*1000:.1f}ms")
        
        # Test CEEMDAN (with reduced trials for speed)
        print("  Testing CEEMDAN...")
        start_time = time.perf_counter()
        ceemdan = CEEMDAN(trials=20)
        imfs_ceemdan = ceemdan(signal)
        ceemdan_time = time.perf_counter() - start_time
        print(f"    ✅ CEEMDAN: {len(imfs_ceemdan)} IMFs in {ceemdan_time*1000:.1f}ms")
        
        # Test reconstruction
        reconstruction_emd = np.sum(imfs_emd, axis=0)
        error_emd = np.mean((signal - reconstruction_emd) ** 2)
        print(f"    📊 EMD Reconstruction Error: {error_emd:.2e}")
        
        return True, {
            'emd_time': emd_time,
            'eemd_time': eemd_time,
            'ceemdan_time': ceemdan_time,
            'emd_imfs': len(imfs_emd),
            'eemd_imfs': len(imfs_eemd),
            'ceemdan_imfs': len(imfs_ceemdan),
            'reconstruction_error': error_emd
        }
        
    except Exception as e:
        print(f"    ❌ PyEMD test failed: {e}")
        return False, None


def test_pyhht():
    """Test PyHHT functionality"""
    print("\n🧪 Testing PyHHT...")
    
    try:
        from pyhht import EMD
        
        # Generate test signal
        t, signal = generate_test_signal(500)
        
        print(f"📊 Test signal: {len(signal)} samples")
        
        # Test PyHHT EMD
        print("  Testing PyHHT EMD...")
        start_time = time.perf_counter()
        emd = EMD(signal)
        imfs = emd.decompose()
        pyhht_time = time.perf_counter() - start_time
        print(f"    ✅ PyHHT EMD: {len(imfs)} IMFs in {pyhht_time*1000:.1f}ms")
        
        # Test reconstruction
        reconstruction = np.sum(imfs, axis=0)
        error = np.mean((signal - reconstruction) ** 2)
        print(f"    📊 Reconstruction Error: {error:.2e}")
        
        return True, {
            'pyhht_time': pyhht_time,
            'pyhht_imfs': len(imfs),
            'reconstruction_error': error
        }
        
    except Exception as e:
        print(f"    ❌ PyHHT test failed: {e}")
        return False, None


def create_test_visualization():
    """Create a simple visualization of EMD decomposition"""
    print("\n📈 Creating test visualization...")
    
    try:
        # Check if we have any working EMD implementation
        pyemd_available = False
        pyhht_available = False
        
        try:
            from PyEMD import EMD
            pyemd_available = True
        except ImportError:
            pass
        
        try:
            from pyhht import EMD as PyHHT_EMD
            pyhht_available = True
        except ImportError:
            pass
        
        if not (pyemd_available or pyhht_available):
            print("    ⚠️  No EMD packages available for visualization")
            return
        
        # Generate test signal
        t, signal = generate_test_signal(1000)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot original signal
        plt.subplot(3, 1, 1)
        plt.plot(t, signal, 'b-', linewidth=1)
        plt.title('Original HFT-like Price Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Try EMD decomposition and plotting
        if pyemd_available:
            try:
                from PyEMD import EMD
                emd = EMD()
                imfs = emd(signal)
                
                plt.subplot(3, 1, 2)
                plt.plot(t, imfs[0], 'r-', linewidth=1, label='IMF 1 (highest freq)')
                plt.plot(t, imfs[1], 'g-', linewidth=1, label='IMF 2')
                if len(imfs) > 2:
                    plt.plot(t, imfs[-1], 'k-', linewidth=1, label='Residue (trend)')
                plt.title('PyEMD - Selected IMFs')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Reconstruction
                reconstruction = np.sum(imfs, axis=0)
                plt.subplot(3, 1, 3)
                plt.plot(t, signal, 'b-', linewidth=1, label='Original', alpha=0.7)
                plt.plot(t, reconstruction, 'r--', linewidth=1, label='Reconstructed')
                plt.title('Original vs Reconstructed Signal')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                print("    ✅ PyEMD visualization created")
                
            except Exception as e:
                print(f"    ❌ PyEMD visualization failed: {e}")
        
        elif pyhht_available:
            try:
                from pyhht import EMD
                emd = EMD(signal)
                imfs = emd.decompose()
                
                plt.subplot(3, 1, 2)
                plt.plot(t, imfs[0], 'r-', linewidth=1, label='IMF 1 (highest freq)')
                if len(imfs) > 1:
                    plt.plot(t, imfs[1], 'g-', linewidth=1, label='IMF 2')
                if len(imfs) > 2:
                    plt.plot(t, imfs[-1], 'k-', linewidth=1, label='Residue (trend)')
                plt.title('PyHHT - Selected IMFs')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Reconstruction
                reconstruction = np.sum(imfs, axis=0)
                plt.subplot(3, 1, 3)
                plt.plot(t, signal, 'b-', linewidth=1, label='Original', alpha=0.7)
                plt.plot(t, reconstruction, 'r--', linewidth=1, label='Reconstructed')
                plt.title('Original vs Reconstructed Signal')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                print("    ✅ PyHHT visualization created")
                
            except Exception as e:
                print(f"    ❌ PyHHT visualization failed: {e}")
        
        plt.tight_layout()
        plt.savefig('emd_test_visualization.png', dpi=150, bbox_inches='tight')
        print("    💾 Visualization saved as 'emd_test_visualization.png'")
        
        # Don't show plot in automated environment
        # plt.show()
        
    except Exception as e:
        print(f"    ❌ Visualization creation failed: {e}")


def main():
    """Main execution function"""
    print("🔧 EMD Package Installation and Testing")
    print("=" * 40)
    
    # Package configurations
    packages_to_test = [
        {
            'name': 'PyEMD',
            'pip_name': 'PyEMD',
            'import_statement': 'from PyEMD import EMD, EEMD, CEEMDAN',
            'test_function': test_pyemd
        },
        {
            'name': 'PyHHT',
            'pip_name': 'PyHHT',
            'import_statement': 'from pyhht import EMD',
            'test_function': test_pyhht
        }
    ]
    
    results = {}
    
    for package_config in packages_to_test:
        name = package_config['name']
        print(f"\n📦 Processing {name}...")
        
        # Test import first
        import_success, import_error = test_package_import(
            name, package_config['import_statement']
        )
        
        if not import_success:
            print(f"🔄 Attempting to install {name}...")
            install_success = install_package(name, package_config['pip_name'])
            
            if install_success:
                # Test import again after installation
                import_success, import_error = test_package_import(
                    name, package_config['import_statement']
                )
        
        if import_success:
            # Run functionality test
            test_success, test_results = package_config['test_function']()
            results[name] = {
                'import_success': True,
                'test_success': test_success,
                'test_results': test_results
            }
        else:
            results[name] = {
                'import_success': False,
                'test_success': False,
                'import_error': import_error
            }
    
    # Summary report
    print("\n📊 SUMMARY REPORT")
    print("=" * 20)
    
    working_packages = []
    for name, result in results.items():
        status = "✅" if result['import_success'] and result.get('test_success', False) else "❌"
        print(f"{status} {name}")
        
        if result['import_success']:
            if result.get('test_success', False):
                working_packages.append(name)
                test_results = result.get('test_results', {})
                for key, value in test_results.items():
                    if 'time' in key:
                        print(f"    {key}: {value*1000:.1f}ms")
                    elif 'error' in key:
                        print(f"    {key}: {value:.2e}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"    Import OK, but functionality test failed")
        else:
            print(f"    Import failed: {result.get('import_error', 'Unknown error')}")
    
    print(f"\n🎯 Working packages: {len(working_packages)}/{len(packages_to_test)}")
    
    if working_packages:
        print(f"✅ Ready to run full benchmark with: {', '.join(working_packages)}")
        
        # Create test visualization
        create_test_visualization()
        
        print(f"\n🚀 Next steps:")
        print(f"1. Run the full benchmark: python emd_performance_benchmark.py")
        print(f"2. Check results in benchmark_results/ directory")
        
    else:
        print(f"❌ No working EMD packages found")
        print(f"📝 Installation suggestions:")
        print(f"   pip install PyEMD")
        print(f"   pip install PyHHT")
        print(f"   pip install scipy  # for custom implementations")
    
    return working_packages


if __name__ == "__main__":
    main()