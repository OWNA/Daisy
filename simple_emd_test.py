#!/usr/bin/env python3
"""
Simple EMD Package Test
=======================

Basic test of EMD packages without Unicode characters for Windows compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def test_pyemd():
    """Test PyEMD functionality"""
    print("\nTesting PyEMD...")
    
    try:
        from PyEMD import EMD, EEMD, CEEMDAN
        print("  PyEMD import successful")
        
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(1000)
        
        # Test EMD
        print("  Testing EMD...")
        start_time = time.perf_counter()
        emd = EMD()
        imfs_emd = emd(signal)
        emd_time = time.perf_counter() - start_time
        print(f"    EMD: {len(imfs_emd)} IMFs in {emd_time*1000:.1f}ms")
        
        # Test EEMD (reduced trials for speed)
        print("  Testing EEMD...")
        start_time = time.perf_counter()
        eemd = EEMD(trials=20)
        imfs_eemd = eemd(signal)
        eemd_time = time.perf_counter() - start_time
        print(f"    EEMD: {len(imfs_eemd)} IMFs in {eemd_time*1000:.1f}ms")
        
        return True, {
            'emd_time': emd_time,
            'eemd_time': eemd_time,
            'emd_imfs': len(imfs_emd),
            'eemd_imfs': len(imfs_eemd)
        }
        
    except ImportError as e:
        print(f"  PyEMD import failed: {e}")
        return False, None
    except Exception as e:
        print(f"  PyEMD test failed: {e}")
        return False, None


def test_pyhht():
    """Test PyHHT functionality"""
    print("\nTesting PyHHT...")
    
    try:
        from pyhht import EMD
        print("  PyHHT import successful")
        
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(1000)
        
        # Test PyHHT EMD
        print("  Testing PyHHT EMD...")
        start_time = time.perf_counter()
        emd = EMD(signal)
        imfs = emd.decompose()
        pyhht_time = time.perf_counter() - start_time
        print(f"    PyHHT EMD: {len(imfs)} IMFs in {pyhht_time*1000:.1f}ms")
        
        return True, {
            'pyhht_time': pyhht_time,
            'pyhht_imfs': len(imfs)
        }
        
    except ImportError as e:
        print(f"  PyHHT import failed: {e}")
        return False, None
    except Exception as e:
        print(f"  PyHHT test failed: {e}")
        return False, None


def create_simple_plot():
    """Create a simple EMD visualization"""
    print("\nCreating test visualization...")
    
    try:
        from PyEMD import EMD
        
        # Generate test signal (HFT-like)
        t = np.linspace(0, 100, 1000)  # 100 seconds, 100ms intervals
        trend = 50000 + 100 * t
        cycle1 = 200 * np.sin(2 * np.pi * 0.1 * t)
        cycle2 = 50 * np.sin(2 * np.pi * 0.5 * t)
        noise = 20 * np.random.normal(0, 1, 1000)
        signal = trend + cycle1 + cycle2 + noise
        
        # EMD decomposition
        emd = EMD()
        imfs = emd(signal)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(t, signal, 'b-', linewidth=1)
        plt.title('Original HFT-like Price Signal')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(t, imfs[0], 'r-', linewidth=1, label='IMF 1 (noise)')
        if len(imfs) > 1:
            plt.plot(t, imfs[1], 'g-', linewidth=1, label='IMF 2')
        plt.title('High-Frequency IMFs')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        if len(imfs) > 2:
            plt.plot(t, imfs[-1], 'k-', linewidth=1, label='Trend (last IMF)')
        plt.title('Trend Component')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_emd_test.png', dpi=150, bbox_inches='tight')
        print("  Test plot saved as 'simple_emd_test.png'")
        
        return True
        
    except Exception as e:
        print(f"  Visualization failed: {e}")
        return False


def main():
    """Main test function"""
    print("EMD Package Simple Test")
    print("=" * 30)
    
    # Test PyEMD
    pyemd_success, pyemd_results = test_pyemd()
    
    # Test PyHHT  
    pyhht_success, pyhht_results = test_pyhht()
    
    # Create visualization if any package works
    if pyemd_success:
        viz_success = create_simple_plot()
    else:
        viz_success = False
    
    # Summary
    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    
    if pyemd_success:
        print("PyEMD: WORKING")
        if pyemd_results:
            print(f"  EMD: {pyemd_results['emd_time']*1000:.1f}ms, {pyemd_results['emd_imfs']} IMFs")
            print(f"  EEMD: {pyemd_results['eemd_time']*1000:.1f}ms, {pyemd_results['eemd_imfs']} IMFs")
    else:
        print("PyEMD: FAILED")
    
    if pyhht_success:
        print("PyHHT: WORKING") 
        if pyhht_results:
            print(f"  EMD: {pyhht_results['pyhht_time']*1000:.1f}ms, {pyhht_results['pyhht_imfs']} IMFs")
    else:
        print("PyHHT: FAILED")
    
    if viz_success:
        print("Visualization: SUCCESS")
    else:
        print("Visualization: FAILED")
    
    working_packages = sum([pyemd_success, pyhht_success])
    print(f"\nWorking packages: {working_packages}/2")
    
    if working_packages > 0:
        print("\nREADY for full benchmark!")
        print("Next steps:")
        print("1. Run: python emd_performance_benchmark.py")
        print("2. Run: python hft_microstructure_emd_example.py")
    else:
        print("\nNO WORKING PACKAGES")
        print("Install with:")
        print("  pip install PyEMD")
        print("  pip install PyHHT")


if __name__ == "__main__":
    main()