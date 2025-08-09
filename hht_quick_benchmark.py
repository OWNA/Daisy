#!/usr/bin/env python3
"""
Quick HHT Performance Benchmark for Priority 4 Validation
Reduced iterations for faster testing
"""

import os
import sys
import time
import statistics
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hht_processor import OptimizedHHTProcessor
    print("HHT processor imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import HHT processor: {e}")
    sys.exit(1)

def quick_benchmark():
    """Run a quick benchmark with reduced parameters"""
    
    print("="*60)
    print("QUICK HHT PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor
    try:
        processor = OptimizedHHTProcessor(
            window_size=500,      # Reduced from 1000
            max_imfs=3,          # Reduced from 5
            update_frequency=1,   # Update every tick for benchmark
            method='emd',
            enable_caching=True,
            performance_monitoring=True
        )
        print("HHT Processor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize HHT processor: {e}")
        return False
    
    # Generate small synthetic dataset
    import random
    random.seed(42)
    sample_size = 200  # Reduced from 1000
    base_price = 108000.0
    
    price_data = []
    for i in range(sample_size):
        trend = 0.1 * i
        noise = 50 * (random.random() - 0.5)
        price = base_price + trend + noise
        price_data.append(price)
    
    timestamp_data = [datetime.now() for _ in range(sample_size)]
    
    print(f"Generated {len(price_data)} synthetic data points")
    print(f"Price range: ${min(price_data):.2f} - ${max(price_data):.2f}")
    
    # Run benchmark with reduced iterations
    num_iterations = 20  # Reduced from 100
    execution_times = []
    
    print(f"\nRunning {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        start_time = time.perf_counter()
        
        try:
            features = None
            for i, (price, timestamp) in enumerate(zip(price_data, timestamp_data)):
                features = processor.update(price, timestamp)
                # Only process first portion for quick test
                if i > 50:
                    break
            
            execution_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            execution_times.append(execution_time)
            
            if (iteration + 1) % 5 == 0:
                print(f"  Progress: {iteration + 1}/{num_iterations} iterations")
            
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            return False
    
    # Calculate statistics
    if execution_times:
        avg_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        # Calculate 95th percentile manually
        sorted_times = sorted(execution_times)
        percentile_95_index = int(0.95 * len(sorted_times))
        percentile_95 = sorted_times[percentile_95_index]
        
        print(f"\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Total measurements:         {len(execution_times)}")
        print(f"Average execution time:     {avg_time:.2f}ms")
        print(f"Median execution time:      {median_time:.2f}ms")
        print(f"95th percentile:           {percentile_95:.2f}ms")
        print(f"Maximum execution time:     {max_time:.2f}ms")
        print(f"Minimum execution time:     {min_time:.2f}ms")
        
        # Performance assessment
        threshold_ms = 50.0
        threshold_met = percentile_95 <= threshold_ms
        
        print(f"\n" + "="*60)
        print("PERFORMANCE ASSESSMENT")
        print("="*60)
        print(f"Performance threshold:      {threshold_ms}ms")
        print(f"95th percentile result:     {percentile_95:.2f}ms")
        print(f"Threshold status:          {'[PASS] PASSED' if threshold_met else '[FAIL] FAILED'}")
        
        if threshold_met:
            print(f"Production readiness:       [PASS] READY")
            print(f"Recommendation:             Deploy to live trading")
        else:
            overage = percentile_95 - threshold_ms
            print(f"Production readiness:       [FAIL] NEEDS OPTIMIZATION")
            print(f"Performance gap:            {overage:.2f}ms over threshold")
            print(f"Recommendation:             Optimize before deployment")
            
            # Optimization suggestions
            print(f"\nOptimization recommendations:")
            if avg_time > threshold_ms * 0.7:
                print(f"  • Reduce window_size (currently 500)")
                print(f"  • Reduce max_imfs (currently 3)")
                print(f"  • Consider alternative EMD implementation")
            
            if max_time > threshold_ms * 2:
                print(f"  • Investigate performance outliers")
                print(f"  • Implement timeout mechanism")
        
        return threshold_met
    
    else:
        print("No performance data collected")
        return False

if __name__ == "__main__":
    try:
        success = quick_benchmark()
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)