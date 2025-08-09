#!/usr/bin/env python3
"""
Test script for L2 Visualizer
Tests the L2-only visualization system with existing database data.
"""

import yaml
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def test_l2_visualizer():
    """Test the L2 visualizer with existing data."""
    print("="*60)
    print("TESTING L2 VISUALIZER")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Load configuration
        with open('config_l2_only.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded config for symbol: {config.get('symbol')}")
        
        # Try to import matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            has_matplotlib = True
            print("✓ Matplotlib available")
        except ImportError:
            plt = None
            has_matplotlib = False
            print("✗ Matplotlib not available")
        
        # Try to import SHAP
        try:
            import shap
            has_shap = True
            print("✓ SHAP available")
        except ImportError:
            shap = None
            has_shap = False
            print("✗ SHAP not available")
        
        # Import the L2 visualizer
        from visualizer_l2 import L2Visualizer
        
        # Override symbol to match what's in the database
        config_override = config.copy()
        config_override['symbol'] = 'BTCUSDT'  # Match database symbol
        
        # Initialize visualizer
        visualizer = L2Visualizer(
            config=config_override,
            has_matplotlib=has_matplotlib,
            plt_module=plt,
            has_shap=has_shap,
            shap_module=shap,
            db_path=config.get('database_path', 'trading_bot.db')
        )
        
        print("✓ L2 Visualizer initialized")
        
        # Test data loading
        print("\nTesting data loading...")
        df = visualizer.load_l2_data(limit=1000)
        
        if df.empty:
            print("✗ No L2 data found in database")
            print("Make sure l2_etl_processor_fixed.py has been run")
            return False
        
        print(f"✓ Loaded {len(df)} L2 records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Columns: {list(df.columns)}")
        
        # Test individual plot functions if matplotlib is available
        if has_matplotlib:
            print("\nTesting plot generation...")
            
            try:
                visualizer.plot_orderbook_heatmap(df, time_window_minutes=30)
                print("✓ Order book heatmap generated")
            except Exception as e:
                print(f"✗ Order book heatmap failed: {e}")
            
            try:
                visualizer.plot_microstructure_features(df)
                print("✓ Microstructure features plot generated")
            except Exception as e:
                print(f"✗ Microstructure features plot failed: {e}")
            
            try:
                visualizer.plot_data_quality_dashboard(df)
                print("✓ Data quality dashboard generated")
            except Exception as e:
                print(f"✗ Data quality dashboard failed: {e}")
            
            # Test generate all plots
            try:
                visualizer.generate_all_l2_plots(limit=1000)
                print("✓ All L2 plots generated successfully")
            except Exception as e:
                print(f"✗ Generate all plots failed: {e}")
        
        else:
            print("Skipping plot tests - matplotlib not available")
        
        # Check if plot files were created
        print("\nChecking generated plot files...")
        plot_files = [
            visualizer.l2_orderbook_heatmap_path,
            visualizer.l2_microstructure_path,
            visualizer.l2_data_quality_path
        ]
        
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                file_size = os.path.getsize(plot_file) / 1024  # KB
                print(f"✓ {os.path.basename(plot_file)} ({file_size:.1f} KB)")
            else:
                print(f"✗ {os.path.basename(plot_file)} not found")
        
        print("\n" + "="*60)
        print("L2 VISUALIZER TEST COMPLETED")
        print("="*60)
        print(f"Completed at: {datetime.now()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_l2_visualizer()
    sys.exit(0 if success else 1)
