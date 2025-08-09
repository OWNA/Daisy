#!/usr/bin/env python3
"""
Clean up project - Remove unnecessary files and organize
"""

import os
import shutil
from pathlib import Path
import json

print("🧹 Project Cleanup")
print("="*60)

# Files to remove
files_to_remove = [
    # Duplicate training scripts
    "train_model.py",
    "train_model_correctly.py", 
    "train_advanced_ml.py",
    "train_all_features.py",
    "train_on_real_l2.py",
    "train_on_clean_l2.py",
    "train_optuna_clean_l2.py",
    "train_full_l2_hht.py",
    "simple_train.py",
    "retrain_with_real_features.py",
    
    # Duplicate backtest scripts
    "run_backtest_working.py",
    "run_backtest_l2_fixed.py",
    "run_full_backtest.py",
    "run_comprehensive_backtest.py",
    "run_l2_only_backtest.py",
    
    # Test/debug scripts
    "test_*.py",  # Will handle with glob
    "debug_hht_features.py",
    "diagnose_training.py",
    "check_ml_quality.py",
    "check_prepared_data.py",
    "check_status.py",
    "run_minimal_test.py",
    "final_validation_test.py",
    
    # Fix/patch scripts
    "fix_configs.py",
    "fix_predictor.py",
    "fix_feature_mismatch.py",
    "fix_hht_features.py",
    
    # Temporary/accidental files
    "New Text Document.txt",
    "New Text Document (2).txt",
    "Claude.txt",
    "=2.1.0",
    "workflow_commands.sh",
    "setup.sh",
    
    # One-time analysis scripts
    "analyze_l2_collection.py",
    "analyze_l2_spread_frequency.py",
    "analyze_predictions.py",
    "analyze_backtest_results.py",
    "analyze_feature_importance.py",
    "compare_l2_performance.py",
    "generate_shap_analysis.py",
    
    # Deprecated WebSocket tests
    "run_websocket_simple.py",
    "run_websocket_simulation.py",
    "test_websocket_sim.py",
    "run_live_test.py",
    "run_live_trading_sim.py",
    "create_compatible_model.py",
    
    # Data processing scripts
    "align_l2_data.py",
    "convert_l2_to_csv.py",
    "prepare_l2_training.py",
    "download_sample_data.py",
    "upload_csv_to_db.py",
    "read_parquet.py",
    "collect_l2_data.py",
    "test_l2_collection.py",
    
    # Cleanup utilities
    "cleanup_duplicate_files.py",
    "organize_output_files.py",
    "install_dependencies.py",
    
    # Unused L2 scripts
    "l2_alignment_diagnostics.py",
    "l2_diagnostics.py",
    "l2_gap_analysis.py",
    "l2_backtest_analyzer.py",
    "l2_paper_trading_validator.py",
    "l2_performance_benchmarker.py",
    "l2_price_reconstructor.py",
    "l2_production_deployment_manager.py",
    "monitor_l2_collection.py",
    
    # Documentation (keep README.md)
    "cursor_trading_bot_project_assistance.md",
    "query_for_tomorrow.md",
    "dependency_resolution_report.md",
    "training_results_report.md",
    "user_flows_error_report.md",
    "project_dependency_analysis.md",
    "L2_MODEL_SIMULATION_SUCCESS_REPORT.md",
    "WEBSOCKET_FIX_SUMMARY.md",
    "ML_PIPELINE_STATUS.md",
    "L2_HHT_FIX_COMPLETE.md",
    
    # Command files
    "run_advanced_training.cmd",
    "test_simple_websocket.cmd", 
    "train_and_test_l2_hht.cmd",
    "train_windows.bat",
    
    # Old menus
    "menu.ps1",
    "trade_menu.py",
    
    # Duplicate models (keep only the best one)
    "lgbm_model_BTC_USDTUSDT_all_features.txt",
    "lgbm_model_BTC_USDTUSDT_clean_l2_hht.txt",
    "lgbm_model_BTC_USDTUSDT_real_l2.txt",
    "lgbm_model_BTC_USDTUSDT_advanced.txt",
    "lgbm_model_BTC_USDTUSDT_None.txt",
    "model_features_BTC_USDTUSDT_None.json",
    "model_features_BTC_USDTUSDT_all_features.json",
    "model_features_BTC_USDTUSDT_advanced.json",
    "model_features_BTC_USDTUSDT_real_l2.json",
    "model_features_BTC_USDTUSDT_clean_l2_hht.json",
    
    # Duplicate configs (keep only config.yaml)
    "config_l2.yaml",
    "config_l2_training.yaml", 
    "config_live_sim.yaml",
    "config_no_l2.yaml",
    "config_wfo.yaml",
    "config_with_l2.yaml",
    
    # Strategy guides
    "L2_ONLY_STRATEGY_GUIDE.md",
    "NEXT_STEPS_TO_PRODUCTION.md",
    
    # Other files
    "run_trading_simple.py",
    "run_complete_workflow.py",
    "model_tester.py",
    "stress_test_suite.py",
    "demo_results/",
    "legacy/",
    "phase2_extras/"
]

# Best model to keep
best_model = "lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt"
best_features = "model_features_BTC_USDTUSDT_optuna_l2_hht.json"

# Backup important files before cleanup
backup_dir = Path("backup_before_cleanup")
backup_dir.mkdir(exist_ok=True)

# Backup the best model and config
if os.path.exists(best_model):
    shutil.copy2(best_model, backup_dir)
if os.path.exists(best_features):
    shutil.copy2(best_features, backup_dir)
if os.path.exists("config_l2_only.yaml"):
    shutil.copy2("config_l2_only.yaml", backup_dir / "config.yaml")

# Remove files
removed_count = 0
for file_pattern in files_to_remove:
    if "*" in file_pattern:
        # Handle glob patterns
        for file in Path(".").glob(file_pattern):
            if file.is_file():
                try:
                    file.unlink()
                    print(f"  ❌ Removed: {file}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠️  Could not remove {file}: {e}")
    else:
        # Handle specific files
        if os.path.exists(file_pattern):
            try:
                if os.path.isdir(file_pattern):
                    shutil.rmtree(file_pattern)
                else:
                    os.remove(file_pattern)
                print(f"  ❌ Removed: {file_pattern}")
                removed_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not remove {file_pattern}: {e}")

# Rename config_l2_only.yaml to config.yaml
if os.path.exists("config_l2_only.yaml") and not os.path.exists("config.yaml"):
    shutil.move("config_l2_only.yaml", "config.yaml")
    print("\n  ✅ Renamed config_l2_only.yaml to config.yaml")

# Clean up empty directories
for dir_path in ["demo_results", "legacy", "phase2_extras", "test_results"]:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"  ❌ Removed directory: {dir_path}")
        except:
            pass

print(f"\n📊 Cleanup Summary:")
print(f"  Files removed: {removed_count}")
print(f"  Best model kept: {best_model}")
print(f"  Config: config.yaml")
print(f"  Backup created in: {backup_dir}")

print("\n✅ Cleanup complete!")
print("="*60)