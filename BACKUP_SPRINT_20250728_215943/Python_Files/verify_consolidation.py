#!/usr/bin/env python3
"""
Simple verification of consolidated system
"""

import os
import ast
import json

print("="*60)
print("VERIFYING CONSOLIDATED SYSTEM")
print("="*60)

# Track results
issues = []
successes = []

# 1. Check core files exist
print("\n1. Checking core files...")
core_files = {
    "main.py": "Main entry point",
    "cli.py": "Unified CLI",
    "modeltrainer.py": "Consolidated model training",
    "modelpredictor.py": "Model prediction",
    "datahandler.py": "Data handling",
    "featureengineer.py": "Feature engineering",
    "labelgenerator.py": "Label generation",
    "advancedriskmanager.py": "Risk management",
    "smartorderexecutor.py": "Order execution",
    "l2_data_collector.py": "L2 data collection",
    "database.py": "Database operations",
    "config.yaml": "Configuration"
}

for file, desc in core_files.items():
    if os.path.exists(file):
        size_kb = os.path.getsize(file) / 1024
        print(f"  ✅ {file:<25} ({size_kb:6.1f} KB) - {desc}")
        successes.append(f"{file} exists")
    else:
        print(f"  ❌ {file:<25} MISSING - {desc}")
        issues.append(f"{file} is missing")

# 2. Check Python syntax in key files
print("\n2. Checking Python syntax...")
python_files = ["main.py", "cli.py", "modeltrainer.py"]

for file in python_files:
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                content = f.read()
            ast.parse(content)
            print(f"  ✅ {file} - Valid Python syntax")
            successes.append(f"{file} has valid syntax")
        except SyntaxError as e:
            print(f"  ❌ {file} - Syntax error: {e}")
            issues.append(f"{file} has syntax error: {e}")

# 3. Check modeltrainer.py improvements
print("\n3. Checking modeltrainer.py enhancements...")
if os.path.exists("modeltrainer.py"):
    with open("modeltrainer.py", 'r') as f:
        content = f.read()
    
    enhancements = {
        "import logging": "Logging support",
        "from typing import": "Type hints",
        "TimeSeriesSplit": "Time series validation",
        "_log_feature_importance": "Feature importance",
        "validate_model": "Model validation",
        "logger.": "Logger usage"
    }
    
    for feature, desc in enhancements.items():
        if feature in content:
            print(f"  ✅ {desc}")
            successes.append(f"ModelTrainer has {desc}")
        else:
            print(f"  ⚠️  {desc} - Not found")

# 4. Check CLI features
print("\n4. Checking cli.py features...")
if os.path.exists("cli.py"):
    with open("cli.py", 'r') as f:
        content = f.read()
    
    cli_features = {
        "argparse": "Command-line parsing",
        "interactive_menu": "Interactive menu",
        "collect_data": "Data collection command",
        "train_model": "Training command",
        "paper_trade": "Paper trading command",
        "check_status": "Status command",
        "rich": "Rich console (optional)"
    }
    
    for feature, desc in cli_features.items():
        if feature in content:
            print(f"  ✅ {desc}")
            successes.append(f"CLI has {desc}")
        else:
            print(f"  ⚠️  {desc} - Not found")

# 5. Check archive structure
print("\n5. Checking archive structure...")
archive_dirs = {
    "ARCHIVE/model_versions": 0,
    "ARCHIVE/cli_versions": 0,
    "ARCHIVE/debug_scripts": 0,
    "ARCHIVE/execution_scripts": 0,
    "ARCHIVE/test_scripts": 0
}

total_archived = 0
for dir_path in archive_dirs:
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.endswith('.py')]
        count = len(files)
        archive_dirs[dir_path] = count
        total_archived += count
        if count > 0:
            print(f"  ✅ {dir_path:<35} - {count} files")
            successes.append(f"{dir_path} has {count} archived files")
    else:
        print(f"  ⚠️  {dir_path:<35} - Directory not found")

print(f"\n  📊 Total archived files: {total_archived}")

# 6. Check imports between modules
print("\n6. Checking module dependencies...")
import_checks = [
    ("main.py", "from modeltrainer import ModelTrainer"),
    ("main.py", "from datahandler import DataHandler"),
    ("main.py", "from featureengineer import FeatureEngineer"),
    ("cli.py", "import subprocess"),
    ("cli.py", "import argparse")
]

for file, import_line in import_checks:
    if os.path.exists(file):
        with open(file, 'r') as f:
            content = f.read()
        if import_line in content:
            print(f"  ✅ {file} imports {import_line.split()[-1]}")
            successes.append(f"{file} has correct imports")
        else:
            print(f"  ❌ {file} missing: {import_line}")
            issues.append(f"{file} missing import: {import_line}")

# Summary
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)

print(f"\n✅ Successes: {len(successes)}")
print(f"❌ Issues: {len(issues)}")

if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n🎉 No issues found! Consolidation successful.")

print(f"\n📊 Statistics:")
print(f"  - Core files: {len([f for f in core_files if os.path.exists(f)])}/{len(core_files)}")
print(f"  - Files archived: {total_archived}")
print(f"  - Consolidation ratio: ~{100 - (len(core_files) / (len(core_files) + total_archived) * 100):.0f}% reduction")

# Save verification report
report = {
    "timestamp": os.popen('date').read().strip(),
    "core_files_found": len([f for f in core_files if os.path.exists(f)]),
    "core_files_total": len(core_files),
    "files_archived": total_archived,
    "successes": len(successes),
    "issues": len(issues),
    "issue_details": issues
}

with open('verification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n📄 Detailed report saved to: verification_report.json")

# Test commands to run
print("\n🚀 Commands to test the system:")
print("  python3 cli.py status")
print("  python3 cli.py --help")
print("  python3 main.py --help")
print("  python3 cli.py menu        # Interactive mode")