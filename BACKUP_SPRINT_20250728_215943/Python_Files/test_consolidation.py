#!/usr/bin/env python3
"""
Test script to verify the consolidated system works correctly
"""

import os
import sys
import subprocess
import json
import yaml
from datetime import datetime

print("="*70)
print("TESTING CONSOLIDATED TRADING SYSTEM")
print("="*70)

# Test results
results = {
    "timestamp": datetime.now().isoformat(),
    "tests": []
}

def run_test(name, command, check_output=False):
    """Run a test command and record results"""
    print(f"\n🧪 Testing: {name}")
    print(f"   Command: {command}")
    
    try:
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        else:
            result = subprocess.run(command, shell=True, timeout=30)
        
        success = result.returncode == 0
        
        test_result = {
            "name": name,
            "command": command,
            "success": success,
            "return_code": result.returncode
        }
        
        if check_output:
            test_result["output"] = result.stdout[:500]  # First 500 chars
            test_result["error"] = result.stderr[:500]
        
        results["tests"].append(test_result)
        
        if success:
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED (return code: {result.returncode})")
            if check_output and result.stderr:
                print(f"   Error: {result.stderr[:200]}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  TIMEOUT")
        results["tests"].append({
            "name": name,
            "command": command,
            "success": False,
            "error": "Timeout after 30 seconds"
        })
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        results["tests"].append({
            "name": name,
            "command": command,
            "success": False,
            "error": str(e)
        })
        return False

# 1. Test imports
print("\n1️⃣  Testing Python imports...")
import_test = """
import modeltrainer
import datahandler
import featureengineer
import modelpredictor
import main
import cli
print('All imports successful')
"""

success = run_test(
    "Python imports",
    f'python -c "{import_test}"',
    check_output=True
)

# 2. Test CLI help
print("\n2️⃣  Testing CLI functionality...")
run_test("CLI help", "python cli.py --help", check_output=True)
run_test("Main help", "python main.py --help", check_output=True)

# 3. Test configuration
print("\n3️⃣  Testing configuration...")
if os.path.exists('config.yaml'):
    run_test("Config validation", "python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"", check_output=True)
else:
    print("   ⚠️  No config.yaml found")

# 4. Test model trainer import
print("\n4️⃣  Testing ModelTrainer...")
trainer_test = """
import yaml
from modeltrainer import ModelTrainer
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
trainer = ModelTrainer(config)
print(f'ModelTrainer initialized with {len(trainer.__dict__)} attributes')
print(f'Model path: {trainer.model_path}')
"""

run_test("ModelTrainer initialization", f'python -c "{trainer_test}"', check_output=True)

# 5. Test CLI status command
print("\n5️⃣  Testing CLI status command...")
run_test("CLI status", "python cli.py status", check_output=False)

# 6. Test main.py components
print("\n6️⃣  Testing main.py system initialization...")
system_test = """
import yaml
from main import TradingSystem
with open('config.yaml', 'r') as f:
    config_path = 'config.yaml'
system = TradingSystem(config_path)
print(f'TradingSystem initialized')
print(f'Database: {system.db is not None}')
print(f'Exchange: {system.exchange is not None}')
"""

run_test("TradingSystem initialization", f'python -c "{system_test}"', check_output=True)

# 7. Check file structure
print("\n7️⃣  Checking file structure...")
required_files = [
    "main.py",
    "cli.py",
    "modeltrainer.py",
    "modelpredictor.py",
    "datahandler.py",
    "featureengineer.py",
    "labelgenerator.py",
    "advancedriskmanager.py",
    "smartorderexecutor.py",
    "l2_data_collector.py",
    "database.py",
    "config.yaml"
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f"   ❌ Missing files: {missing_files}")
    results["tests"].append({
        "name": "File structure check",
        "success": False,
        "missing_files": missing_files
    })
else:
    print(f"   ✅ All {len(required_files)} required files present")
    results["tests"].append({
        "name": "File structure check",
        "success": True,
        "files_checked": len(required_files)
    })

# 8. Check archive structure
print("\n8️⃣  Checking archive structure...")
archive_dirs = [
    "ARCHIVE/debug_scripts",
    "ARCHIVE/execution_scripts",
    "ARCHIVE/test_scripts",
    "ARCHIVE/setup_scripts",
    "ARCHIVE/duplicate_systems",
    "ARCHIVE/utilities",
    "ARCHIVE/model_versions",
    "ARCHIVE/cli_versions",
    "ARCHIVE/l2_processing"
]

archive_count = 0
for dir in archive_dirs:
    if os.path.exists(dir):
        files = len([f for f in os.listdir(dir) if f.endswith('.py')])
        archive_count += files

print(f"   📁 Total files in archive: {archive_count}")

# Save results
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

total_tests = len(results["tests"])
passed_tests = sum(1 for t in results["tests"] if t.get("success", False))
failed_tests = total_tests - passed_tests

print(f"\nTotal tests: {total_tests}")
print(f"✅ Passed: {passed_tests}")
print(f"❌ Failed: {failed_tests}")

if failed_tests == 0:
    print("\n🎉 ALL TESTS PASSED! The consolidated system is working correctly.")
else:
    print(f"\n⚠️  {failed_tests} tests failed. Check test_results.json for details.")
    print("\nFailed tests:")
    for test in results["tests"]:
        if not test.get("success", False):
            print(f"  - {test['name']}")

print("\nNext steps:")
print("1. If all tests passed: Continue with Phase 3 consolidation")
print("2. If tests failed: Fix issues before proceeding")
print("3. Run a quick paper trading test: python cli.py trade --paper")
print("\nDetailed results saved to: test_results.json")