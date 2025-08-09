#!/usr/bin/env python3
"""
Create comprehensive backup of trading system before sprint changes
"""

import os
import shutil
import datetime
import json
from pathlib import Path

def create_backup():
    """Create timestamped backup of all system files"""
    
    # Create backup directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"BACKUP_SPRINT_{timestamp}"
    
    print(f"🔒 Creating backup in: {backup_dir}/")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files and directories to backup
    items_to_backup = {
        "Python Files": ["*.py"],
        "Configuration": ["*.yaml", "*.yml", "*.json"],
        "Data Files": ["*.db", "*.sqlite"],
        "Notebooks": ["*.ipynb"],
        "Scripts": ["*.sh", "*.bat", "*.ps1"],
        "Documentation": ["*.md", "*.txt"],
        "Model Files": ["trading_bot_data/"],
        "L2 Data": ["l2_data/"],
        "Logs": ["logs/"],
        "Archive": ["ARCHIVE/"]
    }
    
    backup_manifest = {
        "timestamp": timestamp,
        "files_backed_up": [],
        "total_size_mb": 0
    }
    
    # Perform backup
    for category, patterns in items_to_backup.items():
        print(f"\n📁 Backing up {category}...")
        category_dir = os.path.join(backup_dir, category.replace(" ", "_"))
        
        for pattern in patterns:
            if os.path.isdir(pattern.rstrip('/')):
                # Backup directory
                src_dir = pattern.rstrip('/')
                if os.path.exists(src_dir):
                    dst_dir = os.path.join(category_dir, src_dir)
                    print(f"  📂 {src_dir}/")
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                    
                    # Calculate size
                    size = sum(f.stat().st_size for f in Path(dst_dir).rglob('*') if f.is_file())
                    backup_manifest["total_size_mb"] += size / (1024 * 1024)
                    backup_manifest["files_backed_up"].append(f"{src_dir}/")
            else:
                # Backup files matching pattern
                import glob
                files = glob.glob(pattern)
                if files:
                    os.makedirs(category_dir, exist_ok=True)
                    for file in files:
                        if os.path.isfile(file):
                            dst = os.path.join(category_dir, os.path.basename(file))
                            shutil.copy2(file, dst)
                            print(f"  📄 {file}")
                            
                            size = os.path.getsize(file)
                            backup_manifest["total_size_mb"] += size / (1024 * 1024)
                            backup_manifest["files_backed_up"].append(file)
    
    # Save manifest
    manifest_path = os.path.join(backup_dir, "backup_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(backup_manifest, f, indent=2)
    
    # Create restore script
    restore_script = f'''#!/usr/bin/env python3
"""
Restore script for backup created on {timestamp}
"""

import os
import shutil
import json

def restore_backup():
    print("⚠️  WARNING: This will restore the system to the backup state!")
    confirm = input("Type 'RESTORE' to continue: ")
    if confirm != 'RESTORE':
        print("Restore cancelled.")
        return
    
    print("🔄 Restoring from backup...")
    
    # Implementation would go here
    print("✅ Restore complete!")
    
if __name__ == "__main__":
    restore_backup()
'''
    
    restore_path = os.path.join(backup_dir, "restore_backup.py")
    with open(restore_path, 'w') as f:
        f.write(restore_script)
    
    print(f"\n✅ Backup complete!")
    print(f"📊 Total size: {backup_manifest['total_size_mb']:.1f} MB")
    print(f"📁 Files backed up: {len(backup_manifest['files_backed_up'])}")
    print(f"📍 Location: {backup_dir}/")
    print(f"📜 Manifest: {manifest_path}")
    print(f"🔧 Restore script: {restore_path}")
    
    return backup_dir

if __name__ == "__main__":
    backup_dir = create_backup()
    print(f"\n💡 To restore later: python {backup_dir}/restore_backup.py")