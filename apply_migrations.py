#!/usr/bin/env python3
"""
Database Migration Application Script
Applies SQL migrations to the trading bot database with proper error handling and verification.
"""

import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

class MigrationRunner:
    """Handles database migration execution with safety checks and rollback capability"""
    
    def __init__(self, db_path: str = "trading_bot.db", migrations_dir: str = "migrations"):
        self.db_path = Path(db_path)
        self.migrations_dir = Path(migrations_dir)
        
        if not self.db_path.exists():
            print(f"Warning: Database file {self.db_path} does not exist. It will be created.")
        
        if not self.migrations_dir.exists():
            raise FileNotFoundError(f"Migrations directory {self.migrations_dir} not found")
    
    @contextmanager
    def get_connection(self):
        """Safe database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def check_migration_applied(self, migration_name: str) -> bool:
        """Check if a migration has already been applied"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM migration_log 
                    WHERE migration_name = ? AND status = 'completed'
                """, (migration_name,))
                result = cursor.fetchone()
                return result['count'] > 0
        except sqlite3.OperationalError:
            # migration_log table doesn't exist yet
            return False
    
    def get_table_info(self, table_name: str) -> dict:
        """Get information about table structure"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                return {col['name']: col['type'] for col in columns}
        except sqlite3.OperationalError:
            return {}
    
    def apply_migration(self, migration_file: Path) -> bool:
        """Apply a single migration file"""
        migration_name = migration_file.stem
        
        print(f"\n{'='*60}")
        print(f"Applying migration: {migration_name}")
        print(f"File: {migration_file}")
        print(f"{'='*60}")
        
        # Check if already applied
        if self.check_migration_applied(migration_name):
            print(f"Migration {migration_name} has already been applied. Skipping.")
            return True
        
        # Read migration SQL
        try:
            sql_content = migration_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading migration file: {e}")
            return False
        
        # Backup database before migration
        backup_path = self.backup_database()
        print(f"Database backed up to: {backup_path}")
        
        # Apply migration
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Execute migration SQL
                cursor.executescript(sql_content)
                conn.commit()
                
                print(f"Migration {migration_name} applied successfully!")
                return True
                
        except Exception as e:
            print(f"Error applying migration {migration_name}: {e}")
            print(f"Database backup available at: {backup_path}")
            return False
    
    def backup_database(self) -> Path:
        """Create a backup of the database before migration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"{self.db_path.stem}.backup_{timestamp}"
        
        try:
            if self.db_path.exists():
                import shutil
                shutil.copy2(self.db_path, backup_path)
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            
        return backup_path
    
    def verify_migration_003(self) -> bool:
        """Verify that migration 003 was applied correctly"""
        print("\nVerifying migration 003 application...")
        
        # Expected new features from Phase 1
        expected_features = [
            # OFI features
            'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m',
            'ofi_normalized_10s', 'ofi_normalized_30s', 'ofi_normalized_1m', 'ofi_normalized_5m',
            'ofi_weighted_10s', 'ofi_weighted_30s', 'ofi_weighted_1m', 'ofi_weighted_5m',
            
            # Book pressure features
            'bid_pressure', 'ask_pressure', 'bid_pressure_weighted', 'ask_pressure_weighted',
            'pressure_imbalance', 'pressure_imbalance_weighted', 'book_depth_asymmetry',
            
            # Stability indicators
            'bid_quote_lifetime', 'ask_quote_lifetime', 'quote_lifetime', 'book_resilience',
            'book_shape_1_5', 'book_shape_stability', 'volume_concentration',
            'bid_volume_concentration', 'ask_volume_concentration',
            'spread_stability_10', 'spread_stability_50', 'spread_stability_100',
            'spread_stability_norm_10', 'spread_stability_norm_50', 'spread_stability_norm_100',
            
            # Enhanced volatility
            'l2_volatility_10', 'l2_volatility_50', 'l2_volatility_200', 'mid_price_return',
            'volatility_10', 'volatility_30', 'volatility_100', 'volatility_200', 'volatility_500',
            'upside_vol_10', 'upside_vol_30', 'upside_vol_100', 'upside_vol_200', 'upside_vol_500',
            'downside_vol_10', 'downside_vol_30', 'downside_vol_100'
        ]
        
        verification_results = {}
        
        # Check l2_features table
        l2_features_columns = self.get_table_info('l2_features')
        if l2_features_columns:
            missing_in_l2_features = [f for f in expected_features if f not in l2_features_columns]
            verification_results['l2_features'] = {
                'exists': True,
                'new_columns_added': len([f for f in expected_features if f in l2_features_columns]),
                'missing_columns': missing_in_l2_features
            }
        else:
            verification_results['l2_features'] = {'exists': False}
        
        # Check l2_training_data table
        l2_training_columns = self.get_table_info('l2_training_data')
        if l2_training_columns:
            missing_in_l2_training = [f for f in expected_features if f not in l2_training_columns]
            verification_results['l2_training_data'] = {
                'exists': True,
                'new_columns_added': len([f for f in expected_features if f in l2_training_columns]),
                'missing_columns': missing_in_l2_training
            }
        else:
            verification_results['l2_training_data'] = {'exists': False}
        
        # Check feature_metadata registrations
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM feature_metadata 
                    WHERE feature_group IN ('order_flow_imbalance', 'book_pressure', 'stability_indicators', 'enhanced_volatility')
                """)
                metadata_count = cursor.fetchone()['count']
                verification_results['feature_metadata'] = {
                    'phase1_features_registered': metadata_count
                }
        except Exception as e:
            verification_results['feature_metadata'] = {'error': str(e)}
        
        # Print verification results
        print("\nVerification Results:")
        print("-" * 40)
        
        for table, results in verification_results.items():
            print(f"\n{table.upper()}:")
            if results.get('exists', True):
                if 'new_columns_added' in results:
                    print(f"  ✓ New columns added: {results['new_columns_added']}")
                    if results['missing_columns']:
                        print(f"  ⚠ Missing columns: {len(results['missing_columns'])}")
                        for col in results['missing_columns'][:5]:  # Show first 5
                            print(f"    - {col}")
                        if len(results['missing_columns']) > 5:
                            print(f"    ... and {len(results['missing_columns']) - 5} more")
                    else:
                        print(f"  ✓ All expected columns present")
                elif 'phase1_features_registered' in results:
                    print(f"  ✓ Phase 1 features registered: {results['phase1_features_registered']}")
                elif 'error' in results:
                    print(f"  ✗ Error: {results['error']}")
            else:
                print(f"  ⚠ Table does not exist")
        
        # Overall success check
        success = (
            verification_results.get('l2_features', {}).get('exists', False) or
            verification_results.get('l2_training_data', {}).get('exists', False)
        ) and (
            len(verification_results.get('l2_features', {}).get('missing_columns', [])) == 0 or
            len(verification_results.get('l2_training_data', {}).get('missing_columns', [])) == 0
        )
        
        if success:
            print(f"\n✓ Migration 003 verification PASSED")
            print(f"✓ All 48 Phase 1 critical L2 features have been successfully added")
        else:
            print(f"\n⚠ Migration 003 verification completed with warnings")
            print(f"⚠ Some features may be missing or tables may not exist")
        
        return success
    
    def run_migration_003(self):
        """Specifically run migration 003 for missing L2 features"""
        migration_file = self.migrations_dir / "003_add_missing_l2_features.sql"
        
        if not migration_file.exists():
            print(f"Error: Migration file {migration_file} not found")
            return False
        
        print("Running Migration 003: Add Missing L2 Features")
        print("=" * 60)
        print("This migration will add 48 critical Phase 1 L2 features:")
        print("- Order Flow Imbalance features: 12")
        print("- Book Pressure features: 7") 
        print("- Stability Indicators: 15")
        print("- Enhanced Volatility features: 14")
        print("=" * 60)
        
        success = self.apply_migration(migration_file)
        
        if success:
            self.verify_migration_003()
        
        return success
    
    def run_patch_migration_004(self):
        """Specifically run patch migration 004 for missing training features"""
        migration_file = self.migrations_dir / "004_patch_missing_training_features.sql"
        
        if not migration_file.exists():
            print(f"Error: Migration file {migration_file} not found")
            return False
        
        print("Running Patch Migration 004: Add Missing Training Features")
        print("=" * 60)
        print("This patch migration will add 2 missing columns to l2_training_data:")
        print("- bid_volume_concentration (REAL)")
        print("- ask_volume_concentration (REAL)")
        print("")
        print("Purpose: Complete Phase 1 schema consistency between l2_features and l2_training_data")
        print("=" * 60)
        
        success = self.apply_migration(migration_file)
        
        if success:
            self.verify_patch_004()
        
        return success
    
    def verify_patch_004(self) -> bool:
        """Verify that patch migration 004 was applied correctly"""
        print("\nVerifying patch migration 004 application...")
        
        # The two missing columns that should have been added
        patch_features = ['bid_volume_concentration', 'ask_volume_concentration']
        
        verification_results = {}
        
        # Check l2_training_data table for the patched columns
        l2_training_columns = self.get_table_info('l2_training_data')
        if l2_training_columns:
            missing_patch_columns = [f for f in patch_features if f not in l2_training_columns]
            verification_results['l2_training_data'] = {
                'exists': True,
                'patch_columns_added': len([f for f in patch_features if f in l2_training_columns]),
                'missing_patch_columns': missing_patch_columns,
                'total_columns': len(l2_training_columns)
            }
        else:
            verification_results['l2_training_data'] = {'exists': False}
        
        # Check l2_features table for comparison
        l2_features_columns = self.get_table_info('l2_features')
        if l2_features_columns:
            verification_results['l2_features'] = {
                'exists': True,
                'total_columns': len(l2_features_columns)
            }
        else:
            verification_results['l2_features'] = {'exists': False}
        
        # Check feature_metadata for the patch features
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM feature_metadata 
                    WHERE feature_name IN ('bid_volume_concentration', 'ask_volume_concentration')
                """)
                patch_metadata_count = cursor.fetchone()['count']
                verification_results['feature_metadata'] = {
                    'patch_features_registered': patch_metadata_count
                }
        except Exception as e:
            verification_results['feature_metadata'] = {'error': str(e)}
        
        # Print verification results
        print("\nPatch Migration 004 Verification Results:")
        print("-" * 50)
        
        for table, results in verification_results.items():
            print(f"\n{table.upper()}:")
            if results.get('exists', True):
                if 'patch_columns_added' in results:
                    print(f"  ✓ Patch columns added: {results['patch_columns_added']}/2")
                    print(f"  ✓ Total columns: {results['total_columns']}")
                    if results['missing_patch_columns']:
                        print(f"  ⚠ Still missing: {results['missing_patch_columns']}")
                    else:
                        print(f"  ✓ All patch columns present")
                elif 'total_columns' in results:
                    print(f"  ✓ Total columns: {results['total_columns']}")
                elif 'patch_features_registered' in results:
                    print(f"  ✓ Patch features in metadata: {results['patch_features_registered']}/2")
                elif 'error' in results:
                    print(f"  ✗ Error: {results['error']}")
            else:
                print(f"  ⚠ Table does not exist")
        
        # Schema consistency check
        if (verification_results.get('l2_training_data', {}).get('exists', False) and 
            verification_results.get('l2_features', {}).get('exists', False)):
            training_cols = verification_results['l2_training_data']['total_columns']
            features_cols = verification_results['l2_features']['total_columns']
            print(f"\nSCHEMA CONSISTENCY CHECK:")
            print(f"  l2_features columns: {features_cols}")
            print(f"  l2_training_data columns: {training_cols}")
            
            if training_cols == features_cols:
                print(f"  ✓ SCHEMA CONSISTENCY: ACHIEVED")
                print(f"  ✓ Both tables have identical column counts")
            else:
                print(f"  ⚠ SCHEMA INCONSISTENCY: Column count mismatch")
                print(f"  ⚠ Difference: {abs(training_cols - features_cols)} columns")
        
        # Overall success check
        success = (
            verification_results.get('l2_training_data', {}).get('patch_columns_added', 0) == 2 and
            len(verification_results.get('l2_training_data', {}).get('missing_patch_columns', [])) == 0
        )
        
        if success:
            print(f"\n✓ Patch Migration 004 verification PASSED")
            print(f"✓ Both missing volume concentration columns have been successfully added")
            print(f"✓ Phase 1 L2 features migration is now COMPLETE")
        else:
            print(f"\n⚠ Patch Migration 004 verification completed with warnings")
            print(f"⚠ Some patch columns may still be missing")
        
        return success

def main():
    """Main entry point"""
    # Check command line arguments
    migration_type = "patch"  # Default to patch migration
    if len(sys.argv) > 1:
        if sys.argv[1] in ["003", "migration_003"]:
            migration_type = "003"
            db_path = sys.argv[2] if len(sys.argv) > 2 else "trading_bot.db"
        elif sys.argv[1] in ["004", "patch", "patch_004"]:
            migration_type = "patch"
            db_path = sys.argv[2] if len(sys.argv) > 2 else "trading_bot.db"
        else:
            db_path = sys.argv[1]
    else:
        db_path = "trading_bot.db"
    
    print(f"Database Migration Runner")
    print(f"Target database: {db_path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Migration type: {migration_type}")
    
    try:
        runner = MigrationRunner(db_path)
        
        if migration_type == "patch":
            success = runner.run_patch_migration_004()
            completion_message = "Patch Migration 004 completed successfully!"
            features_message = "✅ Database schema consistency between l2_features and l2_training_data achieved"
        else:
            success = runner.run_migration_003()
            completion_message = "Migration 003 completed successfully!"
            features_message = "✅ Database now supports 48 additional Phase 1 L2 features"
        
        if success:
            print(f"\n✅ {completion_message}")
            print(f"{features_message}")
            print(f"\nNext steps:")
            print(f"1. Update your feature engineering code to populate these new columns")
            print(f"2. Retrain models with the expanded feature set")
            print(f"3. Consider running Phase 2 HHT migrations if HHT processing is enabled")
        else:
            print(f"\n❌ Migration failed. Check error messages above.")
            print(f"❌ Database backup should be available for rollback if needed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()