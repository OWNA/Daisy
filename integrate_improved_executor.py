"""
Integration Guide for Improved Order Executor
Shows how to integrate the new executor into existing trading systems
"""

import logging
from typing import Dict, Optional
from improved_order_executor import ImprovedOrderExecutor
from smartorderexecutor import SmartOrderExecutor

logger = logging.getLogger(__name__)


class ExecutorAdapter:
    """
    Adapter class to seamlessly integrate improved executor into existing systems
    Provides backward compatibility while enabling new features
    """
    
    def __init__(self, exchange_api, exec_config: dict, use_improved: bool = True):
        """
        Initialize executor adapter
        
        Args:
            exchange_api: CCXT exchange instance
            exec_config: Execution configuration
            use_improved: Use improved executor (True) or legacy (False)
        """
        self.exchange = exchange_api
        self.use_improved = use_improved
        
        if use_improved:
            # Initialize improved executor with enhanced config
            self.improved_config = {
                'min_order_book_depth': exec_config.get('min_order_book_depth', 100),
                'liquidity_impact_threshold': exec_config.get('liquidity_impact_threshold', 0.001),
                'max_single_order_pct': exec_config.get('max_single_order_pct', 0.2),
                'min_order_size_usd': exec_config.get('min_order_size_usd', 10),
                'passive_spread_bps': exec_config.get('passive_spread_bps', 1),
                'aggressive_spread_bps': exec_config.get('aggressive_spread_bps', 5),
                'post_only_retry_limit': exec_config.get('post_only_retry_limit', 3),
                'order_timeout_seconds': exec_config.get('order_timeout_seconds', 30),
                'between_order_delay_ms': exec_config.get('between_order_delay_ms', 100),
                'maker_fee': exec_config.get('maker_fee', -0.00025),
                'taker_fee': exec_config.get('taker_fee', 0.00075)
            }
            self.executor = ImprovedOrderExecutor(exchange_api, self.improved_config)
            logger.info("Using IMPROVED order executor")
        else:
            # Use legacy executor
            self.executor = SmartOrderExecutor(exchange_api, exec_config)
            logger.info("Using LEGACY order executor")
    
    def execute_order(self, symbol: str, side: str, amount: float, 
                     desired_price: float, order_type: str = 'limit',
                     urgency: str = 'medium', **kwargs) -> Optional[Dict]:
        """
        Execute order with backward compatibility
        
        This method maintains the same interface as SmartOrderExecutor
        but uses the improved executor when enabled
        """
        if self.use_improved:
            # Convert to USD amount for improved executor
            amount_usd = amount * desired_price
            
            # Map order_type to urgency if not specified
            if order_type == 'market':
                urgency = 'high'
            
            # Execute with improved system
            result = self.executor.execute_smart_order(
                symbol=symbol,
                side=side,
                amount_usd=amount_usd,
                urgency=urgency,
                signal_strength=kwargs.get('signal_strength')
            )
            
            # Convert result to legacy format for compatibility
            if result['success'] and result['executed_orders']:
                # Return first order for compatibility
                return result['executed_orders'][0]
            else:
                return None
        else:
            # Use legacy executor
            return self.executor.execute_order(
                symbol=symbol,
                side=side,
                amount=amount,
                desired_price=desired_price,
                order_type=order_type
            )
    
    def execute_smart_order(self, symbol: str, side: str, amount_usd: float,
                          urgency: str = 'medium', signal_strength: float = None) -> Dict:
        """
        Execute order using improved system directly
        Only available when use_improved=True
        """
        if not self.use_improved:
            raise NotImplementedError("Smart order execution requires improved executor")
        
        return self.executor.execute_smart_order(
            symbol=symbol,
            side=side,
            amount_usd=amount_usd,
            urgency=urgency,
            signal_strength=signal_strength
        )
    
    def get_execution_metrics(self) -> Dict:
        """Get execution metrics if available"""
        if self.use_improved and hasattr(self.executor, 'metrics'):
            return self.executor.metrics
        else:
            return {}


def update_trading_bot_orchestrator():
    """
    Example of how to update TradingBotOrchestrator to use improved executor
    
    This shows the minimal changes needed in existing code
    """
    print("""
    To update your TradingBotOrchestrator:
    
    1. In tradingbotorchestrator.py, replace the SmartOrderExecutor import:
    
    ```python
    # Old import
    # from smartorderexecutor import SmartOrderExecutor
    
    # New import
    from integrate_improved_executor import ExecutorAdapter
    ```
    
    2. Update the initialization in __init__:
    
    ```python
    # Old initialization
    # self.order_executor = SmartOrderExecutor(self.exchange, exec_config)
    
    # New initialization with toggle
    use_improved_executor = self.config.get('execution', {}).get('use_improved', True)
    self.order_executor = ExecutorAdapter(
        self.exchange, 
        exec_config,
        use_improved=use_improved_executor
    )
    ```
    
    3. Optionally update execute_trade method to use urgency:
    
    ```python
    def execute_trade(self, symbol, signal, predicted_price, confidence):
        # ... existing code ...
        
        # Determine urgency based on signal strength
        if confidence > 0.8:
            urgency = 'high'
        elif confidence > 0.6:
            urgency = 'medium'
        else:
            urgency = 'low'
        
        # Execute with urgency parameter
        order_result = self.order_executor.execute_order(
            symbol=symbol,
            side=signal,
            amount=position_size,
            desired_price=current_price,
            order_type='limit',
            urgency=urgency,  # New parameter
            signal_strength=confidence  # New parameter
        )
    ```
    
    4. Update config.yaml to enable improved executor:
    
    ```yaml
    execution:
      use_improved: true  # Enable improved executor
      slippage_model_pct: 0.0005
      max_order_book_levels: 20
      # New parameters for improved executor
      min_order_book_depth: 100
      liquidity_impact_threshold: 0.001
      max_single_order_pct: 0.2
      min_order_size_usd: 10
      passive_spread_bps: 1
      aggressive_spread_bps: 5
      post_only_retry_limit: 3
      order_timeout_seconds: 30
      between_order_delay_ms: 100
      maker_fee: -0.00025
      taker_fee: 0.00075
    ```
    """)


def create_migration_script():
    """Create a script to help migrate existing systems"""
    migration_code = '''#!/usr/bin/env python3
"""
Migration script to update existing trading system to use improved executor
Run this to automatically update your trading files
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modification"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"Backed up {filepath} to {backup_path}")
    return backup_path

def update_imports(content):
    """Update import statements"""
    # Replace SmartOrderExecutor import
    content = re.sub(
        r'from smartorderexecutor import SmartOrderExecutor',
        'from integrate_improved_executor import ExecutorAdapter',
        content
    )
    return content

def update_initialization(content):
    """Update executor initialization"""
    # Replace executor initialization
    pattern = r'self\.order_executor\s*=\s*SmartOrderExecutor\((.*?)\)'
    replacement = r'self.order_executor = ExecutorAdapter(\\1, use_improved=True)'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    return content

def add_config_parameters(config_content):
    """Add new parameters to config.yaml"""
    if 'use_improved:' not in config_content:
        # Find execution section and add parameters
        execution_section = """execution:
  use_improved: true  # Enable improved executor
  slippage_model_pct: 0.0005
  max_order_book_levels: 20
  # New parameters for improved executor
  min_order_book_depth: 100
  liquidity_impact_threshold: 0.001
  max_single_order_pct: 0.2
  min_order_size_usd: 10
  passive_spread_bps: 1
  aggressive_spread_bps: 5
  post_only_retry_limit: 3
  order_timeout_seconds: 30
  between_order_delay_ms: 100
  maker_fee: -0.00025
  taker_fee: 0.00075"""
        
        # Replace existing execution section or add new one
        if 'execution:' in config_content:
            config_content = re.sub(
                r'execution:.*?(?=\\n[^\\s]|$)',
                execution_section,
                config_content,
                flags=re.DOTALL
            )
        else:
            config_content += '\\n\\n' + execution_section
    
    return config_content

def migrate_file(filepath):
    """Migrate a single file"""
    print(f"\\nMigrating {filepath}...")
    
    # Backup first
    backup_file(filepath)
    
    # Read content
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Apply updates
    original_content = content
    
    if filepath.endswith('.py'):
        content = update_imports(content)
        content = update_initialization(content)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        content = add_config_parameters(content)
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Updated {filepath}")
    else:
        print(f"- No changes needed for {filepath}")

def main():
    """Run migration"""
    print("Improved Executor Migration Script")
    print("==================================")
    
    # Files to migrate
    files_to_check = [
        'tradingbotorchestrator.py',
        'unified_trading_system.py',
        'config.yaml',
        'config_l2_only.yaml'
    ]
    
    # Check which files exist
    files_to_migrate = []
    for filename in files_to_check:
        if os.path.exists(filename):
            files_to_migrate.append(filename)
        else:
            print(f"- {filename} not found, skipping")
    
    if not files_to_migrate:
        print("\\nNo files found to migrate!")
        return
    
    print(f"\\nFound {len(files_to_migrate)} files to migrate:")
    for f in files_to_migrate:
        print(f"  - {f}")
    
    response = input("\\nProceed with migration? (yes/no): ").lower().strip()
    if response != 'yes':
        print("Migration cancelled.")
        return
    
    # Perform migration
    for filepath in files_to_migrate:
        try:
            migrate_file(filepath)
        except Exception as e:
            print(f"✗ Error migrating {filepath}: {e}")
    
    print("\\nMigration complete!")
    print("\\nNext steps:")
    print("1. Review the changes in the updated files")
    print("2. Run validate_improved_executor.py to test the setup")
    print("3. Start with use_improved: false in config to test compatibility")
    print("4. Switch to use_improved: true when ready")

if __name__ == "__main__":
    main()
'''
    
    with open('migrate_to_improved_executor.py', 'w') as f:
        f.write(migration_code)
    
    print("Created migration script: migrate_to_improved_executor.py")


def main():
    """Show integration examples"""
    print("\n=== Improved Order Executor Integration Guide ===\n")
    
    print("This guide shows how to integrate the improved executor into your existing system.\n")
    
    print("Option 1: Manual Integration")
    print("-" * 40)
    update_trading_bot_orchestrator()
    
    print("\n\nOption 2: Automated Migration")
    print("-" * 40)
    print("Creating migration script...")
    create_migration_script()
    
    print("\n\nOption 3: Gradual Rollout")
    print("-" * 40)
    print("""
    For a gradual rollout:
    
    1. Start with use_improved: false (uses legacy executor)
    2. Run validate_improved_executor.py on testnet
    3. Run test_execution_comparison.py to compare performance
    4. Enable for small positions first:
       - Set use_improved: true
       - Set conservative parameters (low max_single_order_pct)
    5. Monitor execution metrics
    6. Gradually increase position sizes and adjust parameters
    
    The ExecutorAdapter class ensures backward compatibility,
    so your existing code will continue to work without changes.
    """)
    
    print("\n\nSafety Checklist")
    print("-" * 40)
    print("""
    Before going live:
    
    [ ] Run validate_improved_executor.py --testnet
    [ ] Run test_execution_comparison.py with paper trading
    [ ] Review execution metrics and compare with legacy system
    [ ] Set conservative parameters initially
    [ ] Enable for one symbol first
    [ ] Monitor logs closely for first 24 hours
    [ ] Have emergency stop procedure ready
    [ ] Keep backup of original files
    """)


if __name__ == "__main__":
    main()