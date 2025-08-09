#!/usr/bin/env python3
"""
component_factory.py - Component Factory for BTC Trading System

This module provides a centralized factory for creating and managing all core
trading system components. It implements dependency injection to decouple
modules and improve testability.

Sprint 2 - Priority 0.1: Implement Component Factory
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager

# Type hints for component interfaces
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Configuration container for component initialization."""
    component_type: str
    config: Dict[str, Any]
    dependencies: Dict[str, str] = None
    singleton: bool = True
    

@runtime_checkable
class ComponentInterface(Protocol):
    """Protocol for all trading system components."""
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the component with given configuration."""
        ...


class ComponentFactory:
    """
    Central factory for creating and managing trading system components.
    
    Features:
    - Dependency injection for loose coupling
    - Singleton pattern for shared components
    - Configuration management
    - Lazy loading and caching
    - Error handling and graceful degradation
    """
    
    def __init__(self, base_config: Dict[str, Any] = None):
        """Initialize the component factory."""
        self.base_config = base_config or {}
        self._components = {}  # Singleton instances
        self._component_configs = {}  # Component configurations
        self._initialization_order = []  # Order for dependency resolution
        
        # Register default component configurations
        self._register_default_components()
        
        logger.info("ComponentFactory initialized")
    
    def _register_default_components(self):
        """Register default component configurations."""
        
        # Database configuration
        db_config = ComponentConfig(
            component_type='database',
            config={
                'db_path': self.base_config.get('database_path', './trading_bot_live.db'),
                'timeout': 30,
                'check_same_thread': False
            },
            singleton=True
        )
        
        # Feature Registry configuration
        feature_registry_config = ComponentConfig(
            component_type='feature_registry',
            config={
                'enable_caching': True,
                'performance_monitoring': True
            },
            singleton=True
        )
        
        # Feature Engineer configuration
        feature_engineer_config = ComponentConfig(
            component_type='feature_engineer',
            config={
                'symbol': 'BTCUSDT',
                'db_path': self.base_config.get('database_path', './trading_bot_live.db'),
                'feature_set': 'enhanced',
                'cache_enabled': True
            },
            dependencies={'database': 'database', 'feature_registry': 'feature_registry'},
            singleton=True
        )
        
        # Model Trainer configuration
        model_trainer_config = ComponentConfig(
            component_type='model_trainer',
            config={
                'symbol': 'BTCUSDT',
                'db_path': self.base_config.get('database_path', './trading_bot_live.db'),
                'base_dir': self.base_config.get('base_dir', './trading_bot_data'),
                'training_enabled': True
            },
            dependencies={'database': 'database', 'feature_engineer': 'feature_engineer'},
            singleton=True
        )
        
        # Model Predictor configuration
        model_predictor_config = ComponentConfig(
            component_type='model_predictor',
            config={
                'symbol': 'BTCUSDT',
                'base_dir': self.base_config.get('base_dir', './trading_bot_data'),
                'fallback_to_mock': True,
                'confidence_threshold': 0.6
            },
            dependencies={'feature_engineer': 'feature_engineer'},
            singleton=True
        )
        
        # Exchange configuration
        exchange_config = ComponentConfig(
            component_type='exchange',
            config={
                'exchange_name': 'bybit',
                'sandbox': False,  # Use Demo Trading (mainnet) instead of testnet
                'api_key': os.getenv('BYBIT_API_KEY_MAIN'),
                'api_secret': os.getenv('BYBIT_API_SECRET_MAIN'),
                'enable_rate_limit': True,
                'adjust_for_time_difference': True,
                'recv_window': 10000
            },
            singleton=True
        )
        
        # Order Executor configuration
        order_executor_config = ComponentConfig(
            component_type='order_executor',
            config={
                'slippage_model_pct': 0.0005,
                'max_order_book_levels': 20,
                'execution_strategy': 'PASSIVE'
            },
            dependencies={'exchange': 'exchange'},
            singleton=True
        )
        
        # Register all configurations
        self._component_configs.update({
            'database': db_config,
            'feature_registry': feature_registry_config,
            'feature_engineer': feature_engineer_config,
            'model_trainer': model_trainer_config,
            'model_predictor': model_predictor_config,
            'exchange': exchange_config,
            'order_executor': order_executor_config
        })
        
        # Set initialization order based on dependencies
        self._initialization_order = [
            'database',
            'feature_registry',
            'exchange', 
            'feature_engineer',
            'model_trainer',
            'model_predictor',
            'order_executor'
        ]
    
    def register_component(self, name: str, config: ComponentConfig):
        """Register a new component configuration."""
        self._component_configs[name] = config
        logger.info(f"Registered component configuration: {name}")
    
    def get_component(self, name: str, force_recreate: bool = False) -> Optional[Any]:
        """
        Get a component instance, creating it if necessary.
        
        Args:
            name: Component name
            force_recreate: Force recreation even if singleton exists
            
        Returns:
            Component instance or None if creation failed
        """
        try:
            # Check if singleton exists and force_recreate is False
            if name in self._components and not force_recreate:
                return self._components[name]
            
            # Get component configuration
            if name not in self._component_configs:
                logger.error(f"Component '{name}' not registered")
                return None
            
            config = self._component_configs[name]
            
            # Resolve dependencies first
            dependencies = {}
            if config.dependencies:
                for dep_name, dep_key in config.dependencies.items():
                    dep_component = self.get_component(dep_key)
                    if dep_component is None:
                        logger.error(f"Failed to resolve dependency '{dep_key}' for component '{name}'")
                        return None
                    dependencies[dep_name] = dep_component
            
            # Create component instance
            component = self._create_component(config.component_type, config.config, dependencies)
            
            if component is None:
                logger.error(f"Failed to create component '{name}'")
                return None
            
            # Store singleton if configured
            if config.singleton:
                self._components[name] = component
            
            logger.info(f"Created component: {name}")
            return component
            
        except Exception as e:
            logger.error(f"Error creating component '{name}': {e}")
            return None
    
    def _create_component(self, component_type: str, config: Dict[str, Any], dependencies: Dict[str, Any]) -> Optional[Any]:
        """Create a specific component type."""
        
        try:
            if component_type == 'database':
                return self._create_database_component(config)
            
            elif component_type == 'feature_registry':
                return self._create_feature_registry_component(config)
            
            elif component_type == 'feature_engineer':
                return self._create_feature_engineer_component(config, dependencies)
            
            elif component_type == 'model_trainer':
                return self._create_model_trainer_component(config, dependencies)
            
            elif component_type == 'model_predictor':
                return self._create_model_predictor_component(config, dependencies)
            
            elif component_type == 'exchange':
                return self._create_exchange_component(config)
            
            elif component_type == 'order_executor':
                return self._create_order_executor_component(config, dependencies)
            
            else:
                logger.error(f"Unknown component type: {component_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating {component_type} component: {e}")
            return None
    
    def _create_database_component(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create database connection component."""
        import sqlite3
        
        try:
            db_path = config['db_path']
            timeout = config.get('timeout', 30)
            check_same_thread = config.get('check_same_thread', False)
            
            # Test connection
            conn = sqlite3.connect(
                db_path, 
                timeout=timeout,
                check_same_thread=check_same_thread
            )
            conn.close()
            
            # Return connection factory instead of single connection
            return lambda: sqlite3.connect(
                db_path,
                timeout=timeout,
                check_same_thread=check_same_thread
            )
            
        except Exception as e:
            logger.error(f"Database component creation failed: {e}")
            return None
    
    def _create_feature_registry_component(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create feature registry component."""
        try:
            from feature_registry import FeatureRegistry
            
            registry = FeatureRegistry()
            
            # Configure caching if specified
            if not config.get('enable_caching', True):
                registry.clear_cache()
            
            return registry
            
        except ImportError as e:
            logger.error(f"Failed to import FeatureRegistry: {e}")
            return None
    
    def _create_feature_engineer_component(self, config: Dict[str, Any], dependencies: Dict[str, Any]) -> Optional[Any]:
        """Create feature engineer component."""
        try:
            from featureengineer_enhanced import EnhancedFeatureEngineer
            
            fe_config = {
                'symbol': config['symbol'],
                'db_path': config['db_path']
            }
            
            return EnhancedFeatureEngineer(fe_config, config['db_path'])
            
        except ImportError as e:
            logger.error(f"Failed to import EnhancedFeatureEngineer: {e}")
            return None
    
    def _create_model_trainer_component(self, config: Dict[str, Any], dependencies: Dict[str, Any]) -> Optional[Any]:
        """Create model trainer component."""
        try:
            from modeltrainer_enhanced import EnhancedModelTrainer
            
            trainer_config = {
                'symbol': config['symbol'],
                'db_path': config['db_path']
            }
            
            return EnhancedModelTrainer(trainer_config)
            
        except ImportError as e:
            logger.error(f"Failed to import EnhancedModelTrainer: {e}")
            return None
    
    def _create_model_predictor_component(self, config: Dict[str, Any], dependencies: Dict[str, Any]) -> Optional[Any]:
        """Create model predictor component."""
        try:
            from modelpredictor_enhanced import EnhancedModelPredictor
            
            predictor_config = {
                'symbol': config['symbol'],
                'base_dir': config['base_dir']
            }
            
            predictor = EnhancedModelPredictor(predictor_config)
            
            # Try to load models
            if config.get('fallback_to_mock', True):
                try:
                    predictor.load_models()
                except Exception as e:
                    logger.warning(f"Model loading failed, will use mock predictions: {e}")
            
            return predictor
            
        except ImportError as e:
            logger.error(f"Failed to import EnhancedModelPredictor: {e}")
            return None
    
    def _create_exchange_component(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create exchange component."""
        try:
            import ccxt
            
            exchange_class = getattr(ccxt, config['exchange_name'])
            
            exchange_config = {
                'apiKey': config['api_key'],
                'secret': config['api_secret'],
                'sandbox': config['sandbox'],
                'enableRateLimit': config['enable_rate_limit'],
                'adjustForTimeDifference': config['adjust_for_time_difference'],
                'options': {
                    'defaultType': 'linear',
                    'recvWindow': config['recv_window'],
                    'adjustForTimeDifference': config['adjust_for_time_difference'],
                    'timeDifference': 0
                }
            }
            
            return exchange_class(exchange_config)
            
        except Exception as e:
            logger.error(f"Exchange component creation failed: {e}")
            return None
    
    def _create_order_executor_component(self, config: Dict[str, Any], dependencies: Dict[str, Any]) -> Optional[Any]:
        """Create order executor component."""
        try:
            from smartorderexecutor import SmartOrderExecutor
            
            exchange = dependencies.get('exchange')
            if exchange is None:
                logger.error("Order executor requires exchange dependency")
                return None
            
            executor_config = {
                'slippage_model_pct': config['slippage_model_pct'],
                'max_order_book_levels': config['max_order_book_levels']
            }
            
            return SmartOrderExecutor(exchange, executor_config)
            
        except ImportError as e:
            logger.error(f"Failed to import SmartOrderExecutor: {e}")
            return None
    
    def initialize_all_components(self) -> Dict[str, bool]:
        """Initialize all components in dependency order."""
        results = {}
        
        for component_name in self._initialization_order:
            try:
                component = self.get_component(component_name)
                results[component_name] = component is not None
                
                if component is None:
                    logger.error(f"Failed to initialize component: {component_name}")
                else:
                    logger.info(f"Successfully initialized component: {component_name}")
                    
            except Exception as e:
                logger.error(f"Error initializing component '{component_name}': {e}")
                results[component_name] = False
        
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"Component initialization complete: {success_count}/{total_count} successful")
        
        return results
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all initialized components."""
        components = {}
        
        for name in self._initialization_order:
            component = self.get_component(name)
            if component is not None:
                components[name] = component
        
        return components
    
    def shutdown_all_components(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down all components...")
        
        # Shutdown in reverse order
        for component_name in reversed(self._initialization_order):
            if component_name in self._components:
                try:
                    component = self._components[component_name]
                    
                    # Call shutdown method if available
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                    elif hasattr(component, 'close'):
                        component.close()
                    
                    logger.info(f"Shutdown component: {component_name}")
                    
                except Exception as e:
                    logger.error(f"Error shutting down component '{component_name}': {e}")
        
        # Clear all components
        self._components.clear()
        logger.info("All components shutdown complete")
    
    @contextmanager
    def component_context(self):
        """Context manager for automatic component lifecycle management."""
        try:
            # Initialize all components
            results = self.initialize_all_components()
            
            # Check if critical components initialized successfully
            critical_components = ['database', 'feature_engineer']
            critical_success = all(results.get(comp, False) for comp in critical_components)
            
            if not critical_success:
                raise RuntimeError("Critical components failed to initialize")
            
            yield self.get_all_components()
            
        finally:
            # Always shutdown components
            self.shutdown_all_components()


def create_factory_from_config(config_file: str = 'config.yaml') -> ComponentFactory:
    """Create a component factory from configuration file."""
    try:
        import yaml
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return ComponentFactory(config)
        
    except Exception as e:
        logger.error(f"Failed to create factory from config: {e}")
        # Return factory with default configuration
        return ComponentFactory()


# Factory instance for global access (optional pattern)
_default_factory = None

def get_default_factory() -> ComponentFactory:
    """Get the default component factory instance."""
    global _default_factory
    
    if _default_factory is None:
        _default_factory = create_factory_from_config()
    
    return _default_factory


def reset_default_factory():
    """Reset the default factory (useful for testing)."""
    global _default_factory
    
    if _default_factory is not None:
        _default_factory.shutdown_all_components()
        _default_factory = None