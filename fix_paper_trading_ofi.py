"""
Fix for paper trading to maintain historical L2 data for OFI calculations
"""

import pandas as pd
from collections import deque
from datetime import datetime

class L2DataBuffer:
    """Maintains a rolling window of L2 data for feature calculations"""
    
    def __init__(self, max_window_seconds=300):  # 5 minutes default
        self.max_window = max_window_seconds
        self.buffer = deque()
        
    def add_snapshot(self, l2_snapshot):
        """Add new L2 snapshot with timestamp"""
        l2_snapshot['timestamp'] = datetime.now()
        self.buffer.append(l2_snapshot)
        
        # Remove old data beyond window
        cutoff_time = datetime.now().timestamp() - self.max_window
        while self.buffer and self.buffer[0]['timestamp'].timestamp() < cutoff_time:
            self.buffer.popleft()
            
    def get_dataframe(self):
        """Get all buffered data as DataFrame"""
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer))
        
    def size(self):
        """Get number of snapshots in buffer"""
        return len(self.buffer)


def patch_paper_trading(main_module):
    """Monkey patch the paper trading to use buffered data"""
    
    # Store original trade method
    original_trade = main_module.UnifiedTradingSystem.trade
    
    def patched_trade(self, paper=True, duration=None):
        """Enhanced trade method with L2 buffer"""
        
        # Initialize buffer
        l2_buffer = L2DataBuffer(max_window_seconds=300)
        
        # Patch the inner trading loop
        logger = main_module.logger
        dh = main_module.DataHandler(self.config)
        fe = main_module.FeatureEngineer(self.config)
        mp = main_module.ModelPredictor(self.config)
        rm = main_module.AdvancedRiskManager(self.config)
        ex = main_module.SmartOrderExecutor(self.exchange, self.config)
        
        logger.info("Trading with L2 buffer for OFI calculations")
        trades = []
        self._running = True
        
        while self._running and (duration is None or len(trades) < duration * 60):
            try:
                # Get L2 data
                ob = self.exchange.fetch_l2_order_book(
                    self.config['symbol'], 
                    limit=self.config.get('l2_websocket_depth', 50)
                )
                l2 = dh._process_l2_snapshot(ob)
                
                # Add to buffer
                l2_buffer.add_snapshot(l2)
                
                # Get buffered data for feature calculation
                buffered_df = l2_buffer.get_dataframe()
                
                if len(buffered_df) > 0:
                    # Generate features with historical context
                    feat = fe.generate_features(buffered_df)
                    
                    # Only use latest row for prediction
                    latest_feat = feat.iloc[[-1]]
                    
                    # Predict
                    sig = mp.predict_signals(latest_feat)
                    if sig is None or sig.empty: 
                        continue
                        
                    # Continue with normal trading logic...
                    price = ob['bids'][0][0] if sig.iloc[0]['signal'] > 0 else ob['asks'][0][0]
                    pos = rm.calculate_position_size(
                        self.balance, price, sig.iloc[0]['signal'], sig.iloc[0]['signal_strength']
                    )
                    
                    if pos > 0:
                        side = 'buy' if sig.iloc[0]['signal'] > 0 else 'sell'
                        trade = {'timestamp': datetime.now(), 'side': side, 
                                'size': pos, 'price': price}
                        trades.append(trade)
                        logger.info(
                            f"Trade: {side} {pos:.4f} @ ${price:,.2f} | "
                            f"Buffer size: {l2_buffer.size()} snapshots"
                        )
                        self.balance += pos * price * (1 if side == 'sell' else -1)
                        
                import time
                time.sleep(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Trade error: {e}")
                
        return trades
        
    # Apply patch
    main_module.UnifiedTradingSystem.trade = patched_trade
    print("Paper trading patched to use L2 buffer for OFI calculations")
    

if __name__ == "__main__":
    print("To apply this fix:")
    print("1. Import this module in your main.py")
    print("2. Call patch_paper_trading(sys.modules[__name__])")
    print("\nOr manually implement the L2DataBuffer in your trading loop.")