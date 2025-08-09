"""
L2 Price Reconstructor stub
"""

class L2PriceReconstructor:
    """Placeholder for L2 price reconstruction"""
    
    def __init__(self, config):
        self.config = config
    
    def reconstruct_price_from_l2(self, l2_data):
        """Return mid price from L2 data"""
        if 'bid_price_1' in l2_data and 'ask_price_1' in l2_data:
            return (l2_data['bid_price_1'] + l2_data['ask_price_1']) / 2
        return None
    
    def reconstruct_price_series(self, l2_data_list):
        """Reconstruct price series from L2 data"""
        prices = []
        timestamps = []
        
        for record in l2_data_list:
            # Extract bid/ask from the record
            if 'b' in record and 'a' in record and len(record['b']) > 0 and len(record['a']) > 0:
                bid = float(record['b'][0][0])
                ask = float(record['a'][0][0])
                mid_price = (bid + ask) / 2
                prices.append(mid_price)
                timestamps.append(record.get('timestamp', None))
        
        return prices, timestamps