# featureengineer.py
# L2-Only Trading Strategy Implementation
# Restructured for L2-only mode with advanced microstructure features

import pandas as pd


class FeatureEngineer:
    """
    Calculates L2 microstructure features for the L2-only trading strategy.
    This version is adapted to work with the 'l2_training_data_practical'
    schema.
    """
    def __init__(self, config: dict):
        """Initializes the FeatureEngineer for L2-only mode."""
        self.config = config
        self.l2_features_config = config.get('l2_features', [])

        print("FeatureEngineer initialized for L2 practical schema.")
        print(
            "Number of L2 features to generate: "
            f"{len(self.l2_features_config)}"
        )

    def generate_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from the L2 data.
        This is the main entry point for feature generation.
        """
        if df_input is None or df_input.empty:
            print("Error: Input L2 DataFrame is empty.")
            return pd.DataFrame()

        print(f"Starting feature generation for {len(df_input)} rows...")
        df = df_input.copy()

        # All feature calculation is now consolidated here
        df = self._calculate_microstructure_features(df)
        df = self._handle_missing_values(df)

        print(f"Feature generation complete. DataFrame shape: {df.shape}")
        return df

    def _calculate_microstructure_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates L2 microstructure features from the practical L2 data
        schema.
        """
        print("Calculating L2 microstructure features...")

        # Basic spread and mid-price
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        
        # Additional spread features (aliases)
        df['bid_ask_spread'] = df['spread']
        df['bid_ask_spread_pct'] = df['spread_bps'] / 10000

        # Weighted prices and microprice
        df['weighted_bid_price'] = df['bid_price_1'] * df['ask_size_1']
        df['weighted_ask_price'] = df['ask_price_1'] * df['bid_size_1']
        
        # Weighted mid price (volume-weighted)
        total_volume = df['bid_size_1'] + df['ask_size_1']
        df['weighted_mid_price'] = (
            df['bid_price_1'] * df['bid_size_1'] + 
            df['ask_price_1'] * df['ask_size_1']
        ) / total_volume
        df['weighted_mid_price'] = df['weighted_mid_price'].fillna(df['mid_price'])

        sum_of_sizes = df['bid_size_1'] + df['ask_size_1']
        df['microprice'] = (
            df['weighted_bid_price'] + df['weighted_ask_price']
        ) / sum_of_sizes
        df['microprice'] = df['microprice'].fillna(df['mid_price'])

        # Order book imbalance
        df['order_book_imbalance'] = (
            df['bid_size_1'] - df['ask_size_1']
        ) / sum_of_sizes
        
        # Order book imbalance at different levels
        for level in [2, 3, 5]:
            bid_sum = df[[f'bid_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            ask_sum = df[[f'ask_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            total_sum = bid_sum + ask_sum
            df[f'order_book_imbalance_{level}'] = (bid_sum - ask_sum) / total_sum
            df[f'order_book_imbalance_{level}'] = df[f'order_book_imbalance_{level}'].fillna(0)

        # Aggregated volumes and slopes
        for i in range(1, 11):
            bid_cols = [f'bid_size_{j}' for j in range(1, i + 1)]
            ask_cols = [f'ask_size_{j}' for j in range(1, i + 1)]
            df[f'total_bid_volume_{i}'] = df[bid_cols].sum(axis=1)
            df[f'total_ask_volume_{i}'] = df[ask_cols].sum(axis=1)

        # Price impact (simplified)
        df['price_impact_ask'] = (
            df['ask_price_1'] - df['microprice']
        ) / df['mid_price']
        df['price_impact_bid'] = (
            df['microprice'] - df['bid_price_1']
        ) / df['mid_price']
        
        # Additional price impact features
        df['price_impact_buy'] = df['price_impact_ask']
        df['price_impact_sell'] = df['price_impact_bid']
        
        # Price impact at different levels
        for level in [1, 5, 10]:
            if level == 1:
                df['price_impact_1'] = (df['ask_price_1'] - df['bid_price_1']) / (2 * df['mid_price'])
            else:
                # Weighted average price to buy/sell at level
                bid_prices = [df[f'bid_price_{i}'] for i in range(1, min(level+1, 11))]
                ask_prices = [df[f'ask_price_{i}'] for i in range(1, min(level+1, 11))]
                df[f'price_impact_{level}'] = (
                    sum(ask_prices) / len(ask_prices) - sum(bid_prices) / len(bid_prices)
                ) / (2 * df['mid_price'])

        # Rolling volatility of mid-price returns
        df['mid_price_return'] = df['mid_price'].pct_change()
        for window in [10, 50, 200]:  # 1-sec, 5-sec, 20-sec at 100ms
            df[f'l2_volatility_{window}'] = df['mid_price_return'].rolling(
                window=window).std()

        print("Microstructure feature calculation completed.")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills or handles NaN values in the feature DataFrame."""
        print("Handling missing values...")
        # Forward-fill most features to propagate last known state
        df.ffill(inplace=True)
        # Backward-fill to handle NaNs at the beginning
        df.bfill(inplace=True)
        # Any remaining NaNs can be filled with 0
        df.fillna(0, inplace=True)
        print("Missing values handled.")
        return df
