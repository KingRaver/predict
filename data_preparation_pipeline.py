from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class DataPipeline:
    def __init__(self, db_connection):
        self.conn = db_connection
        self.scaler = MinMaxScaler()
        
    def get_feature_set(self, symbol="KAITO", lookback_days=60):
        """Extract and prepare features for model training"""
        
        # Get market data
        market_query = f"""
        SELECT timestamp, price, volume, price_change_24h 
        FROM market_data 
        WHERE chain = '{symbol}'
        AND timestamp >= datetime('now', '-{lookback_days} days')
        ORDER BY timestamp ASC
        """
        market_df = pd.read_sql(market_query, self.conn)
        
        # Get smart money indicators
        sm_query = f"""
        SELECT timestamp, volume_z_score, price_volume_divergence, 
               stealth_accumulation, abnormal_volume, volume_vs_hourly_avg 
        FROM smart_money_indicators
        WHERE chain = '{symbol}'
        AND timestamp >= datetime('now', '-{lookback_days} days')
        ORDER BY timestamp ASC
        """
        sm_df = pd.read_sql(sm_query, self.conn)
        
        # Get Layer 1 comparison data
        l1_query = f"""
        SELECT timestamp, vs_layer1_avg_change, vs_layer1_volume_growth, outperforming_layer1s
        FROM kaito_layer1_comparison
        WHERE timestamp >= datetime('now', '-{lookback_days} days')
        ORDER BY timestamp ASC
        """
        l1_df = pd.read_sql(l1_query, self.conn)
        
        # Merge dataframes on nearest timestamp
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
        sm_df['timestamp'] = pd.to_datetime(sm_df['timestamp'])
        l1_df['timestamp'] = pd.to_datetime(l1_df['timestamp'])
        
        # Create features dataframe
        features = self._engineer_features(market_df, sm_df, l1_df)
        
        return features
    
    def _engineer_features(self, market_df, sm_df, l1_df):
        """Create advanced features from raw data"""
        
        # Ensure we have data
        if market_df.empty:
            return pd.DataFrame()
            
        # Create price features
        features = market_df.copy()
        
        # Add technical indicators
        features['price_sma_7'] = features['price'].rolling(7).mean()
        features['price_sma_21'] = features['price'].rolling(21).mean()
        features['price_ema_14'] = features['price'].ewm(span=14).mean()
        
        # Calculate volatility
        features['volatility'] = features['price'].rolling(14).std() / features['price']
        
        # Calculate RSI
        delta = features['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Merge smart money features
        if not sm_df.empty:
            features = pd.merge_asof(features, sm_df, on='timestamp', direction='nearest')
            
            # Convert boolean columns to integers
            bool_cols = ['price_volume_divergence', 'stealth_accumulation', 'abnormal_volume']
            for col in bool_cols:
                if col in features.columns:
                    features[col] = features[col].astype(float)
        
        # Merge Layer 1 comparison features
        if not l1_df.empty:
            features = pd.merge_asof(features, l1_df, on='timestamp', direction='nearest')
            
            # Convert boolean columns to integers
            if 'outperforming_layer1s' in features.columns:
                features['outperforming_layer1s'] = features['outperforming_layer1s'].astype(float)
        
        # Forward fill any missing values from merges
        features = features.ffill()
        
        # Drop rows with NaN that couldn't be filled (like initial rolling window)
        features = features.dropna()
        
        return features
    
    def prepare_for_model(self, features, target_col='price', prediction_horizon=24, sequence_length=14):
        """Prepare data for ML model with a specific prediction horizon (hours)"""
        
        # Create target: future price change percentage
        features['target'] = features[target_col].shift(-prediction_horizon) / features[target_col] - 1
        
        # Drop last N rows where we don't have targets
        features = features.iloc[:-prediction_horizon]
        
        # Drop the timestamp column for modeling
        X = features.drop(['timestamp', 'target'], axis=1)
        y = features['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences for time series models if needed
        if sequence_length > 0:
            X_sequences, y_sequences = [], []
            for i in range(len(X_scaled) - sequence_length):
                X_sequences.append(X_scaled[i:i+sequence_length])
                y_sequences.append(y.iloc[i+sequence_length])
                
            return np.array(X_sequences), np.array(y_sequences), X.columns
        
        return X_scaled, y.values, X.columns
