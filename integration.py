import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import json

class PredictiveAnalyticsManager:
    def __init__(self, db_path, model_path='models/'):
        self.db_path = db_path
        self.model_path = model_path
        self.conn = sqlite3.connect(db_path)
        
        # Create necessary components
        self.data_pipeline = DataPipeline(self.conn)
        self.backtester = BacktestFramework(self.data_pipeline, PredictiveModel)
        
        # Dictionary to store active models
        self.active_models = {}
        
    def run_full_backtest(self, symbol="KAITO", lookback_days=120):
        """Run comprehensive backtesting"""
        results = self.backtester.run_backtest(
            symbol=symbol,
            lookback_days=lookback_days,
            prediction_horizons=[1, 3, 7, 14],
            model_types=['gbm', 'rf', 'lstm'],
            n_splits=3
        )
        
        # Generate report
        report = self.backtester.generate_report()
        
        # Save report
        report_path = os.path.join(self.model_path, f"backtest_report_{datetime.now().strftime('%Y%m%d')}.md")
        with open(report_path, 'w') as f:
            f.write(report)
            
        return results, report
    
    def train_production_models(self, symbol="KAITO", horizons=[1, 3, 7]):
        """Train models for production use"""
        # Get full dataset
        features = self.data_pipeline.get_feature_set(symbol, lookback_days=120)
        
        if features.empty:
            print("Not enough data for training")
            return False
            
        # Best models based on backtesting (this would come from your backtest results)
        best_models = {
            1: 'gbm',     # 1-day horizon: Gradient Boosting
            3: 'rf',      # 3-day horizon: Random Forest
            7: 'lstm'     # 7-day horizon: LSTM
        }
        
        # Train models for each horizon
        for horizon in horizons:
            model_type = best_models.get(horizon, 'gbm')
            print(f"Training {model_type} model for {horizon}-day horizon")
            
            # Prepare data
            use_sequence = model_type in ['lstm', 'transformer']
            X, y, feature_names = self.data_pipeline.prepare_for_model(
                features.copy(),
                prediction_horizon=horizon,
                sequence_length=14 if use_sequence else 0
            )
            
            # Create and train model
            model = PredictiveModel(model_type=model_type, model_path=self.model_path)
            model.train(X, y, feature_names=feature_names)
            
            # Save model
            model_path = model.save_model(f"{symbol}_{model_type}_h{horizon}_prod")
            
            # Add to active models
            self.active_models[f"{horizon}d"] = {
                'model': model,
                'path': model_path,
                'type': model_type,
                'features': feature_names,
                'horizon': horizon
            }
            
        return True
    
    def load_production_models(self, symbol="KAITO", horizons=[1, 3, 7]):
        """Load previously trained production models"""
        best_models = {
            1: 'gbm',     # 1-day horizon
            3: 'rf',      # 3-day horizon
            7: 'lstm'     # 7-day horizon
        }
        
        for horizon in horizons:
            model_type = best_models.get(horizon, 'gbm')
            filename = f"{symbol}_{model_type}_h{horizon}_prod"
            
            try:
                model = PredictiveModel(model_type=model_type, model_path=self.model_path)
                model.load_model(filename)
                
                self.active_models[f"{horizon}d"] = {
                    'model': model,
                    'path': os.path.join(self.model_path, filename),
                    'type': model_type,
                    'horizon': horizon
                }
                
                print(f"Loaded {model_type} model for {horizon}-day horizon")
            except Exception as e:
                print(f"Error loading model {filename}: {str(e)}")
                
        return len(self.active_models) > 0
    
    def generate_predictions(self, symbol="KAITO"):
        """Generate predictions using active models"""
        if not self.active_models:
            print("No active models. Please train or load models first.")
            return None
            
        # Get latest data
        features = self.data_pipeline.get_feature_set(symbol, lookback_days=30)
        
        if features.empty:
            print("Not enough recent data for predictions")
            return None
            
        predictions = {}
        
        # Generate predictions with each model
        for horizon_key, model_info in self.active_models.items():
            model = model_info['model']
            horizon = model_info['horizon']
            model_type = model_info['type']
            
            # Prepare latest data
            use_sequence = model_type in ['lstm', 'transformer']
            sequence_length = 14 if use_sequence else 0
            
            # We only need the most recent complete data point for prediction
            if sequence_length > 0:
                # For sequence models, we need the last sequence_length data points
                X_latest, _, _ = self.data_pipeline.prepare_for_model(
                    features.tail(sequence_length + 5),  # Get a few extra rows to be safe
                    prediction_horizon=horizon,
                    sequence_length=sequence_length
                )
                
                if len(X_latest) > 0:
                    # Take the most recent sequence
                    X_pred = X_latest[-1:] 
                else:
                    print(f"Not enough data for sequence prediction with {horizon}-day horizon")
                    continue
            else:
                # For non-sequence models, prepare the most recent data point
                X_latest, _, _ = self.data_pipeline.prepare_for_model(
                    features.tail(horizon + 5),  # Get a few extra rows to be safe
                    prediction_horizon=horizon,
                    sequence_length=0
                )
                
                if len(X_latest) > 0:
                    # Take the most recent data point
                    X_pred = X_latest[-1:]
                else:
                    print(f"Not enough data for prediction with {horizon}-day horizon")
                    continue
            
            # Generate prediction
            pred_value = model.predict(X_pred)[0]
            
            # For percentage change predictions
            if isinstance(pred_value, (np.ndarray, list)):
                pred_value = pred_value[0]
                
            # Get most recent price
            current_price = features['price'].iloc[-1]
            
            # Calculate predicted price
            predicted_price = current_price * (1 + pred_value)
            
            # Determine direction
            direction = "up" if pred_value > 0 else "down" if pred_value < 0 else "neutral"
            
            # Add to predictions
            predictions[horizon_key] = {
                'current_price': current_price,
                'predicted_change_pct': pred_value * 100,  # Convert to percentage
                'predicted_price': predicted_price,
                'direction': direction,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d')
            }
        
        # Store predictions in database
        self._store_predictions(symbol, predictions)
            
        return predictions
    
    def _store_predictions(self, symbol, predictions):
        """Store predictions in database for future evaluation"""
        try:
            # Create predictions table if it doesn't exist
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_change_pct REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    direction TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    outcome TEXT DEFAULT NULL
                )
            """)
            
            # Insert predictions
            for horizon, pred in predictions.items():
                cursor.execute("""
                    INSERT INTO model_predictions (
                        timestamp, symbol, horizon, current_price, 
                        predicted_change_pct, predicted_price, direction, target_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    symbol,
                    horizon,
                    pred['current_price'],
                    pred['predicted_change_pct'],
                    pred['predicted_price'],
                    pred['direction'],
                    pred['target_date']
                ))
            
            self.conn.commit()
            print(f"Stored {len(predictions)} predictions in database")
        except Exception as e:
            print(f"Error storing predictions: {str(e)}")
    
    def evaluate_past_predictions(self):
        """Evaluate the accuracy of past predictions"""
        try:
            # Get predictions with expired target dates
            query = """
                SELECT * FROM model_predictions
                WHERE target_date <= date('now')
                AND outcome IS NULL
            """
            predictions_df = pd.read_sql(query, self.conn)
            
            if predictions_df.empty:
                print("No predictions to evaluate")
                return None
                
            # For each prediction, get the actual price on the target date
            for index, row in predictions_df.iterrows():
                symbol = row['symbol']
                target_date = row['target_date']
                
                # Get actual price on target date
                price_query = f"""
                    SELECT price FROM market_data
                    WHERE chain = '{symbol}'
                    AND date(timestamp) = '{target_date}'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                actual_df = pd.read_sql(price_query, self.conn)
                
                if not actual_df.empty:
                    actual_price = actual_df['price'].iloc[0]
                    
                    # Calculate actual change
                    current_price = row['current_price']
                    actual_change_pct = ((actual_price / current_price) - 1) * 100
                    
                    # Determine if prediction was correct
                    predicted_direction = row['direction']
                    actual_direction = "up" if actual_change_pct > 0 else "down" if actual_change_pct < 0 else "neutral"
                    
                    is_correct = predicted_direction == actual_direction
                    
                    # Calculate error
                    predicted_change_pct = row['predicted_change_pct']
                    error_pct = predicted_change_pct - actual_change_pct
                    
                    # Update the prediction with outcome
                    outcome = {
                        'actual_price': float(actual_price),
                        'actual_change_pct': float(actual_change_pct),
                        'actual_direction': actual_direction,
                        'is_correct': bool(is_correct),
                        'error_pct': float(error_pct)
                    }
                    
                    # Store outcome in database
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        UPDATE model_predictions
                        SET outcome = ?
                        WHERE id = ?
                    """, (
                        json.dumps(outcome),
                        row['id']
                    ))
            
            self.conn.commit()
            print(f"Evaluated {len(predictions_df)} past predictions")
            
            # Calculate and return accuracy metrics
            return self.get_prediction_accuracy()
        except Exception as e:
            print(f"Error evaluating predictions: {str(e)}")
            return None
    
    def get_prediction_accuracy(self, days=30):
        """Get accuracy metrics for predictions in the past N days"""
        try:
            query = f"""
                SELECT * FROM model_predictions
                WHERE timestamp >= datetime('now', '-{days} days')
                AND outcome IS NOT NULL
            """
            predictions_df = pd.read_sql(query, self.conn)
            
            if predictions_df.empty:
                return {
                    'total_predictions': 0,
                    'accuracy': 0,
                    'avg_error': 0
                }
            
            # Parse outcome JSON
            predictions_df['outcome_parsed'] = predictions_df['outcome'].apply(json.loads)
            
            # Extract metrics
            is_correct = [outcome.get('is_correct', False) for outcome in predictions_df['outcome_parsed']]
            errors = [abs(outcome.get('error_pct', 0)) for outcome in predictions_df['outcome_parsed']]
            
            # Calculate by horizon
            horizons = predictions_df['horizon'].unique()
            
            metrics = {
                'total_predictions': len(predictions_df),
                'overall_accuracy': sum(is_correct) / len(is_correct) if is_correct else 0,
                'avg_error': sum(errors) / len(errors) if errors else 0,
                'by_horizon': {}
            }
            
            for horizon in horizons:
                horizon_df = predictions_df[predictions_df['horizon'] == horizon]
                
                horizon_correct = [outcome.get('is_correct', False) for outcome in horizon_df['outcome_parsed']]
                horizon_errors = [abs(outcome.get('error_pct', 0)) for outcome in horizon_df['outcome_parsed']]
                
                metrics['by_horizon'][horizon] = {
                    'predictions': len(horizon_df),
                    'accuracy': sum(horizon_correct) / len(horizon_correct) if horizon_correct else 0,
                    'avg_error': sum(horizon_errors) / len(horizon_errors) if horizon_errors else 0
                }
                
            return metrics
        except Exception as e:
            print(f"Error getting prediction accuracy: {str(e)}")
            return None
    
    def get_latest_predictions(self):
        """Get the most recent predictions"""
        try:
            query = """
                SELECT * FROM model_predictions
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
            """
            predictions_df = pd.read_sql(query, self.conn)
            
            if predictions_df.empty:
                return None
                
            # Group by horizon and get latest for each
            latest_predictions = {}
            
            for horizon in predictions_df['horizon'].unique():
                horizon_df = predictions_df[predictions_df['horizon'] == horizon]
                latest = horizon_df.iloc[0]
                
                latest_predictions[horizon] = {
                    'current_price': latest['current_price'],
                    'predicted_change_pct': latest['predicted_change_pct'],
                    'predicted_price': latest['predicted_price'],
                    'direction': latest['direction'],
                    'prediction_date': latest['timestamp'],
                    'target_date': latest['target_date']
                }
                
            return latest_predictions
        except Exception as e:
            print(f"Error getting latest predictions: {str(e)}")
            return None
