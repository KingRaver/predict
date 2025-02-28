import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class PredictiveModel:
    def __init__(self, model_type='gbm', model_path='models/'):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.features = None
        os.makedirs(model_path, exist_ok=True)
        
    def build_model(self, input_shape=None, sequence=False):
        """Build the specified model type"""
        if self.model_type == 'gbm':
            self.model = GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=4, 
                random_state=42,
                validation_fraction=0.2
            )
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'lstm' and sequence:
            # Build LSTM model for sequence data
            self.model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
        elif self.model_type == 'transformer' and sequence:
            # More advanced transformer-based model
            inputs = keras.Input(shape=input_shape)
            x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
            x = layers.Attention()([x, x])
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(1)(x)
            
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            
        return self.model
    
    def train(self, X, y, feature_names=None, validation_split=0.2, epochs=50):
        """Train the model with the provided data"""
        self.features = feature_names
        
        if self.model is None:
            if len(X.shape) > 2:  # Sequence data
                self.build_model(input_shape=(X.shape[1], X.shape[2]), sequence=True)
            else:
                self.build_model()
        
        if isinstance(self.model, (GradientBoostingRegressor, RandomForestRegressor)):
            # For sklearn models
            self.model.fit(X, y)
            
            # Calculate feature importance
            if hasattr(self.model, 'feature_importances_') and feature_names is not None:
                importances = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                print("Feature Importance:")
                print(feature_importance.head(10))
                
            return self.model
        else:
            # For deep learning models
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if isinstance(self.model, (GradientBoostingRegressor, RandomForestRegressor)):
            predictions = self.model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # For price direction accuracy
            direction_accuracy = np.mean((y_test > 0) == (predictions > 0))
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
        else:
            # For deep learning models
            loss = self.model.evaluate(X_test, y_test)
            predictions = self.model.predict(X_test)
            
            # Calculate additional metrics
            mae = mean_absolute_error(y_test, predictions)
            direction_accuracy = np.mean((y_test > 0) == (predictions.flatten() > 0))
            
            return {
                'loss': loss,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
    
    def predict(self, X):
        """Generate predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        return self.model.predict(X)
    
    def save_model(self, filename=None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        if filename is None:
            filename = f"{self.model_type}_model_{datetime.now().strftime('%Y%m%d')}"
            
        full_path = os.path.join(self.model_path, filename)
        
        if isinstance(self.model, (GradientBoostingRegressor, RandomForestRegressor)):
            # Save sklearn model
            joblib.dump(self.model, f"{full_path}.joblib")
        else:
            # Save keras model
            self.model.save(f"{full_path}")
            
        # Save feature names if available
        if self.features is not None:
            with open(f"{full_path}_features.txt", 'w') as f:
                f.write(','.join(self.features))
                
        return full_path
    
    def load_model(self, filename):
        """Load a trained model"""
        full_path = os.path.join(self.model_path, filename)
        
        if self.model_type in ['gbm', 'rf']:
            # Load sklearn model
            self.model = joblib.load(f"{full_path}.joblib")
        else:
            # Load keras model
            self.model = keras.models.load_model(full_path)
            
        # Load feature names if available
        try:
            with open(f"{full_path}_features.txt", 'r') as f:
                self.features = f.read().split(',')
        except:
            pass
            
        return self.model
