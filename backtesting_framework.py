import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta

class BacktestFramework:
    def __init__(self, data_pipeline, model_factory):
        self.data_pipeline = data_pipeline
        self.model_factory = model_factory
        self.results = {}
        
    def run_backtest(self, symbol="KAITO", lookback_days=120, prediction_horizons=[1, 3, 7], 
                    model_types=['gbm', 'lstm'], window_size=60, n_splits=3, sequence_length=14):
        """Run a comprehensive backtest across different models and time horizons"""
        
        # Get full dataset
        features = self.data_pipeline.get_feature_set(symbol, lookback_days)
        
        if features.empty:
            print("Not enough data for backtesting")
            return None
            
        # Create time-based splits for backtesting
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=window_size)
        
        results = []
        
        # Loop through different prediction horizons
        for horizon in prediction_horizons:
            # For each model type
            for model_type in model_types:
                print(f"Testing {model_type} model for {horizon}-day prediction horizon")
                
                use_sequence = model_type in ['lstm', 'transformer']
                
                # Prepare data for this horizon
                X, y, feature_names = self.data_pipeline.prepare_for_model(
                    features.copy(), 
                    prediction_horizon=horizon,
                    sequence_length=sequence_length if use_sequence else 0
                )
                
                # For each time split
                split_count = 0
                split_metrics = []
                
                for train_idx, test_idx in tscv.split(X):
                    split_count += 1
                    print(f"  Split {split_count}/{n_splits}")
                    
                    # Get train/test split
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Create and train model
                    model = self.model_factory(model_type=model_type)
                    model.train(X_train, y_train, feature_names=feature_names)
                    
                    # Evaluate
                    metrics = model.evaluate(X_test, y_test)
                    metrics['split'] = split_count
                    metrics['horizon'] = horizon
                    metrics['model_type'] = model_type
                    
                    split_metrics.append(metrics)
                    
                    # Generate signals and simulate trading
                    predictions = model.predict(X_test)
                    
                    if use_sequence:
                        y_test_flat = y_test
                        predictions_flat = predictions.flatten()
                    else:
                        y_test_flat = y_test
                        predictions_flat = predictions
                    
                    # Create trade signals
                    signals = np.sign(predictions_flat)
                    
                    # Calculate cumulative return
                    # Simplified calculation, in reality you'd account for transaction costs
                    cumulative_return = np.sum(signals * y_test_flat)
                    
                    # Compare to buy and hold
                    buy_and_hold = np.sum(y_test_flat)
                    
                    # Add to metrics
                    metrics['cumulative_return'] = cumulative_return
                    metrics['buy_and_hold'] = buy_and_hold
                    metrics['excess_return'] = cumulative_return - buy_and_hold
                    
                    # Add to results
                    results.append(metrics)
                
                # Calculate average metrics across splits
                avg_metrics = {
                    'model_type': model_type,
                    'horizon': horizon,
                    'direction_accuracy': np.mean([m['direction_accuracy'] for m in split_metrics]),
                    'mean_return': np.mean([m['cumulative_return'] for m in split_metrics]),
                    'excess_return': np.mean([m['excess_return'] for m in split_metrics])
                }
                
                # Add other metrics if present
                if 'rmse' in split_metrics[0]:
                    avg_metrics['rmse'] = np.mean([m['rmse'] for m in split_metrics])
                
                if 'mae' in split_metrics[0]:
                    avg_metrics['mae'] = np.mean([m['mae'] for m in split_metrics])
                
                results.append({**avg_metrics, 'is_average': True})
                
                # Save best model
                best_split = np.argmax([m['direction_accuracy'] for m in split_metrics])
                model.save_model(f"{symbol}_{model_type}_h{horizon}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def plot_results(self):
        """Plot backtest results"""
        if self.results.empty:
            print("No results to plot")
            return
            
        # Only use average results
        avg_results = self.results[self.results.get('is_average', False)]
        
        # Plot direction accuracy by model and horizon
        plt.figure(figsize=(12, 6))
        
        horizons = sorted(avg_results['horizon'].unique())
        model_types = sorted(avg_results['model_type'].unique())
        
        bar_width = 0.8 / len(model_types)
        positions = np.arange(len(horizons))
        
        for i, model in enumerate(model_types):
            model_data = avg_results[avg_results['model_type'] == model]
            accuracies = [model_data[model_data['horizon'] == h]['direction_accuracy'].values[0] 
                          for h in horizons]
            
            plt.bar(positions + i*bar_width, accuracies, bar_width, 
                    label=model.upper(), alpha=0.7)
        
        plt.xlabel('Prediction Horizon (Days)')
        plt.ylabel('Direction Accuracy')
        plt.title('Model Performance by Prediction Horizon')
        plt.xticks(positions + bar_width/2, horizons)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot excess returns
        plt.figure(figsize=(12, 6))
        
        for i, model in enumerate(model_types):
            model_data = avg_results[avg_results['model_type'] == model]
            returns = [model_data[model_data['horizon'] == h]['excess_return'].values[0] 
                      for h in horizons]
            
            plt.bar(positions + i*bar_width, returns, bar_width, 
                    label=model.upper(), alpha=0.7)
        
        plt.xlabel('Prediction Horizon (Days)')
        plt.ylabel('Excess Return vs Buy & Hold')
        plt.title('Model Excess Return by Prediction Horizon')
        plt.xticks(positions + bar_width/2, horizons)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self):
        """Generate a comprehensive backtest report"""
        if self.results.empty:
            return "No backtest results available"
            
        # Filter for average results
        avg_results = self.results[self.results.get('is_average', False)].copy()
        
        # Get best model overall
        best_idx = avg_results['direction_accuracy'].idxmax()
        best_model = avg_results.loc[best_idx]
        
        report = "# Backtest Performance Report\n\n"
        report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        report += "## Best Model Performance\n\n"
        report += f"- Model: {best_model['model_type'].upper()}\n"
        report += f"- Prediction Horizon: {best_model['horizon']} days\n"
        report += f"- Direction Accuracy: {best_model['direction_accuracy']*100:.2f}%\n"
        report += f"- Excess Return vs Buy & Hold: {best_model['excess_return']*100:.2f}%\n"
        
        if 'rmse' in best_model:
            report += f"- RMSE: {best_model['rmse']:.4f}\n"
        
        if 'mae' in best_model:
            report += f"- MAE: {best_model['mae']:.4f}\n"
        
        report += "\n## All Models Summary\n\n"
        
        # Create summary table
        summary = avg_results.pivot_table(
            values='direction_accuracy', 
            index='model_type', 
            columns='horizon', 
            aggfunc='mean'
        )
        
        report += summary.to_markdown()
        report += "\n\n"
        
        report += "## Excess Returns Summary\n\n"
        
        # Create returns table
        returns = avg_results.pivot_table(
            values='excess_return', 
            index='model_type', 
            columns='horizon', 
            aggfunc='mean'
        )
        
        report += returns.to_markdown()
        report += "\n\n"
        
        report += "## Recommendations\n\n"
        
        # Generate recommendations based on results
        best_horizon = best_model['horizon']
        
        report += f"Based on backtesting results, the {best_model['model_type'].upper()} model with a "
        report += f"{best_horizon}-day prediction horizon demonstrated the strongest performance.\n\n"
        
        report += "Recommended implementation:\n\n"
        report += f"1. Deploy the {best_model['model_type'].upper()} model for {best_horizon}-day price movement predictions\n"
        report += "2. Generate trading signals based on predicted direction\n"
        report += "3. Combine with existing smart money indicators for confirmation\n"
        report += "4. Retrain model weekly with new data to maintain performance\n\n"
        
        report += "## Next Steps\n\n"
        report += "1. Implement ensemble approach combining multiple models\n"
        report += "2. Add more external features (market sentiment, on-chain metrics)\n"
        report += "3. Develop dynamic position sizing based on prediction confidence\n"
        
        return report
