import numpy as np
from sklearn.metrics import confusion_matrix

def large_move_metrics(y_true, y_pred, threshold=0.5):
    """
    Computes metrics focused only on large price moves with error relative to prediction size.
    
    Parameters:
    - y_true: np.ndarray, true target values
    - y_pred: np.ndarray, predicted target values
    - threshold: float, defines a 'large move' (e.g., 0.02 = 2%)
    
    Returns:
    - metrics: dict with direction accuracy, relative MSE, relative MAE, count of large moves
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Select only large moves based on predicted values
    large_moves_idx = np.where(np.abs(y_true) >= threshold)[0]

    if len(large_moves_idx) == 0:
        print("Warning: No large moves found above threshold.")
        return None

    y_true_large = y_true[large_moves_idx]
    y_pred_large = y_pred[large_moves_idx]

    # Directional accuracy (sign matching)
    true_sign = np.sign(y_true_large)
    pred_sign = np.sign(y_pred_large)
    directional_accuracy = np.mean(true_sign == pred_sign)

    # Relative MSE and MAE only for large moves
    relative_mse = np.mean(np.square(y_true_large - y_pred_large) / np.square(np.abs(y_pred_large)))
    relative_mae = np.mean(np.abs(y_true_large - y_pred_large) / np.abs(y_pred_large))

    # New combined metric that incorporates directional accuracy, relative MSE, and large move count
    relative_errors = (np.abs(y_true_large - y_pred_large) / np.abs(y_pred_large))

    combined_metric = directional_accuracy * relative_mae

    metrics = {
        'directional_accuracy': directional_accuracy,
        'relative_mse': relative_mse,
        'relative_mae': relative_mae,
        'combined_metric': combined_metric  # New combined metric
    }

    return metrics

def get_model_comparison(model_results, threshold=1):
    model_comparison = []
    
    for model_name, results in model_results.items():
        y_pred = results["pred"]
        y_real = results["real"]
       
        # Compute the large move metrics
        metrics = large_move_metrics_2(y_real, y_pred, threshold=threshold)
       
        if metrics is not None:
            metrics["model_name"] = model_name
            model_comparison.append(metrics)
    
    return model_comparison


def large_move_metrics_2(y_true, y_pred, threshold=0.5, prediction_weight=0.4, freq_weight=0.3, error_weight=0.3):
    """
    Computes comprehensive metrics focused on large price moves with error relative to prediction size.
   
    Parameters:
    - y_true: np.ndarray, true target values
    - y_pred: np.ndarray, predicted target values
    - threshold: float, defines a 'large move' (e.g., 0.02 = 2%)
    - prediction_weight: float, weight for directional accuracy in combined metric
    - freq_weight: float, weight for prediction frequency in combined metric
    - error_weight: float, weight for error component in combined metric
   
    Returns:
    - metrics: dict with comprehensive evaluation metrics for large price moves
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Identify large moves in true and predicted values
    true_large_idx = np.where(np.abs(y_true) >= threshold)[0]
    pred_large_idx = np.where(np.abs(y_pred) >= threshold)[0]
    
    # Handle case with no large moves
    if len(true_large_idx) == 0:
        print("Warning: No large moves found in true values above threshold.")
        return None
    
    # Calculate metrics for all data points
    all_directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # Get large move samples from true values
    y_true_large = y_true[true_large_idx]
    y_pred_for_true_large = y_pred[true_large_idx]
    
    # Calculate directional accuracy for large moves
    true_sign = np.sign(y_true_large)
    pred_sign = np.sign(y_pred_for_true_large)
    directional_accuracy = np.mean(true_sign == pred_sign)
    
    # Calculate magnitude accuracy for large moves
    # Use clip to prevent division by zero or extremely small values
    epsilon = 1e-10  # Small value to prevent division by zero
    abs_true_large = np.abs(y_true_large)
    abs_pred_for_true_large = np.abs(y_pred_for_true_large).clip(min=epsilon)
    
    # Relative error metrics
    relative_mse = np.mean(np.square(y_true_large - y_pred_for_true_large) / np.square(abs_true_large))
    relative_mae = np.mean(np.abs(y_true_large - y_pred_for_true_large) / abs_true_large)
    
    # Calculate magnitude ratio (how well the model captures the magnitude)
    magnitude_ratio = np.mean(abs_pred_for_true_large / abs_true_large)
    
    # Binary classification metrics for large moves
    # Create binary arrays for detected large moves
    y_true_binary = np.zeros_like(y_true)
    y_true_binary[true_large_idx] = 1
    
    y_pred_binary = np.zeros_like(y_pred)
    y_pred_binary[pred_large_idx] = 1
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    # Calculate precision, recall and F1 for large move detection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Frequency metrics
    true_frequency = len(true_large_idx) / len(y_true)
    pred_frequency = len(pred_large_idx) / len(y_pred)
    frequency_ratio = pred_frequency / true_frequency if true_frequency > 0 else 0
    
    # Calculate new combined metric
    # Balance between directional accuracy, frequency matching, and error metrics
    error_component = 1 / (1 + relative_mae)  # Transforms error to [0,1] range where higher is better
    frequency_component = 1 - abs(1 - frequency_ratio) if frequency_ratio <= 2 else 1 / frequency_ratio
    
    combined_metric = (
        prediction_weight * directional_accuracy +
        freq_weight * frequency_component +
        error_weight * error_component
    ) * f1_score  # Scale by F1 to ensure both precision and recall matter
    
    # Information value - how much $ could be made per prediction on average
    expected_value = np.mean(np.abs(y_true_large) * (2 * (true_sign == pred_sign).astype(float) - 1))
    
    metrics = {
        'directional_accuracy': directional_accuracy,
        'all_directional_accuracy': all_directional_accuracy,
        'relative_mse': relative_mse,
        'relative_mae': relative_mae,
        'magnitude_ratio': magnitude_ratio,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_large_count': len(true_large_idx),
        'pred_large_count': len(pred_large_idx),
        'frequency_ratio': frequency_ratio,
        'expected_value': expected_value,
        'combined_metric': combined_metric
    }
    
    return metrics