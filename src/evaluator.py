from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import json
import os
from config import Config

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model with detailed metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Evaluation metrics including per-class details
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate overall metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    
    # Get per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate per-class metrics for each emotion
    per_class_metrics = {}
    for label in set(y_test):
        if label in report:
            per_class_metrics[label] = {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1_score': report[label]['f1-score'],
                'support': report[label]['support']
            }
    
    metrics['per_class'] = per_class_metrics
    metrics['confusion_matrix'] = cm
    
    return metrics, y_pred

def evaluate_scenario(training_result):
    """
    Evaluate both models in a scenario

    Args:
        training_result: Training result dictionary from model_trainer

    Returns:
        dict: Evaluation results for both models
    """
    if 'error' in training_result:
        return {'error': training_result['error']}

    data_split = training_result['data_split']
    X_test = data_split['X_test']
    y_test = data_split['y_test']

    # Evaluate Naive Bayes
    nb_model = training_result['models']['naive_bayes']
    nb_metrics, nb_predictions = evaluate_model(nb_model, X_test, y_test)

    # Evaluate SVM
    svm_model = training_result['models']['svm']
    svm_metrics, svm_predictions = evaluate_model(svm_model, X_test, y_test)

    return {
        'scenario_name': training_result['scenario_name'],
        'scenario_display_name': training_result['scenario_display_name'],
        'naive_bayes': {
            'metrics': nb_metrics,
            'predictions': nb_predictions,
            'model_path': training_result['model_paths']['naive_bayes']
        },
        'svm': {
            'metrics': svm_metrics,
            'predictions': svm_predictions,
            'model_path': training_result['model_paths']['svm']
        },
        'vectorizer_path': training_result['vectorizer_path'],
        'config': training_result['config'],
        'training_time': training_result['training_time'],
        'y_test': y_test
    }

def evaluate_all_scenarios(training_results):
    """
    Evaluate all trained scenarios

    Args:
        training_results: Dictionary of training results from train_all_scenarios

    Returns:
        list: List of evaluation results
    """
    evaluation_results = []

    for scenario_name, training_result in training_results.items():
        print(f"Evaluating {scenario_name}...")
        eval_result = evaluate_scenario(training_result)
        evaluation_results.append(eval_result)

    return evaluation_results

def create_comparison_table(evaluation_results):
    """
    Create comparison table for all models
    
    Args:
        evaluation_results: List of evaluation results
    
    Returns:
        list: List of dictionaries with comparison data
    """
    comparison_data = []

    for eval_result in evaluation_results:
        if 'error' in eval_result:
            continue

        scenario_name = eval_result['scenario_display_name']
        training_time = eval_result.get('training_time', 0)
        
        # If training_time is a dict, extract per-model times, otherwise use total time
        if isinstance(training_time, dict):
            nb_time = training_time.get('naive_bayes', 0)
            svm_time = training_time.get('svm', 0)
        else:
            # If it's a single float, split it approximately (NB is usually faster)
            nb_time = training_time * 0.4  # Naive Bayes typically takes ~40%
            svm_time = training_time * 0.6  # SVM typically takes ~60%

        # Naive Bayes row
        nb_metrics = eval_result['naive_bayes']['metrics']
        comparison_data.append({
            'scenario': scenario_name,
            'scenario_id': eval_result['scenario_name'],
            'algorithm': 'Naive Bayes',
            'algorithm_id': 'naive_bayes',
            'accuracy': round(nb_metrics['accuracy'], 4),
            'precision': round(nb_metrics['precision'], 4),
            'recall': round(nb_metrics['recall'], 4),
            'f1_score': round(nb_metrics['f1_score'], 4),
            'training_time': round(nb_time, 2),
            'per_class': nb_metrics.get('per_class', {}),
            'confusion_matrix': nb_metrics.get('confusion_matrix', []).tolist() if hasattr(nb_metrics.get('confusion_matrix', []), 'tolist') else [],
            'model_path': eval_result['naive_bayes']['model_path'],
            'vectorizer_path': eval_result['vectorizer_path'],
            'config': eval_result['config']
        })

        # SVM row
        svm_metrics = eval_result['svm']['metrics']
        comparison_data.append({
            'scenario': scenario_name,
            'scenario_id': eval_result['scenario_name'],
            'algorithm': 'SVM',
            'algorithm_id': 'svm',
            'accuracy': round(svm_metrics['accuracy'], 4),
            'precision': round(svm_metrics['precision'], 4),
            'recall': round(svm_metrics['recall'], 4),
            'f1_score': round(svm_metrics['f1_score'], 4),
            'training_time': round(svm_time, 2),
            'per_class': svm_metrics.get('per_class', {}),
            'confusion_matrix': svm_metrics.get('confusion_matrix', []).tolist() if hasattr(svm_metrics.get('confusion_matrix', []), 'tolist') else [],
            'model_path': eval_result['svm']['model_path'],
            'vectorizer_path': eval_result['vectorizer_path'],
            'config': eval_result['config']
        })

    return comparison_data

def select_best_model(comparison_data):
    """
    Select the best model based on F1-score and accuracy

    Args:
        comparison_data: List of model comparison data

    Returns:
        dict: Best model information
    """
    if not comparison_data:
        return None

    # Sort by F1-score (primary) and accuracy (secondary)
    best = max(comparison_data, key=lambda x: (x['f1_score'], x['accuracy']))

    return best

def get_confusion_matrix(eval_result, algorithm='svm'):
    """
    Get confusion matrix for a specific model

    Args:
        eval_result: Evaluation result dictionary
        algorithm: 'naive_bayes' or 'svm'

    Returns:
        numpy array: Confusion matrix
    """
    y_test = eval_result['y_test']
    y_pred = eval_result[algorithm]['predictions']

    cm = confusion_matrix(y_test, y_pred)
    return cm

def get_classification_report_dict(eval_result, algorithm='svm'):
    """
    Get classification report as dictionary

    Args:
        eval_result: Evaluation result dictionary
        algorithm: 'naive_bayes' or 'svm'

    Returns:
        dict: Classification report
    """
    y_test = eval_result['y_test']
    y_pred = eval_result[algorithm]['predictions']

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report

def save_best_model_metadata(best_model_info, evaluation_results):
    """
    Save best model metadata to JSON file

    Args:
        best_model_info: Best model dictionary
        evaluation_results: All evaluation results

    Returns:
        bool: True if successful
    """
    metadata = {
        'best_model': {
            'scenario': best_model_info['scenario'],
            'scenario_id': best_model_info['scenario_id'],
            'algorithm': best_model_info['algorithm'],
            'algorithm_id': best_model_info['algorithm_id'],
            'model_path': best_model_info['model_path'],
            'vectorizer_path': best_model_info['vectorizer_path'],
            'metrics': {
                'accuracy': best_model_info['accuracy'],
                'precision': best_model_info['precision'],
                'recall': best_model_info['recall'],
                'f1_score': best_model_info['f1_score']
            },
            'config': best_model_info['config']
        }
    }

    try:
        metadata_path = os.path.join(Config.MODELS_FOLDER, 'best_model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def load_best_model_metadata():
    """
    Load best model metadata from JSON file

    Returns:
        dict: Best model metadata or None if not found
    """
    try:
        metadata_path = os.path.join(Config.MODELS_FOLDER, 'best_model_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None
