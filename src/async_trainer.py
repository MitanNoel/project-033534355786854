"""
Asynchronous training module to prevent browser timeouts
"""
import threading
import json
import os
from datetime import datetime
from config import Config
from src.model_trainer import train_all_scenarios
from src.evaluator import evaluate_all_scenarios, create_comparison_table, select_best_model, save_best_model_metadata

# Global training state
training_state = {
    'status': 'idle',  # idle, running, completed, error
    'progress': 0,
    'current_scenario': None,
    'message': '',
    'results': None,
    'error': None,
    'start_time': None,
    'end_time': None
}

def get_training_state():
    """Get current training state"""
    return training_state.copy()

def reset_training_state():
    """Reset training state to idle"""
    global training_state
    training_state = {
        'status': 'idle',
        'progress': 0,
        'current_scenario': None,
        'message': '',
        'results': None,
        'error': None,
        'start_time': None,
        'end_time': None
    }

def update_progress(scenario_name, progress, message):
    """Update training progress"""
    global training_state
    training_state['current_scenario'] = scenario_name
    training_state['progress'] = progress
    training_state['message'] = message

def train_async(texts, labels, selected_scenarios):
    """
    Asynchronous training function that runs in a separate thread
    
    Args:
        texts: List of text strings
        labels: List of labels
        selected_scenarios: List of scenario names to train
    """
    global training_state
    
    try:
        training_state['status'] = 'running'
        training_state['start_time'] = datetime.now().isoformat()
        training_state['progress'] = 0
        training_state['message'] = 'Memulai proses training...'
        
        total_scenarios = len(selected_scenarios)
        
        # Train each scenario
        all_results = {}
        for idx, scenario_name in enumerate(selected_scenarios):
            scenario_num = idx + 1
            progress = int((idx / total_scenarios) * 80)  # 0-80% for training
            
            update_progress(
                scenario_name, 
                progress,
                f'Training {scenario_name} ({scenario_num}/{total_scenarios})...'
            )
            
            scenario_config = Config.SCENARIOS[scenario_name]
            
            # Import here to avoid circular imports
            from src.model_trainer import train_scenario
            
            try:
                result = train_scenario(texts, labels, scenario_name, scenario_config)
                all_results[scenario_name] = result
            except Exception as e:
                all_results[scenario_name] = {'error': str(e)}
        
        # Evaluate models
        update_progress(None, 85, 'Mengevaluasi model...')
        evaluation_results = evaluate_all_scenarios(all_results)
        
        # Create comparison
        update_progress(None, 90, 'Membuat perbandingan...')
        comparison_data = create_comparison_table(evaluation_results)
        
        # Select best model
        update_progress(None, 95, 'Memilih model terbaik...')
        best_model = select_best_model(comparison_data)
        
        # Save metadata
        if best_model:
            save_best_model_metadata(best_model, evaluation_results)
        
        # Convert to JSON-serializable format before storing
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Store results (already converted)
        training_state['results'] = {
            'comparison_data': convert_numpy(comparison_data),
            'best_model': convert_numpy(best_model),
            'evaluation_results': convert_numpy(evaluation_results)
        }
        
        training_state['status'] = 'completed'
        training_state['progress'] = 100
        training_state['message'] = 'Training selesai!'
        training_state['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        training_state['status'] = 'error'
        training_state['error'] = str(e)
        training_state['message'] = f'Error: {str(e)}'
        training_state['end_time'] = datetime.now().isoformat()

def start_training_thread(texts, labels, selected_scenarios):
    """
    Start training in a background thread
    
    Args:
        texts: List of text strings
        labels: List of labels
        selected_scenarios: List of scenario names to train
    
    Returns:
        bool: True if training started successfully
    """
    global training_state
    
    # Don't start if already running
    if training_state['status'] == 'running':
        return False
    
    # Reset state
    reset_training_state()
    
    # Start training thread
    thread = threading.Thread(
        target=train_async,
        args=(texts, labels, selected_scenarios),
        daemon=True
    )
    thread.start()
    
    return True
