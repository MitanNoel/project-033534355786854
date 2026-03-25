from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import joblib
import os
import time
from config import Config
from src.preprocessor import preprocess_texts
from src.feature_extractor import fit_transform_features, transform_features, save_vectorizer

def train_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Train Multinomial Naive Bayes classifier

    Args:
        X_train: Training features
        y_train: Training labels
        alpha: Smoothing parameter

    Returns:
        Trained model
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, C=1.0, kernel='linear', max_iter=1000):
    """
    Train SVM classifier

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter
        kernel: Kernel type (for LinearSVC, this is ignored)
        max_iter: Maximum number of iterations

    Returns:
        Trained model
    """
    model = LinearSVC(C=C, max_iter=max_iter, random_state=Config.RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def split_data(X, y, test_size=None, random_state=None):
    """
    Split data into training and testing sets

    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set (default from Config)
        random_state: Random state (default from Config)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = Config.TEST_SIZE
    if random_state is None:
        random_state = Config.RANDOM_STATE

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_model(model, filepath):
    """
    Save model to file

    Args:
        model: Trained model
        filepath: Path to save the model

    Returns:
        bool: True if successful
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filepath):
    """
    Load model from file

    Args:
        filepath: Path to the model file

    Returns:
        Loaded model
    """
    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def train_scenario(texts, labels, scenario_name, scenario_config):
    """
    Train both NB and SVM models for a scenario

    Args:
        texts: List of text strings
        labels: List of labels
        scenario_name: Name of the scenario (e.g., 'scenario_1')
        scenario_config: Configuration dictionary for the scenario

    Returns:
        dict: Training results with models, vectorizer, and metadata
    """
    start_time = time.time()

    # Preprocess texts
    print(f"  Preprocessing teks untuk {scenario_name}...")
    preprocessing_config = scenario_config['preprocessing']
    processed_texts = preprocess_texts(texts, preprocessing_config)

    # Extract features
    print(f"  Ekstraksi fitur untuk {scenario_name}...")
    vectorization_config = scenario_config['vectorization']
    X, vectorizer = fit_transform_features(
        processed_texts,
        method=vectorization_config['method'],
        max_features=vectorization_config['max_features'],
        ngram_range=vectorization_config['ngram_range']
    )

    # Split data
    print(f"  Membagi data training/testing untuk {scenario_name}...")
    X_train, X_test, y_train, y_test = split_data(X, labels)

    # Train Naive Bayes
    print(f"  Training Naive Bayes untuk {scenario_name}...")
    nb_params = scenario_config['models']['naive_bayes']
    nb_model = train_naive_bayes(X_train, y_train, alpha=nb_params['alpha'])

    # Train SVM
    print(f"  Training SVM untuk {scenario_name}...")
    svm_params = scenario_config['models']['svm']
    svm_model = train_svm(
        X_train, y_train,
        C=svm_params['C'],
        max_iter=svm_params.get('max_iter', 1000)
    )

    training_time = time.time() - start_time

    # Save models and vectorizer
    models_folder = Config.MODELS_FOLDER
    os.makedirs(models_folder, exist_ok=True)

    nb_path = os.path.join(models_folder, f'{scenario_name}_nb.joblib')
    svm_path = os.path.join(models_folder, f'{scenario_name}_svm.joblib')
    vectorizer_path = os.path.join(models_folder, f'vectorizer_{scenario_name}.joblib')

    save_model(nb_model, nb_path)
    save_model(svm_model, svm_path)
    save_vectorizer(vectorizer, vectorizer_path)

    return {
        'scenario_name': scenario_name,
        'scenario_display_name': scenario_config['name'],
        'models': {
            'naive_bayes': nb_model,
            'svm': svm_model
        },
        'model_paths': {
            'naive_bayes': nb_path,
            'svm': svm_path
        },
        'vectorizer': vectorizer,
        'vectorizer_path': vectorizer_path,
        'data_split': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        },
        'config': scenario_config,
        'training_time': training_time
    }

def train_all_scenarios(texts, labels, selected_scenarios=None):
    """
    Train models for all selected scenarios

    Args:
        texts: List of text strings
        labels: List of labels
        selected_scenarios: List of scenario names to train (None = all)

    Returns:
        dict: Training results for all scenarios
    """
    if selected_scenarios is None:
        selected_scenarios = list(Config.SCENARIOS.keys())

    all_results = {}

    for scenario_name in selected_scenarios:
        if scenario_name not in Config.SCENARIOS:
            print(f"Warning: Scenario {scenario_name} not found in config. Skipping.")
            continue

        print(f"\n=== Training {scenario_name} ===")
        scenario_config = Config.SCENARIOS[scenario_name]

        try:
            result = train_scenario(texts, labels, scenario_name, scenario_config)
            all_results[scenario_name] = result
            print(f"  ✓ {scenario_name} selesai dalam {result['training_time']:.2f} detik")
        except Exception as e:
            print(f"  ✗ Error training {scenario_name}: {str(e)}")
            all_results[scenario_name] = {'error': str(e)}

    return all_results
