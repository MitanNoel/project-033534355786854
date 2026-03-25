from src.model_trainer import load_model
from src.feature_extractor import load_vectorizer, transform_features
from src.preprocessor import preprocess_texts
from src.evaluator import load_best_model_metadata

def load_best_model():
    """
    Load the best model, vectorizer, and config

    Returns:
        tuple: (model, vectorizer, config, metadata) or (None, None, None, None) if not found
    """
    # Load metadata
    metadata = load_best_model_metadata()

    if metadata is None:
        print("Best model metadata not found. Please train models first.")
        return None, None, None, None

    best_model_info = metadata['best_model']

    # Load model
    model = load_model(best_model_info['model_path'])
    if model is None:
        print(f"Failed to load model from {best_model_info['model_path']}")
        return None, None, None, None

    # Load vectorizer
    vectorizer = load_vectorizer(best_model_info['vectorizer_path'])
    if vectorizer is None:
        print(f"Failed to load vectorizer from {best_model_info['vectorizer_path']}")
        return None, None, None, None

    # Get config
    config = best_model_info['config']

    return model, vectorizer, config, metadata

def predict_single(text, model, vectorizer, preprocessing_config):
    """
    Predict emotion for a single text

    Args:
        text: Input text string
        model: Trained model
        vectorizer: Fitted vectorizer
        preprocessing_config: Preprocessing configuration dict

    Returns:
        str: Predicted emotion label
    """
    # Preprocess text
    processed_text = preprocess_texts([text], preprocessing_config)[0]

    # Transform to features
    features = transform_features([processed_text], vectorizer)

    # Predict
    prediction = model.predict(features)[0]

    return prediction

def predict_batch(texts, model, vectorizer, preprocessing_config):
    """
    Predict emotions for multiple texts

    Args:
        texts: List of text strings
        model: Trained model
        vectorizer: Fitted vectorizer
        preprocessing_config: Preprocessing configuration dict

    Returns:
        list: List of predicted emotion labels
    """
    # Preprocess texts
    processed_texts = preprocess_texts(texts, preprocessing_config)

    # Transform to features
    features = transform_features(processed_texts, vectorizer)

    # Predict
    predictions = model.predict(features)

    return predictions.tolist()

def predict_with_best_model(texts):
    """
    Predict emotions using the best trained model

    Args:
        texts: List of text strings or single text string

    Returns:
        dict: Prediction results with metadata
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False

    # Load best model
    model, vectorizer, config, metadata = load_best_model()

    if model is None:
        return {
            'success': False,
            'error': 'Model belum tersedia. Silakan lakukan training terlebih dahulu.'
        }

    # Get preprocessing config
    preprocessing_config = config['preprocessing']

    # Make predictions
    try:
        predictions = predict_batch(texts, model, vectorizer, preprocessing_config)

        # Format results
        results = []
        for i, (text, prediction) in enumerate(zip(texts, predictions)):
            results.append({
                'no': i + 1,
                'text': text,
                'emotion': prediction
            })

        return {
            'success': True,
            'predictions': results,
            'model_info': {
                'scenario': metadata['best_model']['scenario'],
                'algorithm': metadata['best_model']['algorithm'],
                'accuracy': metadata['best_model']['metrics']['accuracy'],
                'f1_score': metadata['best_model']['metrics']['f1_score']
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Error saat melakukan prediksi: {str(e)}'
        }

def get_prediction_confidence(text, model, vectorizer, preprocessing_config):
    """
    Get prediction with confidence scores (for models that support it)

    Args:
        text: Input text string
        model: Trained model
        vectorizer: Fitted vectorizer
        preprocessing_config: Preprocessing configuration dict

    Returns:
        dict: Prediction and confidence scores
    """
    # Preprocess text
    processed_text = preprocess_texts([text], preprocessing_config)[0]

    # Transform to features
    features = transform_features([processed_text], vectorizer)

    # Predict
    prediction = model.predict(features)[0]

    # Try to get confidence scores (if model supports it)
    confidence_scores = None
    try:
        if hasattr(model, 'predict_proba'):
            # For models like MultinomialNB
            probabilities = model.predict_proba(features)[0]
            confidence_scores = {
                label: float(prob)
                for label, prob in zip(model.classes_, probabilities)
            }
        elif hasattr(model, 'decision_function'):
            # For models like LinearSVC
            decision_values = model.decision_function(features)[0]
            confidence_scores = {
                label: float(score)
                for label, score in zip(model.classes_, decision_values)
            }
    except Exception as e:
        print(f"Could not get confidence scores: {e}")

    return {
        'prediction': prediction,
        'confidence_scores': confidence_scores,
        'processed_text': processed_text
    }
