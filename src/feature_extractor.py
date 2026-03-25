from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
import os

def create_vectorizer(method='tfidf', max_features=5000, ngram_range=(1, 2)):
    """
    Create a text vectorizer

    Args:
        method: 'tfidf' or 'count'
        max_features: Maximum number of features
        ngram_range: Tuple of (min_n, max_n) for n-grams

    Returns:
        Vectorizer object
    """
    if method == 'tfidf':
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
    elif method == 'count':
        return CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tfidf' or 'count'.")

def fit_transform_features(texts, method='tfidf', max_features=5000, ngram_range=(1, 2)):
    """
    Fit vectorizer on texts and transform them to features

    Args:
        texts: List of text strings
        method: 'tfidf' or 'count'
        max_features: Maximum number of features
        ngram_range: Tuple of (min_n, max_n) for n-grams

    Returns:
        tuple: (features matrix, fitted vectorizer)
    """
    vectorizer = create_vectorizer(method, max_features, ngram_range)
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def transform_features(texts, vectorizer):
    """
    Transform texts to features using a fitted vectorizer

    Args:
        texts: List of text strings
        vectorizer: Fitted vectorizer object

    Returns:
        Features matrix
    """
    return vectorizer.transform(texts)

def save_vectorizer(vectorizer, filepath):
    """
    Save vectorizer to file

    Args:
        vectorizer: Fitted vectorizer object
        filepath: Path to save the vectorizer

    Returns:
        bool: True if successful
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(vectorizer, filepath)
        return True
    except Exception as e:
        print(f"Error saving vectorizer: {e}")
        return False

def load_vectorizer(filepath):
    """
    Load vectorizer from file

    Args:
        filepath: Path to the vectorizer file

    Returns:
        Loaded vectorizer object
    """
    try:
        vectorizer = joblib.load(filepath)
        return vectorizer
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return None

def get_feature_names(vectorizer):
    """
    Get feature names from vectorizer

    Args:
        vectorizer: Fitted vectorizer object

    Returns:
        list: List of feature names
    """
    try:
        # For newer versions of scikit-learn
        return vectorizer.get_feature_names_out()
    except AttributeError:
        # For older versions of scikit-learn
        return vectorizer.get_feature_names()

def get_top_features(vectorizer, feature_vector, top_n=10):
    """
    Get top N features from a feature vector

    Args:
        vectorizer: Fitted vectorizer object
        feature_vector: Sparse feature vector
        top_n: Number of top features to return

    Returns:
        list: List of tuples (feature_name, score)
    """
    feature_names = get_feature_names(vectorizer)
    feature_array = feature_vector.toarray()[0]

    # Get indices of top features
    top_indices = feature_array.argsort()[-top_n:][::-1]

    # Get feature names and scores
    top_features = [(feature_names[i], feature_array[i]) for i in top_indices]

    return top_features
