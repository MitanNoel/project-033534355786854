import os

class Config:
    """Configuration class for the Indonesian Emotion Detection application"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size (Codespace limit)
    ALLOWED_EXTENSIONS = {'csv'}

    # Data requirements
    MIN_DATA_POINTS = 800

    # Model settings
    MODELS_FOLDER = 'models'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Emotion labels
    EMOTION_LABELS = ['happy', 'sadness', 'anger', 'fear', 'love']

    # Emotion colors for UI badges
    EMOTION_COLORS = {
        'happy': '#28a745',      # Green
        'sadness': '#6c757d',    # Gray
        'anger': '#dc3545',      # Red
        'fear': '#ffc107',       # Yellow/Orange
        'love': '#e83e8c'        # Pink
    }

    # Modeling scenarios configuration
    SCENARIOS = {
        'scenario_1': {
            'name': 'Skenario 1: Konfigurasi Dasar',
            'description': 'Preprocessing dasar dengan TF-IDF unigram',
            'preprocessing': {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_numbers': True,
                'remove_stopwords': False,
                'stemming': False
            },
            'vectorization': {
                'method': 'tfidf',
                'max_features': 3000,
                'ngram_range': (1, 1)
            },
            'models': {
                'naive_bayes': {'alpha': 1.0},
                'svm': {'C': 1.0, 'kernel': 'linear', 'max_iter': 1000}
            }
        },
        'scenario_2': {
            'name': 'Skenario 2: Preprocessing Lanjutan',
            'description': 'Preprocessing lengkap dengan TF-IDF bigram',
            'preprocessing': {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_numbers': True,
                'remove_stopwords': True,
                'stemming': True
            },
            'vectorization': {
                'method': 'tfidf',
                'max_features': 5000,
                'ngram_range': (1, 2)
            },
            'models': {
                'naive_bayes': {'alpha': 0.5},
                'svm': {'C': 1.0, 'kernel': 'linear', 'max_iter': 1000}
            }
        },
        'scenario_3': {
            'name': 'Skenario 3: Count Vectorizer',
            'description': 'CountVectorizer dengan preprocessing lengkap',
            'preprocessing': {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_numbers': True,
                'remove_stopwords': True,
                'stemming': True
            },
            'vectorization': {
                'method': 'count',
                'max_features': 5000,
                'ngram_range': (1, 2)
            },
            'models': {
                'naive_bayes': {'alpha': 1.0},
                'svm': {'C': 1.0, 'kernel': 'linear', 'max_iter': 1000}
            }
        },
        'scenario_4': {
            'name': 'Skenario 4: Konfigurasi Optimal',
            'description': 'TF-IDF trigram dengan hyperparameter optimal',
            'preprocessing': {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_numbers': True,
                'remove_stopwords': True,
                'stemming': True
            },
            'vectorization': {
                'method': 'tfidf',
                'max_features': 7000,
                'ngram_range': (1, 3)
            },
            'models': {
                'naive_bayes': {'alpha': 0.1},
                'svm': {'C': 10.0, 'kernel': 'linear', 'max_iter': 1000}
            }
        }
    }
