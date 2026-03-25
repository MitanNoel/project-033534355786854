import re
import string

# Initialize Sastrawi stemmer and stopword remover (lazy initialization)
_stemmer = None
_stopwords = None

def get_stemmer():
    """Get or initialize Sastrawi stemmer"""
    global _stemmer
    if _stemmer is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            _stemmer = factory.create_stemmer()
        except ImportError:
            print("Warning: Sastrawi not installed. Stemming will be skipped.")
            _stemmer = None
    return _stemmer

def get_stopwords():
    """Get or initialize Indonesian stopwords"""
    global _stopwords
    if _stopwords is None:
        try:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            factory = StopWordRemoverFactory()
            _stopwords = factory.get_stop_words()
        except ImportError:
            print("Warning: Sastrawi not installed. Using basic stopwords list.")
            # Basic Indonesian stopwords as fallback
            _stopwords = set([
                'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
                'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'karena',
                'oleh', 'itu', 'dalam', 'bisa', 'dari', 'saya', 'dengan', 'akan', 'kami', 'telah'
            ])
    return _stopwords

def clean_text(text):
    """
    Basic text cleaning: lowercase and remove URLs

    Args:
        text: Input text string

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove mentions and hashtags (optional, keeping for now)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    return text

def to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()

def remove_punctuation(text):
    """
    Remove punctuation from text

    Args:
        text: Input text string

    Returns:
        str: Text without punctuation
    """
    # Remove all punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def remove_numbers(text):
    """
    Remove numbers from text

    Args:
        text: Input text string

    Returns:
        str: Text without numbers
    """
    text = re.sub(r'\d+', '', text)
    return text

def remove_extra_whitespace(text):
    """
    Remove extra whitespace from text

    Args:
        text: Input text string

    Returns:
        str: Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text

def handle_repeated_characters(text):
    """
    Handle repeated characters (e.g., 'senaaang' -> 'senang')

    Args:
        text: Input text string

    Returns:
        str: Text with normalized repeated characters
    """
    # Replace 3 or more repeated characters with single character
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def remove_stopwords_from_text(text, custom_stopwords=None):
    """
    Remove Indonesian stopwords from text

    Args:
        text: Input text string
        custom_stopwords: Optional set of custom stopwords to add

    Returns:
        str: Text without stopwords
    """
    stopwords = get_stopwords()

    if custom_stopwords:
        stopwords = stopwords.union(set(custom_stopwords))

    # Split text into words
    words = text.split()

    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords]

    return ' '.join(filtered_words)

def stem_text(text):
    """
    Apply Sastrawi stemming to Indonesian text

    Args:
        text: Input text string

    Returns:
        str: Stemmed text
    """
    stemmer = get_stemmer()

    if stemmer is None:
        return text

    return stemmer.stem(text)

def preprocess_text(text, config):
    """
    Apply preprocessing pipeline based on configuration

    Args:
        text: Input text string
        config: Preprocessing configuration dict with boolean flags

    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)

    # Always clean basic elements
    text = clean_text(text)

    # Apply lowercase if configured
    if config.get('lowercase', True):
        text = to_lowercase(text)

    # Handle repeated characters (before other processing)
    text = handle_repeated_characters(text)

    # Remove punctuation if configured
    if config.get('remove_punctuation', True):
        text = remove_punctuation(text)

    # Remove numbers if configured
    if config.get('remove_numbers', True):
        text = remove_numbers(text)

    # Remove extra whitespace
    text = remove_extra_whitespace(text)

    # Remove stopwords if configured
    if config.get('remove_stopwords', False):
        text = remove_stopwords_from_text(text)

    # Apply stemming if configured (should be last step)
    if config.get('stemming', False):
        text = stem_text(text)

    # Final whitespace cleanup
    text = remove_extra_whitespace(text)

    return text

def preprocess_texts(texts, config):
    """
    Apply preprocessing to a list of texts

    Args:
        texts: List of text strings
        config: Preprocessing configuration dict

    Returns:
        list: List of preprocessed texts
    """
    return [preprocess_text(text, config) for text in texts]

def get_preprocessing_preview(text, config):
    """
    Get before and after preview of preprocessing

    Args:
        text: Input text string
        config: Preprocessing configuration dict

    Returns:
        dict: Dictionary with 'original' and 'processed' texts
    """
    return {
        'original': text,
        'processed': preprocess_text(text, config)
    }
