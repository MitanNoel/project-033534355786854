import pandas as pd
from config import Config

def validate_csv(file_path):
    """
    Validate CSV file format and structure

    Args:
        file_path: Path to the CSV file

    Returns:
        dict: Validation result with status, message, and data info
    """
    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path, encoding='utf-8')

        # Check if dataframe is empty
        if df.empty:
            return {
                'valid': False,
                'message': 'File CSV kosong. Silakan upload file yang berisi data.',
                'data': None
            }

        # Check minimum rows requirement
        if len(df) < Config.MIN_DATA_POINTS:
            return {
                'valid': False,
                'message': f'Data tidak mencukupi. Minimal {Config.MIN_DATA_POINTS} baris data diperlukan. File Anda memiliki {len(df)} baris.',
                'data': None
            }

        # Check if dataframe has at least 2 columns
        if len(df.columns) < 2:
            return {
                'valid': False,
                'message': 'File CSV harus memiliki minimal 2 kolom (teks dan label emosi).',
                'data': None
            }

        return {
            'valid': True,
            'message': f'File valid! Total {len(df)} baris data ditemukan.',
            'data': df
        }

    except UnicodeDecodeError:
        # Try with different encoding
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            if df.empty:
                return {
                    'valid': False,
                    'message': 'File CSV kosong. Silakan upload file yang berisi data.',
                    'data': None
                }

            if len(df) < Config.MIN_DATA_POINTS:
                return {
                    'valid': False,
                    'message': f'Data tidak mencukupi. Minimal {Config.MIN_DATA_POINTS} baris data diperlukan. File Anda memiliki {len(df)} baris.',
                    'data': None
                }

            return {
                'valid': True,
                'message': f'File valid! Total {len(df)} baris data ditemukan.',
                'data': df
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'Error membaca file CSV: {str(e)}. Pastikan file dalam format CSV yang valid.',
                'data': None
            }
    except Exception as e:
        return {
            'valid': False,
            'message': f'Error membaca file CSV: {str(e)}',
            'data': None
        }

def load_data(file_path):
    """
    Load CSV data into pandas DataFrame

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except UnicodeDecodeError:
        # Try with different encoding
        df = pd.read_csv(file_path, encoding='latin-1')
        return df

def get_column_names(df):
    """
    Get column names from DataFrame

    Args:
        df: pandas DataFrame

    Returns:
        list: List of column names
    """
    return df.columns.tolist()

def check_data_quality(df, text_col, label_col):
    """
    Check data quality for missing values and distribution

    Args:
        df: pandas DataFrame
        text_col: Name of text column
        label_col: Name of label/emotion column

    Returns:
        dict: Data quality report
    """
    report = {
        'total_rows': len(df),
        'missing_text': df[text_col].isna().sum(),
        'missing_label': df[label_col].isna().sum(),
        'unique_labels': df[label_col].nunique(),
        'label_distribution': df[label_col].value_counts().to_dict(),
        'has_issues': False,
        'messages': []
    }

    # Check for missing values
    if report['missing_text'] > 0:
        report['has_issues'] = True
        report['messages'].append(f"Ditemukan {report['missing_text']} baris dengan teks kosong.")

    if report['missing_label'] > 0:
        report['has_issues'] = True
        report['messages'].append(f"Ditemukan {report['missing_label']} baris dengan label kosong.")

    # Check for class imbalance
    if report['unique_labels'] < 2:
        report['has_issues'] = True
        report['messages'].append("Data harus memiliki minimal 2 kategori emosi yang berbeda.")

    if not report['has_issues']:
        report['messages'].append("Kualitas data baik! Tidak ada masalah ditemukan.")

    return report

def clean_data(df, text_col, label_col):
    """
    Clean data by removing rows with missing values

    Args:
        df: pandas DataFrame
        text_col: Name of text column
        label_col: Name of label/emotion column

    Returns:
        DataFrame: Cleaned data
    """
    # Remove rows with missing values in text or label columns
    df_cleaned = df.dropna(subset=[text_col, label_col])

    # Remove rows with empty strings
    df_cleaned = df_cleaned[df_cleaned[text_col].str.strip() != '']
    df_cleaned = df_cleaned[df_cleaned[label_col].str.strip() != '']

    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned

def prepare_data(df, text_col, label_col):
    """
    Prepare data for model training

    Args:
        df: pandas DataFrame
        text_col: Name of text column
        label_col: Name of label/emotion column

    Returns:
        tuple: (texts, labels) as lists
    """
    # Clean data first
    df_clean = clean_data(df, text_col, label_col)

    # Extract texts and labels
    texts = df_clean[text_col].tolist()
    labels = df_clean[label_col].tolist()

    return texts, labels
