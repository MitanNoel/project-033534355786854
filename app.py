from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from config import Config
from src.data_handler import validate_csv, load_data, get_column_names, check_data_quality, prepare_data
from src.preprocessor import get_preprocessing_preview
from src.model_trainer import train_all_scenarios
from src.evaluator import evaluate_all_scenarios, create_comparison_table, select_best_model, save_best_model_metadata, get_confusion_matrix
from src.predictor import predict_with_best_model, load_best_model

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload and model folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

def convert_to_serializable(obj):
    """
    Convert numpy/pandas types to Python native types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """CSV upload page"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('Tidak ada file yang diupload', 'danger')
            return redirect(request.url)

        file = request.files['file']

        # Check if file was selected
        if file.filename == '':
            flash('Tidak ada file yang dipilih', 'danger')
            return redirect(request.url)

        # Check if file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Validate CSV
            validation_result = validate_csv(filepath)

            if not validation_result['valid']:
                # Delete invalid file
                os.remove(filepath)
                flash(validation_result['message'], 'danger')
                return redirect(request.url)

            # Store file path in session
            session['uploaded_file'] = filepath
            session['filename'] = filename

            # Load data for preview
            df = validation_result['data']
            columns = get_column_names(df)

            # Store data info in session (convert to native Python types)
            session['total_rows'] = int(len(df))
            session['columns'] = columns

            flash(validation_result['message'], 'success')

            # Show column selection page
            return render_template('upload.html',
                                   uploaded=True,
                                   filename=filename,
                                   columns=columns,
                                   preview_data=df.head(10).to_dict('records'),
                                   total_rows=len(df))

        else:
            flash('Format file tidak valid. Silakan upload file CSV.', 'danger')
            return redirect(request.url)

    return render_template('upload.html', uploaded=False)

@app.route('/select_columns', methods=['POST'])
def select_columns():
    """Handle column selection"""
    text_col = request.form.get('text_col')
    label_col = request.form.get('label_col')

    if not text_col or not label_col:
        flash('Silakan pilih kolom teks dan label', 'danger')
        return redirect(url_for('upload'))

    if text_col == label_col:
        flash('Kolom teks dan label harus berbeda', 'danger')
        return redirect(url_for('upload'))

    # Store in session
    session['text_col'] = text_col
    session['label_col'] = label_col

    # Load data and check quality
    filepath = session.get('uploaded_file')
    df = load_data(filepath)

    # Check data quality
    quality_report = check_data_quality(df, text_col, label_col)

    if quality_report['has_issues']:
        for message in quality_report['messages']:
            flash(message, 'warning')

    # Store quality report in session (convert numpy types to native Python)
    session['quality_report'] = convert_to_serializable(quality_report)

    return redirect(url_for('preprocessing'))

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    """Preprocessing configuration page"""
    if 'uploaded_file' not in session:
        flash('Silakan upload file CSV terlebih dahulu', 'warning')
        return redirect(url_for('upload'))

    if request.method == 'POST':
        # Get preprocessing options
        preprocessing_config = {
            'lowercase': 'lowercase' in request.form,
            'remove_punctuation': 'remove_punctuation' in request.form,
            'remove_numbers': 'remove_numbers' in request.form,
            'remove_stopwords': 'remove_stopwords' in request.form,
            'stemming': 'stemming' in request.form
        }

        # Store in session
        session['preprocessing_config'] = preprocessing_config

        return redirect(url_for('training'))

    # Get sample text for preview
    filepath = session.get('uploaded_file')
    text_col = session.get('text_col')

    if filepath and text_col:
        df = load_data(filepath)
        sample_text = df[text_col].iloc[0]
    else:
        sample_text = "Hari ini saya sangat bahagia!"

    return render_template('preprocessing.html', sample_text=sample_text)

@app.route('/preview_preprocessing', methods=['POST'])
def preview_preprocessing():
    """AJAX endpoint for preprocessing preview"""
    data = request.get_json()
    text = data.get('text', '')
    config = data.get('config', {})

    preview = get_preprocessing_preview(text, config)

    return jsonify(preview)

@app.route('/training', methods=['GET', 'POST'])
def training():
    """Model training page"""
    # Check if file is uploaded
    if 'uploaded_file' not in session:
        flash('Silakan upload file CSV terlebih dahulu', 'warning')
        return redirect(url_for('upload'))

    # Check if columns are selected
    if 'text_col' not in session or 'label_col' not in session:
        flash('Silakan pilih kolom teks dan label terlebih dahulu', 'warning')
        return redirect(url_for('upload'))

    # Use default preprocessing config if not set
    if 'preprocessing_config' not in session:
        session['preprocessing_config'] = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': True,
            'remove_stopwords': False,
            'stemming': False
        }

    if request.method == 'POST':
        # Get selected scenarios
        selected_scenarios = request.form.getlist('scenarios')

        if not selected_scenarios:
            flash('Pilih minimal satu skenario untuk dilatih', 'warning')
            return redirect(request.url)

        # Load data
        filepath = session.get('uploaded_file')
        text_col = session.get('text_col')
        label_col = session.get('label_col')

        df = load_data(filepath)
        texts, labels = prepare_data(df, text_col, label_col)

        # Train models
        try:
            flash('Memulai proses training...', 'info')
            training_results = train_all_scenarios(texts, labels, selected_scenarios)

            # Evaluate models
            evaluation_results = evaluate_all_scenarios(training_results)

            # Create comparison table
            comparison_data = create_comparison_table(evaluation_results)

            # Select best model
            best_model = select_best_model(comparison_data)

            # Save best model metadata
            if best_model:
                save_best_model_metadata(best_model, evaluation_results)

            # Store results in session (convert numpy types to native Python)
            session['comparison_data'] = convert_to_serializable(comparison_data)
            session['best_model'] = convert_to_serializable(best_model)
            # Note: evaluation_results contains model objects, so we don't store it in session
            # It's only used temporarily for saving best model metadata

            flash('Training selesai!', 'success')
            return redirect(url_for('evaluation'))

        except Exception as e:
            flash(f'Error saat training: {str(e)}', 'danger')
            return redirect(request.url)

    # Get scenario configurations
    scenarios = Config.SCENARIOS

    return render_template('training.html', scenarios=scenarios)

@app.route('/evaluation')
def evaluation():
    """Model evaluation results page"""
    if 'comparison_data' not in session:
        flash('Silakan lakukan training terlebih dahulu', 'warning')
        return redirect(url_for('training'))

    comparison_data = session.get('comparison_data')
    best_model = session.get('best_model')

    return render_template('evaluation.html',
                           comparison_data=comparison_data,
                           best_model=best_model,
                           emotion_colors=Config.EMOTION_COLORS)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """Prediction interface page"""
    # Check if best model exists
    model, vectorizer, config, metadata = load_best_model()

    if model is None:
        flash('Model belum tersedia. Silakan lakukan training terlebih dahulu.', 'warning')
        return redirect(url_for('training'))

    if request.method == 'POST':
        # Get 5 text inputs
        texts = []
        for i in range(1, 6):
            text = request.form.get(f'text_{i}', '').strip()
            if text:
                texts.append(text)

        if not texts:
            flash('Silakan masukkan minimal satu teks untuk diprediksi', 'warning')
            return redirect(request.url)

        # Make predictions
        prediction_result = predict_with_best_model(texts)

        if prediction_result['success']:
            # Store in session (convert numpy types to native Python)
            session['prediction_results'] = convert_to_serializable(prediction_result['predictions'])
            session['model_info'] = convert_to_serializable(prediction_result['model_info'])

            return redirect(url_for('results'))
        else:
            flash(prediction_result['error'], 'danger')
            return redirect(request.url)

    # Get model info for display
    model_info = {
        'scenario': metadata['best_model']['scenario'],
        'algorithm': metadata['best_model']['algorithm'],
        'accuracy': round(metadata['best_model']['metrics']['accuracy'], 4),
        'f1_score': round(metadata['best_model']['metrics']['f1_score'], 4)
    }

    return render_template('prediction.html', model_info=model_info)

@app.route('/results')
def results():
    """Prediction results page"""
    if 'prediction_results' not in session:
        flash('Silakan lakukan prediksi terlebih dahulu', 'warning')
        return redirect(url_for('prediction'))

    prediction_results = session.get('prediction_results')
    model_info = session.get('model_info')

    return render_template('results.html',
                           predictions=prediction_results,
                           model_info=model_info,
                           emotion_colors=Config.EMOTION_COLORS)

@app.route('/reset')
def reset():
    """Reset session and start over"""
    # Clean up uploaded file
    if 'uploaded_file' in session:
        filepath = session.get('uploaded_file')
        if os.path.exists(filepath):
            os.remove(filepath)

    # Clear session
    session.clear()

    flash('Session direset. Silakan mulai dari awal.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
