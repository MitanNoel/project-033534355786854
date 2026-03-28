#!/usr/bin/env python3
"""
Complete Experiment Runner for Indonesian Emotion Detection
Generates comprehensive results, comparison tables, confusion matrices, and visualizations
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from src.data_handler import load_data, prepare_data, clean_data, check_data_quality
from src.preprocessor import preprocess_texts, clean_text
from src.feature_extractor import fit_transform_features
from src.model_trainer import train_naive_bayes, train_svm, split_data, save_model
from src.evaluator import evaluate_model, create_comparison_table, select_best_model, save_best_model_metadata

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')
sns.set_style("whitegrid")

def run_full_experiment():
    """Run complete experiment pipeline"""

    print("=" * 80)
    print("INDONESIAN EMOTION DETECTION - COMPLETE EXPERIMENT")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load dataset
    print("[1/6] Loading dataset...")
    dataset_path = "uploads/Twitter_Emotion_Dataset.csv"
    df = load_data(dataset_path)
    print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {df.columns.tolist()}")

    # Data quality check
    print("\n[2/6] Checking data quality...")
    text_col = 'tweet'
    label_col = 'label'

    quality_report = check_data_quality(df, text_col, label_col)
    print(f"✓ Total rows: {quality_report['total_rows']}")
    print(f"✓ Missing text: {quality_report['missing_text']}")
    print(f"✓ Missing labels: {quality_report['missing_label']}")
    print(f"✓ Unique labels: {quality_report['unique_labels']}")
    print(f"✓ Label distribution:")
    for label, count in quality_report['label_distribution'].items():
        print(f"    - {label}: {count} ({count/quality_report['total_rows']*100:.1f}%)")

    # Clean data
    df_clean = clean_data(df, text_col, label_col)
    texts, labels = prepare_data(df, text_col, label_col)
    print(f"\n✓ Data cleaned: {len(texts)} valid samples")

    # Initialize results storage
    all_results = {}
    comparison_data = []

    # Run experiments for each scenario
    print(f"\n[3/6] Running experiments for {len(Config.SCENARIOS)} scenarios...")
    print("-" * 80)

    for scenario_id, scenario_config in Config.SCENARIOS.items():
        print(f"\n📊 {scenario_config['name']}")
        print(f"   Description: {scenario_config['description']}")

        # Preprocessing
        print("   • Preprocessing texts...")
        preprocessed_texts = preprocess_texts(texts, scenario_config['preprocessing'])

        # Feature extraction
        print("   • Extracting features...")
        X, vectorizer = fit_transform_features(
            preprocessed_texts,
            method=scenario_config['vectorization']['method'],
            max_features=scenario_config['vectorization']['max_features'],
            ngram_range=tuple(scenario_config['vectorization']['ngram_range'])
        )
        print(f"     Features shape: {X.shape}")

        # Split data
        print("   • Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = split_data(X, labels)
        print(f"     Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        # Train Naive Bayes
        print("   • Training Naive Bayes...")
        nb_model = train_naive_bayes(X_train, y_train,
                                     alpha=scenario_config['models']['naive_bayes']['alpha'])
        nb_metrics, nb_predictions = evaluate_model(nb_model, X_test, y_test)
        print(f"     NB F1-Score: {nb_metrics['f1_score']:.4f}")

        # Train SVM
        print("   • Training SVM...")
        svm_model = train_svm(X_train, y_train,
                             C=scenario_config['models']['svm']['C'],
                             max_iter=scenario_config['models']['svm']['max_iter'])
        svm_metrics, svm_predictions = evaluate_model(svm_model, X_test, y_test)
        print(f"     SVM F1-Score: {svm_metrics['f1_score']:.4f}")

        # Save models
        nb_path = f"models/{scenario_id}_nb.joblib"
        svm_path = f"models/{scenario_id}_svm.joblib"
        vectorizer_path = f"models/vectorizer_{scenario_id}.joblib"

        save_model(nb_model, nb_path)
        save_model(svm_model, svm_path)
        from src.feature_extractor import save_vectorizer
        save_vectorizer(vectorizer, vectorizer_path)

        # Store results
        all_results[scenario_id] = {
            'scenario_name': scenario_id,
            'scenario_display_name': scenario_config['name'],
            'config': scenario_config,
            'naive_bayes': {
                'model': nb_model,
                'metrics': nb_metrics,
                'predictions': nb_predictions,
                'model_path': nb_path
            },
            'svm': {
                'model': svm_model,
                'metrics': svm_metrics,
                'predictions': svm_predictions,
                'model_path': svm_path
            },
            'vectorizer_path': vectorizer_path,
            'X_test': X_test,
            'y_test': y_test
        }

        # Add to comparison data
        comparison_data.append({
            'scenario': scenario_config['name'],
            'scenario_id': scenario_id,
            'algorithm': 'Naive Bayes',
            'algorithm_id': 'naive_bayes',
            'accuracy': round(nb_metrics['accuracy'], 4),
            'precision': round(nb_metrics['precision'], 4),
            'recall': round(nb_metrics['recall'], 4),
            'f1_score': round(nb_metrics['f1_score'], 4),
            'per_class': nb_metrics.get('per_class', {}),
            'confusion_matrix': nb_metrics.get('confusion_matrix', []).tolist() if hasattr(nb_metrics.get('confusion_matrix', []), 'tolist') else [],
            'model_path': nb_path,
            'vectorizer_path': vectorizer_path,
            'config': scenario_config
        })

        comparison_data.append({
            'scenario': scenario_config['name'],
            'scenario_id': scenario_id,
            'algorithm': 'SVM',
            'algorithm_id': 'svm',
            'accuracy': round(svm_metrics['accuracy'], 4),
            'precision': round(svm_metrics['precision'], 4),
            'recall': round(svm_metrics['recall'], 4),
            'f1_score': round(svm_metrics['f1_score'], 4),
            'per_class': svm_metrics.get('per_class', {}),
            'confusion_matrix': svm_metrics.get('confusion_matrix', []).tolist() if hasattr(svm_metrics.get('confusion_matrix', []), 'tolist') else [],
            'model_path': svm_path,
            'vectorizer_path': vectorizer_path,
            'config': scenario_config
        })

    # Select best model
    print(f"\n[4/6] Selecting best model...")
    best_model = select_best_model(comparison_data)
    print(f"✓ Best model: {best_model['scenario']} - {best_model['algorithm']}")
    print(f"  F1-Score: {best_model['f1_score']:.4f}, Accuracy: {best_model['accuracy']:.4f}")

    # Save best model metadata
    save_best_model_metadata(best_model, all_results)

    # Generate visualizations
    print(f"\n[5/6] Generating visualizations...")
    generate_visualizations(comparison_data, all_results)
    print("✓ Visualizations saved to static/results/")

    # Save detailed results
    print(f"\n[6/6] Saving detailed results...")
    save_detailed_results(comparison_data, all_results)
    print("✓ Results saved to logs/experiment_results.json")

    print("\n" + "=" * 80)
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return comparison_data, all_results

def generate_visualizations(comparison_data, all_results):
    """Generate comprehensive visualizations"""

    os.makedirs('static/results', exist_ok=True)

    # 1. Comparison of all metrics
    df_comparison = pd.DataFrame(comparison_data)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison - All Scenarios', fontsize=16, fontweight='bold')

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Prepare data for plotting
        plot_data = []
        labels_plot = []
        for _, row in df_comparison.iterrows():
            plot_data.append(row[metric])
            labels_plot.append(f"{row['scenario_id'].upper()}\n{row['algorithm']}")

        colors = ['#1f77b4' if alg == 'Naive Bayes' else '#ff7f0e'
                  for alg in df_comparison['algorithm'].values]

        bars = ax.bar(range(len(plot_data)), plot_data, color=colors)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(labels_plot, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, plot_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('static/results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ metrics_comparison.png")
    plt.close()

    # 2. F1-Score comparison (primary metric)
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = df_comparison['scenario_id'].unique()
    x_pos = np.arange(len(scenarios))
    width = 0.35

    nb_scores = df_comparison[df_comparison['algorithm'] == 'Naive Bayes']['f1_score'].values
    svm_scores = df_comparison[df_comparison['algorithm'] == 'SVM']['f1_score'].values

    ax.bar(x_pos - width/2, nb_scores, width, label='Naive Bayes', color='#1f77b4')
    ax.bar(x_pos + width/2, svm_scores, width, label='SVM', color='#ff7f0e')

    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('F1-Score Comparison Across Scenarios', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (nb, svm) in enumerate(zip(nb_scores, svm_scores)):
        ax.text(i - width/2, nb, f'{nb:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, svm, f'{svm:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('static/results/f1_score_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ f1_score_comparison.png")
    plt.close()

    # 3. Confusion matrices for top 4 models
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices - Best Models per Scenario', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    # Get best model per scenario
    best_per_scenario = df_comparison.sort_values(['scenario_id', 'f1_score']).groupby('scenario_id').tail(1)

    for idx, (_, model_row) in enumerate(best_per_scenario.iterrows()):
        ax = axes[idx]
        scenario_id = model_row['scenario_id']
        algorithm = model_row['algorithm_id']

        # Get confusion matrix
        cm = np.array(model_row['confusion_matrix'])

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, xticklabels=Config.EMOTION_LABELS,
                   yticklabels=Config.EMOTION_LABELS)
        ax.set_title(f"{scenario_id.upper()} - {model_row['algorithm']}\nF1: {model_row['f1_score']:.4f}",
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)

    plt.tight_layout()
    plt.savefig('static/results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("  ✓ confusion_matrices.png")
    plt.close()

    # 4. Per-class metrics for best model
    best_scenario = all_results[best_per_scenario.iloc[0]['scenario_id']]
    best_algorithm = best_per_scenario.iloc[0]['algorithm_id']
    best_metrics = best_scenario[best_algorithm]['metrics']

    fig, ax = plt.subplots(figsize=(12, 6))

    emotions = list(Config.EMOTION_LABELS)
    precisions = [best_metrics['per_class'].get(e, {}).get('precision', 0) for e in emotions]
    recalls = [best_metrics['per_class'].get(e, {}).get('recall', 0) for e in emotions]
    f1_scores = [best_metrics['per_class'].get(e, {}).get('f1_score', 0) for e in emotions]

    x_pos = np.arange(len(emotions))
    width = 0.25

    ax.bar(x_pos - width, precisions, width, label='Precision', color='#2ca02c')
    ax.bar(x_pos, recalls, width, label='Recall', color='#d62728')
    ax.bar(x_pos + width, f1_scores, width, label='F1-Score', color='#1f77b4')

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Per-Class Metrics - Best Model\n({best_per_scenario.iloc[0]["scenario"]} - {best_per_scenario.iloc[0]["algorithm"]})',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(emotions)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('static/results/per_class_metrics.png', dpi=300, bbox_inches='tight')
    print("  ✓ per_class_metrics.png")
    plt.close()

def save_detailed_results(comparison_data, all_results):
    """Save detailed results to JSON"""

    os.makedirs('logs', exist_ok=True)

    # Prepare results for JSON
    results_to_save = {
        'experiment_timestamp': datetime.now().isoformat(),
        'dataset': {
            'path': 'uploads/Twitter_Emotion_Dataset.csv',
            'total_samples': sum(1 for _ in open('uploads/Twitter_Emotion_Dataset.csv')) - 1,
            'emotions': Config.EMOTION_LABELS,
            'train_test_split': f"{int((1-Config.TEST_SIZE)*100)}/{int(Config.TEST_SIZE*100)}"
        },
        'models': []
    }

    for model_data in comparison_data:
        results_to_save['models'].append({
            'scenario': model_data['scenario'],
            'scenario_id': model_data['scenario_id'],
            'algorithm': model_data['algorithm'],
            'algorithm_id': model_data['algorithm_id'],
            'metrics': {
                'accuracy': model_data['accuracy'],
                'precision': model_data['precision'],
                'recall': model_data['recall'],
                'f1_score': model_data['f1_score']
            },
            'per_class_metrics': model_data['per_class'],
            'model_path': model_data['model_path'],
            'vectorizer_path': model_data['vectorizer_path']
        })

    with open('logs/experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    # Also save as CSV for easier reading
    df_results = pd.DataFrame(comparison_data)
    df_export = df_results[[
        'scenario', 'scenario_id', 'algorithm', 'algorithm_id',
        'accuracy', 'precision', 'recall', 'f1_score'
    ]].copy()
    df_export.to_csv('logs/experiment_results.csv', index=False)

    print("  ✓ experiment_results.json")
    print("  ✓ experiment_results.csv")

if __name__ == "__main__":
    comparison_data, all_results = run_full_experiment()

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY - ALL MODELS")
    print("=" * 80)

    df_summary = pd.DataFrame(comparison_data)
    df_summary = df_summary[[
        'scenario_id', 'algorithm', 'accuracy', 'precision', 'recall', 'f1_score'
    ]].copy()

    print(df_summary.to_string(index=False))

    print("\n" + "=" * 80)
