#!/usr/bin/env python3
"""
Main entry point for the machine learning project.

This script provides a single command to execute the full ML pipeline:
1. Load data from data/dataset.csv
2. Train a linear regression model
3. Evaluate the model
4. Save the model and evaluation results

Usage:
    python main.py --target-column <target_name>
"""

import argparse
import logging
# import numpy as np
# import sklearn
# import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import pickle
from pathlib import Path

# Add the project root to the Python path to find the 'src' module
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project-specific functions from the 'src' directory
try:
    from src.my_utils import load_data_csv, extract_features, create_target_variable
    from src.model_training_and_evaluation import train_linear_regression, evaluate_model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the 'src' directory exists and contains 'my_utils.py' and 'model_training_and_evaluation.py'.")
    sys.exit(1)


def setup_logging():
    """Configures basic logging to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute the full machine learning pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-filename",
        type=str,
        default="dataset.csv",
        help="The name of the CSV file in the 'data' directory."
    )

    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="The name of the target variable column in the dataset."
    )
    
    parser.add_argument(
        "--feature-column",
        type=str,
        default=None,
        help="Optional: A single feature column to use. If not provided, all other columns are used."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save the trained model and evaluation results."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split."
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for random number generators for reproducibility."
    )
    
    # If no command is provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    return parser.parse_args()


def main():
    """Main function to run the ML pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting the machine learning pipeline...")
    
    # --- 1. Load Data ---
    data_path = project_root / 'data' / args.data_filename
    df = load_data_csv(str(data_path))
    if df is None:
        logger.error("Halting execution due to data loading failure.")
        sys.exit(1)
        
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
        
    # --- 2. Preprocess Data ---
    logger.info(f"Extracting features and target variable...")

    # Determine which features to use
    if args.feature_column:
        # Use only the single specified feature, wrapping it in a list
        feature_columns = [args.feature_column]
        logger.info(f"Using single feature: {args.feature_column}")
    else:
        # Use all columns except the target
        feature_columns = [col for col in df.columns if col != args.target_column]
        logger.info(f"Using all columns except target as features: {feature_columns}")

    X = extract_features(df, feature_columns)
    y = create_target_variable(df, args.target_column)
    
    # --- 3. Split Data ---
    from sklearn.model_selection import train_test_split
    logger.info(f"Splitting data into training and testing sets (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- 4. Train the Model ---
    logger.info("Training the linear regression model...")
    try:
        model = train_linear_regression(X_train, y_train)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)

    # --- 5. Evaluate the Model ---
    logger.info("Evaluating the model on the test set...")
    try:
        metrics = evaluate_model(model, X_test, y_test)
        logger.info("Model evaluation completed.")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        sys.exit(1)

    # --- 6. Save Model and Results ---
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}...")
    
    # Save the trained model
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_model.joblib"
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Trained model saved to {model_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Save the prediction plot
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "prediction_plot.png"
    predictions = model.predict(X_test)
    try:
        with open(figure_path, 'wb'):
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [predictions.min(), predictions.max()], '--r', linewidth=2)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('True Values vs. Predicted Values')
            plt.savefig(figure_path)
        logger.info(f"Prediction plot saved to {figure_dir}")
    except Exception as e:
        logger.error(f"Failed to save prediction plot: {e}")

    # Save the evaluation metrics to a text file
    results_path = output_dir / "evaluation_results.txt"
    try:
        with open(results_path, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("-------------------------\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        logger.info(f"Evaluation results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        
    logger.info("Machine learning pipeline finished successfully.")


if __name__ == "__main__":
    main()

