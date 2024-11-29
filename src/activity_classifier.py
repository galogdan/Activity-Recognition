import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


class ActivityClassifier:
    """ Activity classification using best features."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.ai_ready_dir = self.project_root / 'data' / 'ai_ready'  # Changed to use AI-ready data
        self.results_dir = self.project_root / 'results' / 'model'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }

        # Model parameters for grid search
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 4]
            }
        }

        self.best_model = None
        self.scaler = StandardScaler()

    def train_model(self):
        """ Train and evaluate models with cross-validation."""
        print("Loading AI-ready dataset...")
        train_df = pd.read_csv(self.ai_ready_dir / 'train.csv')
        test_df = pd.read_csv(self.ai_ready_dir / 'test.csv')

        # Separate features and targets
        X_train = train_df.drop(['activity', 'position'], axis=1)
        y_train = train_df['activity']
        X_test = test_df.drop(['activity', 'position'], axis=1)
        y_test = test_df['activity']

        # Train and evaluate each model
        results = {}
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                model,
                self.param_grids[model_name],
                cv=StratifiedKFold(n_splits=5),
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Store results
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_score': grid_search.score(X_test, y_test)
            }

            # Save best model
            if self.best_model is None or grid_search.best_score_ > self.best_model['score']:
                self.best_model = {
                    'name': model_name,
                    'model': grid_search.best_estimator_,
                    'score': grid_search.best_score_
                }

        # Evaluate best model
        self._evaluate_best_model(X_test, y_test)
        self._save_results(results)

    def _evaluate_best_model(self, X_test, y_test):
        """ Evaluate the best model thoroughly."""
        model = self.best_model['model']
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Generate reports
        self._generate_classification_report(y_test, y_pred)
        self._generate_confusion_matrix(y_test, y_pred)
        self._plot_feature_importance()

        # Print summary
        print(f"\nBest Model: {self.best_model['name']}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def _generate_classification_report(self, y_test, y_pred):
        """ Generate and save detailed classification report."""
        report = classification_report(y_test, y_pred)
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Model: {self.best_model['name']}\n\n")
            f.write(report)

    def _generate_confusion_matrix(self, y_test, y_pred):
        """ Generate and save confusion matrix plot."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=sorted(set(y_test)),
            yticklabels=sorted(set(y_test))
        )
        plt.title(f'Confusion Matrix - {self.best_model["name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()

    def _plot_feature_importance(self):
        """ Plot feature importance for the best model."""
        if not hasattr(self.best_model['model'], 'feature_importances_'):
            return

        importance = self.best_model['model'].feature_importances_
        features = list(pd.read_csv(self.ai_ready_dir / 'train.csv').drop(['activity', 'position'], axis=1).columns)

        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)

        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Feature Importance - {self.best_model["name"]}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png')
        plt.close()

    def _save_results(self, results):
        """ Save detailed results to file."""
        with open(self.results_dir / 'model_comparison.txt', 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("======================\n\n")
            for model_name, result in results.items():
                f.write(f"{model_name}:\n")
                f.write(f"Best Parameters: {result['best_params']}\n")
                f.write(f"Cross-validation Score: {result['best_score']:.4f}\n")
                f.write(f"Test Score: {result['test_score']:.4f}\n\n")


def main():
    """ Main execution function."""
    print("\nTraining Enhanced Activity Recognition Model...")
    classifier = ActivityClassifier()
    classifier.train_model()
    print("\nTraining complete! Check results directory for outputs.")


if __name__ == "__main__":
    main()