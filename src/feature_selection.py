import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureSelector:
    """ Feature selection for human activity recognition dataset."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.results_dir = self.project_root / 'results' / 'feature_selection'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Define feature categories
        self.feature_categories = {
            'time_domain': ['mean', 'std', 'max', 'min', 'rms', 'zero_crossings'],
            'frequency_domain': ['dominant_freq', 'freq_mean', 'freq_energy', 'spectral_entropy', 'spectral_centroid'],
            'wavelet': ['wavelet_level'],
            'statistical': ['magnitude_mean', 'magnitude_std', 'magnitude_skew', 'magnitude_kurtosis'],
            'power_bands': ['power_low', 'power_medium', 'power_high']
        }

    def select_features(self, n_features=50):
        """ Select most important features using multiple methods."""
        # Load and combine all processed data
        all_data = self._load_all_data()
        if all_data is None:
            return None

        # Separate features and labels
        X = all_data.drop(['activity', 'position'], axis=1)
        y = all_data['activity']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # F-Score Selection
        f_selector = SelectKBest(f_classif, k=n_features)
        f_selector.fit(X_scaled, y)
        f_scores = pd.DataFrame({
            'Feature': X.columns,
            'F_Score': f_selector.scores_
        })

        # Mutual Information Selection
        mi_selector = SelectKBest(mutual_info_classif, k=n_features)
        mi_selector.fit(X_scaled, y)
        mi_scores = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_selector.scores_
        })

        # Random Forest Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_scores = pd.DataFrame({
            'Feature': X.columns,
            'RF_Importance': rf.feature_importances_
        })

        # Combine scores
        feature_scores = f_scores.merge(mi_scores, on='Feature')
        feature_scores = feature_scores.merge(rf_scores, on='Feature')

        # Calculate combined score
        feature_scores['Combined_Score'] = (
                                                   feature_scores['F_Score'] / feature_scores['F_Score'].max() +
                                                   feature_scores['MI_Score'] / feature_scores['MI_Score'].max() +
                                                   feature_scores['RF_Importance'] / feature_scores[
                                                       'RF_Importance'].max()
                                           ) / 3

        # Sort by combined score
        feature_scores = feature_scores.sort_values('Combined_Score', ascending=False)

        # Add feature categories
        feature_scores['Category'] = self._categorize_features(feature_scores['Feature'])

        # Generate visualizations
        self._plot_feature_importance(feature_scores)
        self._plot_category_importance(feature_scores)
        self._save_results(feature_scores)

        return feature_scores

    def _categorize_features(self, feature_names):
        """ Categorize features based on their names."""
        categories = []
        for feature in feature_names:
            found_category = 'other'
            for category, patterns in self.feature_categories.items():
                if any(pattern in feature for pattern in patterns):
                    found_category = category
                    break
            categories.append(found_category)
        return categories

    def _plot_feature_importance(self, feature_scores: pd.DataFrame):
        """ Plot detailed feature importance analysis."""
        # Top 20 features plot
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        sns.barplot(
            data=feature_scores.head(20),
            x='Combined_Score',
            y='Feature',
            hue='Category',
            dodge=False
        )
        plt.title('Top 20 Most Important Features')

        # Category distribution in top features
        plt.subplot(2, 1, 2)
        category_counts = feature_scores.head(50)['Category'].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Feature Category Distribution (Top 50 Features)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png')
        plt.close()

    def _plot_category_importance(self, feature_scores: pd.DataFrame):
        """ Plot importance by feature category."""
        plt.figure(figsize=(12, 6))
        category_scores = feature_scores.groupby('Category')['Combined_Score'].mean().sort_values(ascending=False)
        sns.barplot(x=category_scores.index, y=category_scores.values)
        plt.title('Average Importance Score by Feature Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'category_importance.png')
        plt.close()

    def _save_results(self, feature_scores: pd.DataFrame):
        """ Save detailed feature selection results."""
        # Save full results
        feature_scores.to_csv(self.results_dir / 'feature_scores.csv', index=False)

        # Save summary by category
        category_summary = feature_scores.groupby('Category').agg({
            'Combined_Score': ['mean', 'count'],
            'Feature': lambda x: list(x[:5])  # Top 5 features per category
        }).round(4)

        with open(self.results_dir / 'feature_analysis.md', 'w') as f:
            f.write("# Feature Selection Analysis\n\n")
            f.write("## Category Summary\n")
            f.write(category_summary.to_markdown())
            f.write("\n\n## Top 10 Features Overall\n")
            f.write(feature_scores[['Feature', 'Category', 'Combined_Score']].head(10).to_markdown())

    def _load_all_data(self):
        """ Load all processed feature files."""
        all_data = []

        for file in self.processed_dir.glob("*_features.csv"):
            df = pd.read_csv(file)
            activity, position = self._parse_filename(file.stem)
            df['activity'] = activity
            df['position'] = position
            all_data.append(df)

        if not all_data:
            print("No processed data found!")
            return None

        return pd.concat(all_data, ignore_index=True)

    def _parse_filename(self, filename: str) -> tuple:
        """ Parse activity and position from filename."""
        parts = filename.replace('_features', '').split('_')
        activity = parts[0]
        position = '_'.join(parts[1:])
        return activity, position


def main():
    selector = FeatureSelector()
    feature_scores = selector.select_features()

    if feature_scores is not None:
        print("\nFeature Selection Summary:")
        print("========================")
        print("\nTop 10 Features Overall:")
        print(feature_scores[['Feature', 'Category', 'Combined_Score']].head(10))

        print("\nFeature Category Distribution:")
        print(feature_scores['Category'].value_counts())

        # Print category-wise average importance
        print("\nCategory Average Importance:")
        print(feature_scores.groupby('Category')['Combined_Score'].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()