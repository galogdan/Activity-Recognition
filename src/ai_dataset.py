import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class AIDatasetCreator:
    """Create the AI-ready dataset."""

    def __init__(self, n_features=20):
        self.project_root = Path(__file__).parent.parent
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.ai_ready_dir = self.project_root / 'data' / 'ai_ready'
        self.feature_scores_path = self.project_root / 'results' / 'feature_selection' / 'feature_scores.csv'
        self.ai_ready_dir.mkdir(parents=True, exist_ok=True)
        self.n_features = n_features

        # Feature categories for documentation
        self.feature_categories = {
            'wavelet': ['wavelet_level'],
            'time_domain': ['zero_crossings', 'dominant_freq'],
            'power_bands': ['power_'],
            'spectral': ['spectral_entropy']
        }

    def get_selected_features(self) -> list:
        """Get top features from pre-calculated scores."""
        if not self.feature_scores_path.exists():
            raise FileNotFoundError(f"Feature scores file not found at {self.feature_scores_path}")

        # Load feature scores
        feature_scores = pd.read_csv(self.feature_scores_path)

        # Sort by Combined_Score and get top features
        return feature_scores.nlargest(self.n_features, 'Combined_Score')['Feature'].tolist()

    def create_dataset(self):
        """Create and save AI-ready dataset."""
        print("Loading data...")
        all_data = []
        for file in self.processed_dir.glob("*_features.csv"):
            df = pd.read_csv(file)
            activity, position = self._parse_filename(file.stem)
            df['activity'] = activity
            df['position'] = position
            all_data.append(df)

        if not all_data:
            raise ValueError("No processed data found!")

        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Total samples loaded: {len(combined_data)}")

        # Get selected features
        selected_features = self.get_selected_features()
        print(f"\nSelected top {len(selected_features)} features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        # Verify all selected features exist
        missing_features = [f for f in selected_features if f not in combined_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")

        # Select features and prepare data
        X = combined_data[selected_features]
        y = combined_data['activity']
        positions = combined_data['position']

        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Create stratified splits
        print("Creating stratified train/test split...")
        X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
            X_scaled, y, positions, test_size=0.2, random_state=42,
            stratify=pd.DataFrame({'activity': y, 'position': positions})
        )

        # Create final datasets
        train_df = pd.concat([
            X_train,
            pd.Series(y_train, name='activity'),
            pd.Series(pos_train, name='position')
        ], axis=1)

        test_df = pd.concat([
            X_test,
            pd.Series(y_test, name='activity'),
            pd.Series(pos_test, name='position')
        ], axis=1)

        # Save datasets
        print("Saving datasets...")
        train_df.to_csv(self.ai_ready_dir / 'train.csv', index=False)
        test_df.to_csv(self.ai_ready_dir / 'test.csv', index=False)

        # Save description
        self._save_description(train_df, test_df, selected_features)
        self._print_dataset_summary(train_df, test_df, selected_features)

    def _save_description(self, train_df: pd.DataFrame, test_df: pd.DataFrame, selected_features: list):
        """Save dataset description with feature importance information."""
        # Load feature scores for detailed information
        feature_scores = pd.read_csv(self.feature_scores_path)

        description = "# AI-Ready Dataset Description\n\n"

        # Overview
        description += "## Dataset Overview\n"
        description += f"- Total Samples: {len(train_df) + len(test_df)}\n"
        description += f"- Training Samples: {len(train_df)}\n"
        description += f"- Testing Samples: {len(test_df)}\n"
        description += f"- Selected Features: {len(selected_features)}\n\n"

        # Feature Details
        description += "## Selected Features and Their Importance Scores\n"
        selected_scores = feature_scores[feature_scores['Feature'].isin(selected_features)]
        for _, row in selected_scores.iterrows():
            description += f"- {row['Feature']} (Score: {row['Combined_Score']:.4f}, Category: {row['Category']})\n"

        # Feature Categories
        description += "\n## Features by Category\n"
        for category in sorted(set(selected_scores['Category'])):
            category_features = selected_scores[selected_scores['Category'] == category]
            if not category_features.empty:
                description += f"\n### {category.title()}\n"
                for _, row in category_features.iterrows():
                    description += f"- {row['Feature']} (Score: {row['Combined_Score']:.4f})\n"

        # Distribution Analysis
        description += "\n## Class Distribution\n"
        description += "### Activities\n"
        activity_dist = train_df['activity'].value_counts()
        for activity, count in activity_dist.items():
            percentage = (count / len(train_df)) * 100
            description += f"- {activity}: {count} ({percentage:.1f}%)\n"

        description += "\n### Positions\n"
        position_dist = train_df['position'].value_counts()
        for position, count in position_dist.items():
            percentage = (count / len(train_df)) * 100
            description += f"- {position}: {count} ({percentage:.1f}%)\n"

        # Save
        with open(self.ai_ready_dir / 'dataset_description.md', 'w', encoding='utf-8') as f:
            f.write(description)

    def _print_dataset_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame, selected_features: list):
        """Print detailed dataset summary."""
        print("\nEnhanced AI-ready Dataset Created Successfully!")
        print("==========================================")

        # Load feature scores for category information
        feature_scores = pd.read_csv(self.feature_scores_path)
        selected_scores = feature_scores[feature_scores['Feature'].isin(selected_features)]

        print(f"\nDataset Size:")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Testing samples: {len(test_df)}")
        print(f"  Selected features: {len(selected_features)}")

        print("\nFeature Categories:")
        for category in sorted(set(selected_scores['Category'])):
            category_count = sum(selected_scores['Category'] == category)
            print(f"  {category.title()}: {category_count} features")

        print("\nTop 5 Most Important Selected Features:")
        top_5 = selected_scores.nlargest(5, 'Combined_Score')
        for _, row in top_5.iterrows():
            print(f"  {row['Feature']} (Score: {row['Combined_Score']:.4f})")

        print("\nClass Distribution (Training Set):")
        print("\nActivities:")
        activity_dist = train_df['activity'].value_counts()
        for activity, count in activity_dist.items():
            percentage = (count / len(train_df)) * 100
            print(f"  {activity}: {count} ({percentage:.1f}%)")

        print("\nPositions:")
        position_dist = train_df['position'].value_counts()
        for position, count in position_dist.items():
            percentage = (count / len(train_df)) * 100
            print(f"  {position}: {count} ({percentage:.1f}%)")

    def _parse_filename(self, filename: str) -> tuple:
        """Parse activity and position from filename."""
        parts = filename.replace('_features', '').split('_')
        activity = parts[0]
        position = '_'.join(parts[1:])
        return activity, position


def main():
    """Main execution function."""
    print("\nCreating AI-ready Dataset with Dynamic Feature Selection...")
    creator = AIDatasetCreator(n_features=20)  # Select top 20 features
    creator.create_dataset()
    print("\nDataset creation complete! Check ai_ready directory for outputs.")


if __name__ == "__main__":
    main()