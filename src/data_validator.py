import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class DataValidator:
    """ Validate and analyze preprocessed data."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.validation_dir = self.project_root / 'results' / 'validation'
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Define feature categories for validation
        self.feature_categories = {
            'wavelet': ['wavelet_level'],
            'time_domain': ['mean', 'std', 'max', 'min', 'rms', 'zero_crossings'],
            'frequency_domain': ['dominant_freq', 'freq_mean', 'freq_energy', 'spectral_entropy', 'spectral_centroid'],
            'power_bands': ['power_low', 'power_medium', 'power_high'],
            'statistical': ['magnitude_mean', 'magnitude_std', 'magnitude_skew', 'magnitude_kurtosis']
        }

    def validate_dataset(self):
        """ Analyze the preprocessed dataset."""
        # Load all processed files
        all_features = []
        for file in self.processed_dir.glob("*_features.csv"):
            df = pd.read_csv(file)
            activity, position = self._parse_filename(file.stem)
            df['activity'] = activity
            df['position'] = position
            all_features.append(df)

        if not all_features:
            print("No processed data found!")
            return

        # Combine all data
        combined_df = pd.concat(all_features, ignore_index=True)

        # Generate validation report
        self._generate_validation_report(combined_df)

    def _parse_filename(self, filename: str) -> tuple:
        """ Parse activity and position from filename."""
        parts = filename.replace('_features', '').split('_')
        activity = parts[0]
        position = '_'.join(parts[1:])
        return activity, position

    def _generate_validation_report(self, df: pd.DataFrame):
        """ Generate comprehensive validation report."""
        print("\nEnhanced Data Validation Report")
        print("============================")

        # 1. Data Distribution
        print("\n1. Data Distribution by Activity and Position:")
        dist = df.groupby(['activity', 'position']).size()
        print(dist)

        # 2. Feature Category Analysis
        self._analyze_feature_categories(df)

        # 3. Analyze Class Balance
        self._analyze_class_balance(df)

        # 4. Validate Wavelet Features
        self._validate_wavelet_features(df)

        # 5. Analyze Sensor Relationships
        self._analyze_sensor_relationships(df)

        # 6. Generate Recommendations
        self._generate_enhanced_recommendations(df)

    def _analyze_feature_categories(self, df: pd.DataFrame):
        """ Analyze features by category and plot distributions."""
        print("\n2. Feature Category Analysis:")

        for category, patterns in self.feature_categories.items():
            category_features = [col for col in df.columns
                                 if any(pattern in col for pattern in patterns)]

            if category_features:
                print(f"\n{category.title()} Features:")
                print(f"- Total features: {len(category_features)}")

                # Print only mean and std for first few features
                sample_features = category_features[:3]
                stats = df[sample_features].agg(['mean', 'std']).round(4)
                print("- Sample feature statistics:")
                print(stats)

    def _validate_wavelet_features(self, df: pd.DataFrame):
        """ Validate wavelet-based features."""
        print("\n3. Wavelet Feature Validation:")

        wavelet_features = [col for col in df.columns if 'wavelet' in col.lower()]
        if not wavelet_features:
            print("No wavelet features found!")
            return

        # Analyze each wavelet level
        for level in range(4):  # 4 levels (0-3)
            level_features = [f for f in wavelet_features if f'level{level}' in f]
            if level_features:
                print(f"\nLevel {level} Features:")
                # Calculate statistics for each level
                stats = df[level_features].agg(['mean', 'std', 'min', 'max']).round(4)
                print(stats)

                # Check for potential issues
                for feature in level_features:
                    std = df[feature].std()
                    if std < 1e-6:
                        print(f"Warning: Low variation in {feature}")

    def _analyze_sensor_relationships(self, df: pd.DataFrame):
        """ Analyze relationships between different sensors."""
        print("\n4. Cross-Sensor Analysis:")

        # Analyze correlations between main sensor features
        sensors = ['accel', 'gyro', 'mag']
        correlation_matrix = np.zeros((len(sensors), len(sensors)))

        for i, sensor1 in enumerate(sensors):
            for j, sensor2 in enumerate(sensors):
                feat1 = f'{sensor1}_magnitude_mean'
                feat2 = f'{sensor2}_magnitude_mean'
                if feat1 in df.columns and feat2 in df.columns:
                    correlation_matrix[i, j] = df[feat1].corr(df[feat2])

        print("\nSensor Correlation Matrix:")
        corr_df = pd.DataFrame(correlation_matrix, index=sensors, columns=sensors)
        print(corr_df.round(4))

        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix,
                    annot=True,
                    cmap='coolwarm',
                    xticklabels=sensors,
                    yticklabels=sensors,
                    vmin=-1,
                    vmax=1)
        plt.title('Sensor Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.validation_dir / 'sensor_correlations.png')
        plt.close()

    def _plot_feature_distributions(self, df: pd.DataFrame):
        """ Plot key feature distributions by activity."""
        # Plot traditional features
        key_features = [
            'accel_x_mean', 'accel_y_mean', 'accel_z_mean',
            'gyro_x_mean', 'gyro_y_mean', 'gyro_z_mean'
        ]

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(key_features, 1):
            if feature in df.columns:
                plt.subplot(2, 3, i)
                sns.boxplot(data=df, x='activity', y=feature)
                plt.xticks(rotation=45)
                plt.title(feature)

        plt.tight_layout()
        plt.savefig(self.validation_dir / 'traditional_feature_distributions.png')
        plt.close()

    def _plot_category_distributions(self, df: pd.DataFrame, features: List[str], category: str):
        """ Plot feature distributions for each category."""
        if not features:
            return

        # Select only top features (e.g., first 6) to avoid too many plots
        selected_features = features[:6]

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(selected_features, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=df, x='activity', y=feature)
            plt.xticks(rotation=45)
            plt.title(feature)

        plt.tight_layout()
        plt.savefig(self.validation_dir / f'{category}_distributions.png')
        plt.close()

    def _analyze_class_balance(self, df: pd.DataFrame):
        """ Analyze class balance in the dataset."""
        print("\n5. Class Balance Analysis:")

        # Activity balance
        activity_dist = df['activity'].value_counts()
        print("\nActivity distribution:")
        print(activity_dist)

        # Calculate percentages
        activity_percentages = (activity_dist / len(df) * 100).round(2)
        print("\nActivity percentages:")
        print(activity_percentages)

        # Position balance
        position_dist = df['position'].value_counts()
        print("\nPosition distribution:")
        print(position_dist)

        # Activity-Position combinations
        combo_dist = df.groupby(['activity', 'position']).size()
        print("\nActivity-Position combinations:")
        print(combo_dist)

    def _generate_enhanced_recommendations(self, df: pd.DataFrame):
        """ Generate enhanced recommendations based on comprehensive analysis."""
        print("\n6. Enhanced Recommendations:")

        # Class balance recommendations
        activity_counts = df['activity'].value_counts()
        min_activity = activity_counts.min()
        max_activity = activity_counts.max()

        if max_activity / min_activity > 1.5:
            print("\nData Balance Issues:")
            print(f"- Data imbalance detected (max/min ratio: {(max_activity / min_activity):.2f})")
            print(f"- Recommended minimum samples per activity: {max_activity}")

        # Feature quality recommendations
        self._check_feature_quality(df)

        # Position coverage recommendations
        self._check_position_coverage(df)

    def _check_feature_quality(self, df: pd.DataFrame):
        """Check quality of features and generate recommendations."""
        print("\nFeature Quality Checks:")

        for category, patterns in self.feature_categories.items():
            category_features = [col for col in df.columns
                                 if any(pattern in col for pattern in patterns)]

            if category_features:
                # Check for low variance features
                low_var_features = [f for f in category_features
                                    if df[f].std() < 1e-6]
                if low_var_features:
                    print(f"- Low variance detected in {category} features:")
                    for feature in low_var_features:
                        print(f"  * {feature}")

    def _check_position_coverage(self, df: pd.DataFrame):
        """Check coverage of sensor positions."""
        print("\nPosition Coverage Analysis:")

        for activity in df['activity'].unique():
            positions = df[df['activity'] == activity]['position'].unique()
            if len(positions) < 3:
                print(f"- Incomplete position coverage for {activity}:")
                missing = set(['right_pocket', 'left_pocket', 'hand']) - set(positions)
                print(f"  * Missing positions: {', '.join(missing)}")


def main():
    validator = DataValidator()
    validator.validate_dataset()


if __name__ == "__main__":
    main()