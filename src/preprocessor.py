import pandas as pd
import numpy as np
import pywt
from scipy.stats import entropy
from scipy.signal import welch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from scipy import stats
from eda import SensorAnalysis  # Import your existing EDA class


class ActivityPreprocessor:
    """ Preprocessor for human activity recognition sensor data.
        This class is mainly to create the processed data for the AI-ready dataset  """

    def __init__(self):
        """ Initialize preprocessor with configuration."""

        self.wavelet_type = 'db4'
        self.wavelet_level = 3
        self.sampling_rate = 100  # Hz
        self.window_size = 256  # samples (2.56 seconds at 100Hz)
        self.overlap = 0.5  # 50% overlap

        # Initialize paths
        self.project_root = Path(__file__).parent.parent
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize EDA for data loading
        self.eda = SensorAnalysis()

        # Add frequency bands for analysis
        self.freq_bands = {
            'low': (0.5, 3),
            'medium': (3, 10),
            'high': (10, 20)
        }

    def _extract_wavelet_features(self, segment: pd.DataFrame) -> Dict:
        """ Extract wavelet-based features from the signal."""
        features = {}

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                data = segment[col].values

                # Perform wavelet decomposition
                coeffs = pywt.wavedec(data, self.wavelet_type, level=self.wavelet_level)

                # Extract features from each decomposition level
                for level, coef in enumerate(coeffs):
                    features.update({
                        f'{col}_wavelet_level{level}_mean': np.mean(np.abs(coef)),
                        f'{col}_wavelet_level{level}_std': np.std(coef),
                        f'{col}_wavelet_level{level}_energy': np.sum(coef ** 2),
                        f'{col}_wavelet_level{level}_entropy': entropy(np.abs(coef))
                    })

        return features

    def _extract_enhanced_frequency_features(self, segment: pd.DataFrame) -> Dict:
        """ Extract enhanced frequency domain features."""
        features = {}

        # Adapt window size to segment length
        nperseg = min(256, len(segment))
        # Ensure nperseg is even
        nperseg = nperseg - 1 if nperseg % 2 != 0 else nperseg

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                data = segment[col].values

                # Calculate power spectral density
                freqs, psd = welch(data, fs=self.sampling_rate,
                                   nperseg=nperseg,
                                   noverlap=nperseg // 2)

                # Calculate band powers
                for band_name, (low, high) in self.freq_bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(psd[mask])
                    features[f'{col}_power_{band_name}'] = band_power

                # Calculate spectral entropy
                spectral_entropy = entropy(psd)
                features[f'{col}_spectral_entropy'] = spectral_entropy

                # Calculate spectral centroid
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                features[f'{col}_spectral_centroid'] = spectral_centroid

        return features

    def extract_features_from_segments(self, segments: List[pd.DataFrame]) -> pd.DataFrame:
        """ Extract features from each segment."""
        all_features = []

        for segment in segments:
            features = {}

            # Original features
            features.update(self._extract_time_domain_features(segment))
            features.update(self._extract_frequency_domain_features(segment))
            features.update(self._extract_statistical_features(segment))

            # New features
            features.update(self._extract_wavelet_features(segment))
            features.update(self._extract_enhanced_frequency_features(segment))

            all_features.append(features)

        return pd.DataFrame(all_features)

    def merge_sensor_data(self, accel_df: pd.DataFrame,
                          gyro_df: pd.DataFrame,
                          mag_df: pd.DataFrame,
                          activity: str,
                          position: str) -> pd.DataFrame:
        """ Merge sensor data and add metadata."""
        # Rename columns to match your actual CSV structure
        accel_df.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z']
        gyro_df.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
        mag_df.columns = ['timestamp', 'mag_x', 'mag_y', 'mag_z']

        # Convert scientific notation to float
        for df in [accel_df, gyro_df, mag_df]:
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = df[col].astype(float)

        # Merge data
        df = pd.merge_asof(accel_df.sort_values('timestamp'),
                           gyro_df.sort_values('timestamp'),
                           on='timestamp',
                           direction='nearest')
        df = pd.merge_asof(df,
                           mag_df.sort_values('timestamp'),
                           on='timestamp',
                           direction='nearest')

        # Add metadata
        df['activity'] = activity
        df['position'] = position

        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Process and clean sensor data."""
        if df is None:
            return None

        # Remove outliers
        df = self.remove_outliers(df)

        # Add magnitude columns
        for sensor in ['accel', 'gyro', 'mag']:
            df[f'{sensor}_magnitude'] = np.sqrt(
                df[f'{sensor}_x'] ** 2 +
                df[f'{sensor}_y'] ** 2 +
                df[f'{sensor}_z'] ** 2
            )

        # Add additional features
        df = self.add_signal_features(df)

        return df

    def remove_outliers(self, df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """ Remove outliers using z-score method."""
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_process = [col for col in numeric_cols if col != 'timestamp']
        combined_mask = pd.Series(True, index=df.index)

        for col in cols_to_process:
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            combined_mask &= (z_scores < threshold)

        return df_clean.loc[combined_mask]

    def add_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Add derived signal features."""
        for sensor in ['accel', 'gyro']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                df[f'{col}_jerk'] = np.gradient(df[col], df['timestamp'])
        return df

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate basic statistics for all sensors."""
        sensor_cols = [col for col in df.columns if any(
            s in col for s in ['accel', 'gyro', 'mag'])]
        return df[sensor_cols].agg([
            'mean', 'std', 'min', 'max',
            'skew', 'kurtosis'
        ]).round(4)

    def process_all_activities(self) -> Dict:
        """Process all available activities."""
        results = {}

        # Using EDA data structure
        activities = [d.name for d in self.eda.data_dir.glob("*") if d.is_dir()]

        for activity in activities:
            activity_results = {}
            for position in ['right_pocket', 'left_pocket', 'hand']:
                try:
                    if activity == 'falling':
                        # Process both falling samples
                        all_features = []
                        for sample_num in [1, 2]:
                            # Load data using existing EDA with sample number
                            df = self.eda.load_data(activity, position, is_second_sample=(sample_num == 2))
                            if df is not None:
                                # Process each sample
                                processed_data = self._process_single_sample(df, activity, position, sample_num)
                                if processed_data is not None:
                                    all_features.append(processed_data['features'])

                        # Combine features from both samples
                        if all_features:
                            combined_features = pd.concat(all_features, ignore_index=True)
                            # Save combined features
                            self.save_processed_data(combined_features, activity, position)
                            activity_results[position] = {
                                'num_segments': sum(len(feat) for feat in all_features),
                                'num_features': combined_features.shape[1],
                                'feature_names': list(combined_features.columns)
                            }
                    else:
                        # Process non-falling activities as before
                        df = self.eda.load_data(activity, position)
                        if df is not None:
                            processed_data = self._process_single_sample(df, activity, position)
                            if processed_data is not None:
                                activity_results[position] = processed_data

                except Exception as e:
                    print(f"Error processing {activity}-{position}: {str(e)}")
                    continue

            if activity_results:
                results[activity] = activity_results

        return results

    def _process_single_sample(self, df: pd.DataFrame, activity: str,
                               position: str, sample_num: Optional[int] = None) -> Optional[Dict]:
        """Process a single data sample."""
        try:
            # 1. Apply filters
            filtered_data = self.apply_filters(df)

            # 2. Segment the data
            segments = self.create_segments(filtered_data)
            if sample_num:
                print(f"Created {len(segments)} segments for {activity} sample {sample_num}")
            else:
                print(f"Created {len(segments)} segments for {activity}")

            # 3. Extract features from segments
            features = self.extract_features_from_segments(segments)

            return {
                'features': features,
                'num_segments': len(segments),
                'num_features': features.shape[1],
                'feature_names': list(features.columns)
            }

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_activity_data(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
                              activity: str, position: str) -> Optional[Dict]:
        """Process data for a single activity."""
        try:
            if activity == 'falling' and isinstance(df, tuple):
                sample1_df, sample2_df = df
                all_features = []

                # Process each sample separately
                for i, sample_df in enumerate([sample1_df, sample2_df], 1):
                    print(f"Processing falling sample {i}")
                    # 1. Apply filters
                    filtered_data = self.apply_filters(sample_df)

                    # 2. Segment the data
                    segments = self.create_segments(filtered_data)
                    print(f"Created {len(segments)} segments for falling sample {i}")

                    # 3. Extract features from segments
                    features = self.extract_features_from_segments(segments)
                    all_features.append(features)

                # Combine features from both samples
                combined_features = pd.concat(all_features, ignore_index=True)

                # 4. Save processed data
                self.save_processed_data(combined_features, activity, position)

                return {
                    'num_segments': sum(len(segments) for segments in all_features),
                    'num_features': combined_features.shape[1],
                    'feature_names': list(combined_features.columns)
                }
            else:
                # Original processing for non-falling activities
                # Apply filters
                filtered_data = self.apply_filters(df)

                # Segment the data
                segments = self.create_segments(filtered_data)
                print(f"Created {len(segments)} segments for {activity}")

                # Extract features from segments
                features = self.extract_features_from_segments(segments)

                # Save processed data
                self.save_processed_data(features, activity, position)

                return {
                    'num_segments': len(segments),
                    'num_features': features.shape[1],
                    'feature_names': list(features.columns)
                }

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply preprocessing filters to the data."""
        df_filtered = df.copy()

        # Apply low-pass filter to remove noise
        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                df_filtered[col] = self.apply_lowpass_filter(df_filtered[col].values)

        # Normalize the signals
        df_filtered = self.normalize_signals(df_filtered)

        return df_filtered

    def apply_lowpass_filter(self, data: np.ndarray,
                             cutoff: float = 3.0) -> np.ndarray:
        """ Apply a low-pass filter to remove high-frequency noise."""
        nyquist = self.sampling_rate * 0.5
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        return signal.filtfilt(b, a, data)

    def normalize_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Normalize sensor signals using z-score normalization."""
        df_norm = df.copy()

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                df_norm[col] = (df[col] - df[col].mean()) / df[col].std()

        return df_norm

    def create_segments(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Create overlapping segments of the data."""
        segments = []

        # Choose window size based on activity
        activity = df['activity'].iloc[0]
        if activity == 'falling':
            window_size = 80  # For falling activity
        elif activity == 'standing':
            window_size = 100  # For standing
        else:
            window_size = 150  # For walking/running

        stride = int(window_size * (1 - self.overlap))

        # For falling activity, find the timestamp gaps to identify separate samples
        if activity == 'falling':
            timestamp_gaps = df['timestamp'].diff()
            sample_breaks = [0] + timestamp_gaps[timestamp_gaps > 1.0].index.tolist() + [len(df)]

            # Process each sample separately
            for i in range(len(sample_breaks) - 1):
                start_idx = sample_breaks[i]
                end_idx = sample_breaks[i + 1]
                sample_df = df.iloc[start_idx:end_idx]

                # Create segments for this sample
                for j in range(0, len(sample_df) - window_size + 1, stride):
                    segment = sample_df.iloc[j:j + window_size].copy()
                    if len(segment) == window_size:
                        segments.append(segment)
        else:
            # Original segmentation for non-falling activities
            for i in range(0, len(df) - window_size + 1, stride):
                segment = df.iloc[i:i + window_size].copy()
                if len(segment) == window_size:
                    segments.append(segment)

        print(f"Created {len(segments)} segments for {activity}")
        return segments


    def _extract_time_domain_features(self, segment: pd.DataFrame) -> Dict:
        """ Extract time domain features."""
        features = {}

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                data = segment[col].values

                features.update({
                    f'{col}_mean': np.mean(data),
                    f'{col}_std': np.std(data),
                    f'{col}_max': np.max(data),
                    f'{col}_min': np.min(data),
                    f'{col}_rms': np.sqrt(np.mean(data ** 2)),
                    f'{col}_zero_crossings': np.sum(np.diff(np.signbit(data)))
                })

        return features

    def _extract_frequency_domain_features(self, segment: pd.DataFrame) -> Dict:
        """ Extract frequency domain features."""
        features = {}

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                data = segment[col].values

                # Apply FFT
                fft_values = np.abs(np.fft.fft(data))
                freqs = np.fft.fftfreq(len(data), 1 / self.sampling_rate)

                # Get positive frequencies
                pos_mask = freqs > 0
                freqs = freqs[pos_mask]
                fft_values = fft_values[pos_mask]

                features.update({
                    f'{col}_dominant_freq': freqs[np.argmax(fft_values)],
                    f'{col}_freq_mean': np.mean(freqs * fft_values) / np.mean(fft_values),
                    f'{col}_freq_energy': np.sum(fft_values ** 2) / len(fft_values)
                })

        return features

    def _extract_statistical_features(self, segment: pd.DataFrame) -> Dict:
        """ Extract statistical features."""
        features = {}

        for sensor in ['accel', 'gyro', 'mag']:
            magnitude = np.sqrt(
                segment[f'{sensor}_x'] ** 2 +
                segment[f'{sensor}_y'] ** 2 +
                segment[f'{sensor}_z'] ** 2
            )

            features.update({
                f'{sensor}_magnitude_mean': np.mean(magnitude),
                f'{sensor}_magnitude_std': np.std(magnitude),
                f'{sensor}_magnitude_skew': stats.skew(magnitude),
                f'{sensor}_magnitude_kurtosis': stats.kurtosis(magnitude)
            })

        return features

    def save_processed_data(self, features: pd.DataFrame,
                            activity: str, position: str) -> None:
        """ Save processed features to disk."""
        output_path = self.processed_dir / f"{activity}_{position}_features.csv"
        features.to_csv(output_path, index=False)
        print(f"Saved processed data to: {output_path}")


def main():
    """ Main execution function with enhanced reporting."""
    preprocessor = ActivityPreprocessor()
    results = preprocessor.process_all_activities()

    print("\nPreprocessing Summary:")
    print("============================")

    # Track feature categories
    feature_categories = {
        'time_domain': ['mean', 'std', 'max', 'min', 'rms', 'zero_crossings'],
        'frequency_domain': ['dominant_freq', 'freq_mean', 'freq_energy', 'spectral_entropy', 'spectral_centroid'],
        'wavelet': ['wavelet_level'],
        'statistical': ['magnitude_mean', 'magnitude_std', 'magnitude_skew', 'magnitude_kurtosis'],
        'power_bands': ['power_low', 'power_medium', 'power_high']
    }

    for activity, positions in results.items():
        print(f"\n{activity.upper()}:")
        for position, data in positions.items():
            print(f"\n  Position: {position}")
            print(f"    Total Segments: {data['num_segments']}")
            print(f"    Total Features: {data['num_features']}")

            # Count features by category
            if 'feature_names' in data:
                print("\n    Feature Breakdown:")
                for category, patterns in feature_categories.items():
                    category_count = sum(1 for feature in data['feature_names']
                                         for pattern in patterns
                                         if pattern in feature)
                    print(f"      - {category.replace('_', ' ').title()}: {category_count}")

            # Sample segment characteristics
            print(f"\n    Window Sizes:")
            if activity == 'falling':
                print(f"      - Using shorter windows (80 samples) for fall detection")
            elif activity == 'standing':
                print(f"      - Using standard windows (100 samples)")
            else:
                print(f"      - Using longer windows (150 samples) for periodic activities")

    print("\nFeature Extraction Complete!")
    return results


if __name__ == "__main__":
    results = main()