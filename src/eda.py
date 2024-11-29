import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import pywt
from scipy import signal
from scipy import stats
from scipy.fft import fft


class SensorAnalysis:
    """ Main analysis class for EDA."""

    def __init__(self):
        """ Initialize analysis settings."""
        # Get the absolute path to the project root directory
        self.project_root = Path(__file__).parent.parent

        # Set paths relative to project root
        self.data_dir = self.project_root / 'data' / 'raw_data'
        self.output_dir = self.project_root / 'results'

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default plotting style
        plt.style.use('seaborn-v0_8')
        self.setup_plot_style()

        # Print directory structure for verification
        print("\nProject Structure:")
        print(f"Project Root: {self.project_root}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Output Directory: {self.output_dir}")

        # Verify data directory exists
        if not self.data_dir.exists():
            print(f"\nWARNING: Data directory not found at {self.data_dir}")
            print("Creating data directory structure...")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def setup_plot_style(self):
        """ Configure  plot styling."""
        plt.style.use('seaborn-v0_8')

        # Set consistent style parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })

    def verify_data_structure(self):
        """ Verify and print available data structure."""
        print("\nChecking available data:")
        if self.data_dir.exists():
            for activity in self.data_dir.glob("*"):
                if activity.is_dir():
                    print(f"\nActivity: {activity.name}")
                    for position in activity.glob("*"):
                        if position.is_dir():
                            print(f"  Position: {position.name}")
                            sensor_files = list(position.glob("*.csv"))
                            print(f"    Files: {[f.name for f in sensor_files]}")
        else:
            print("No data directory found!")

    def load_data(self, activity: str, position: str, is_second_sample: bool = False) -> Optional[pd.DataFrame]:
        """Load sensor data for given activity and position."""
        try:
            # Base data path
            data_path = self.data_dir / activity / position

            if activity == 'falling':
                # For falling, load from sample1 or sample2 subdirectory
                sample_path = data_path / ('sample2' if is_second_sample else 'sample1')
                if sample_path.exists():
                    print(f"Loading falling sample {2 if is_second_sample else 1} from {sample_path}")
                    return self._load_single_sample(sample_path, activity, position)
                else:
                    print(f"Sample path not found: {sample_path}")
                    return None
            else:
                # For non-falling activities, load directly from position directory
                return self._load_single_sample(data_path, activity, position)

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _load_single_sample(self, data_path: Path, activity: str, position: str) -> pd.DataFrame:
        """Load data from a single sample directory."""
        try:
            # List available files
            available_files = list(data_path.glob("*.csv"))
            print(f"Available files in {data_path.name}: {[f.name for f in available_files]}")

            # Load individual sensor files
            accel_df = pd.read_csv(data_path / 'Accelerometer.csv',
                                   names=['timestamp', 'accel_x', 'accel_y', 'accel_z'],
                                   skiprows=1)
            gyro_df = pd.read_csv(data_path / 'Gyroscope.csv',
                                  names=['timestamp', 'gyro_x', 'gyro_y', 'gyro_z'],
                                  skiprows=1)
            mag_df = pd.read_csv(data_path / 'Magnetometer.csv',
                                 names=['timestamp', 'mag_x', 'mag_y', 'mag_z'],
                                 skiprows=1)

            # Merge and return
            return self.merge_sensor_data(accel_df, gyro_df, mag_df, activity, position)

        except Exception as e:
            print(f"Error loading sample from {data_path}: {str(e)}")
            return None

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
        # Create a copy of the original DataFrame
        df_clean = df.copy()

        # Process numeric columns without timestamp
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_process = [col for col in numeric_cols if col != 'timestamp']

        # Create a combined mask for all columns
        combined_mask = pd.Series(True, index=df.index)

        for col in cols_to_process:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)

            # Update the combined mask
            combined_mask &= (z_scores < threshold)

        # Apply the combined mask once
        return df_clean.loc[combined_mask]

    def add_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Add derived signal features."""
        # Add jerk (derivative of acceleration)
        for sensor in ['accel', 'gyro']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                df[f'{col}_jerk'] = np.gradient(df[col], df['timestamp'])

        return df

    def analyze_activities(self):
        """ Compare patterns across activities."""
        activities = ['walking', 'standing', 'running', 'falling']
        positions = ['right_pocket', 'left_pocket', 'hand']
        total_combinations = len(activities) * len(positions)
        processed = 0

        print("\nStarting Activity Analysis:")
        print(f"Total combinations to process: {total_combinations}")

        all_features = {}
        summary_stats = {
            'activity': [],
            'position': [],
            'samples': [],
            'duration': [],
            'mean_intensity': [],
            'peak_intensity': []
        }

        for activity in activities:
            activity_features = {}
            for position in positions:
                processed += 1
                print(f"\nProcessing combination {processed}/{total_combinations}")
                print(f"Activity: {activity}, Position: {position}")

                try:
                    df = self.load_data(activity, position, 2)
                    if df is not None and self.validate_sensor_data(df):
                        processed_df = self.process_data(df)
                        features = self.analyze_activity_patterns(processed_df, activity, position)
                        activity_features[position] = features

                        # Collect summary statistics
                        summary_stats['activity'].append(activity)
                        summary_stats['position'].append(position)
                        summary_stats['samples'].append(len(df))
                        summary_stats['duration'].append(df['timestamp'].max() - df['timestamp'].min())
                        summary_stats['mean_intensity'].append(processed_df['accel_magnitude'].mean())
                        summary_stats['peak_intensity'].append(processed_df['accel_magnitude'].max())

                except Exception as e:
                    print(f"Error processing {activity} - {position}: {str(e)}")
                    continue

            if activity_features:
                all_features[activity] = activity_features

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        print("\nData Collection Summary:")
        print(summary_df.to_string())

        return all_features, summary_df

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate statistics for all sensors."""
        sensor_cols = [col for col in df.columns if any(
            s in col for s in ['accel', 'gyro', 'mag'])]

        return df[sensor_cols].agg([
            'mean', 'std', 'min', 'max',
            'skew', 'kurtosis'
        ]).round(4)

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """ Analyze sensor patterns and characteristics."""
        patterns = {}

        for sensor in ['accel', 'gyro', 'mag']:
            # Analyze magnitude
            mag_col = f'{sensor}_magnitude'
            patterns[f'{sensor}_stats'] = {
                'peak_value': df[mag_col].max(),
                'mean_value': df[mag_col].mean(),
                'std_value': df[mag_col].std(),
                'rms_value': np.sqrt(np.mean(df[mag_col] ** 2))
            }

            # Zero crossings
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                patterns[f'{col}_zero_crossings'] = np.sum(np.diff(np.signbit(df[col])))

        return patterns

    def calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate correlations between sensor axes."""
        sensor_cols = [col for col in df.columns if any(
            s in col for s in ['accel', 'gyro', 'mag'])]
        return df[sensor_cols].corr()

    def analyze_frequency_domain(self, df: pd.DataFrame) -> Dict:
        """ Perform frequency domain analysis."""
        freq_analysis = {}
        sampling_rate = 100  # Hz

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_{axis}'
                signal_data = df[col].values

                # Compute FFT
                fft_values = np.abs(np.fft.fft(signal_data))
                freqs = np.fft.fftfreq(len(signal_data), 1 / sampling_rate)

                # Get positive frequencies only
                pos_mask = freqs > 0
                freqs = freqs[pos_mask]
                fft_values = fft_values[pos_mask]

                freq_analysis[f'{col}_freq'] = {
                    'dominant_freq': freqs[np.argmax(fft_values)],
                    'mean_freq': np.mean(freqs * fft_values) / np.mean(fft_values),
                    'freq_entropy': stats.entropy(fft_values + 1e-10)
                }

        return freq_analysis

    def plot_activity(self, df: pd.DataFrame, activity: str, position: str) -> None:
        """Create and save all plots for an activity."""
        if df is None:
            return

        # For falling activity, split the data and create separate plots
        if activity == 'falling':
            # Find timestamp gaps that are significantly larger than normal
            timestamp_diffs = df['timestamp'].diff()
            median_diff = timestamp_diffs.median()
            # Consider a gap significant if it's 10 times larger than the median time difference
            large_gaps = timestamp_diffs[timestamp_diffs > 10 * median_diff].index

            if len(large_gaps) > 0:
                # Split into two samples
                sample1 = df.iloc[:large_gaps[0]].copy()
                sample2 = df.iloc[large_gaps[0]:].copy()

                # Process each sample
                for i, sample_df in enumerate([sample1, sample2], 1):
                    # Reset timestamp to start from 0
                    sample_df['timestamp'] = sample_df['timestamp'] - sample_df['timestamp'].min()

                    # Create sample-specific directory
                    plot_dir = self.output_dir / 'plots' / activity / position / f'sample{i}'
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    print(f"Generating plots for {activity} - {position} - Sample {i}")

                    # Generate plots for this sample
                    self.plot_time_series(sample_df, activity, position, plot_dir)
                    self.plot_distributions(sample_df, activity, position, plot_dir)
                    self.plot_correlations(sample_df, activity, position, plot_dir)
                    self.plot_frequency_analysis(sample_df, activity, position, plot_dir)
                    self.plot_wavelet_analysis(sample_df, activity, position, plot_dir)
                    self.plot_sensor_correlations(sample_df, activity, position, plot_dir)
                    self.plot_enhanced_spectral(sample_df, activity, position, plot_dir)
                    self.plot_signal_processing_comparison(sample_df, activity, position, plot_dir)
            else:
                print(f"No significant timestamp gaps found in {activity} - {position} data")
        else:
            # Normal plotting for other activities
            plot_dir = self.output_dir / 'plots' / activity / position
            plot_dir.mkdir(parents=True, exist_ok=True)

            self.plot_time_series(df, activity, position, plot_dir)
            self.plot_distributions(df, activity, position, plot_dir)
            self.plot_correlations(df, activity, position, plot_dir)
            self.plot_frequency_analysis(df, activity, position, plot_dir)
            self.plot_wavelet_analysis(df, activity, position, plot_dir)
            self.plot_sensor_correlations(df, activity, position, plot_dir)
            self.plot_enhanced_spectral(df, activity, position, plot_dir)
            self.plot_signal_processing_comparison(df, activity, position, plot_dir)



    def plot_time_series(self, df: pd.DataFrame, activity: str,
                         position: str, plot_dir: Path) -> None:
        """ Plot time series data for all sensors."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{activity.capitalize()} Activity - {position} Position')

        for i, sensor in enumerate(['accel', 'gyro', 'mag']):
            for axis in ['x', 'y', 'z']:
                axes[i].plot(df['timestamp'], df[f'{sensor}_{axis}'],
                             label=f'{axis}-axis')
            axes[i].set_title(f'{sensor.capitalize()} Data')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.savefig(plot_dir / 'time_series.png')
        plt.close()

    def plot_distributions(self, df: pd.DataFrame, activity: str,
                           position: str, plot_dir: Path) -> None:
        """ Plot distribution of sensor values."""
        for sensor in ['accel', 'gyro', 'mag']:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{sensor.capitalize()} Distributions - {activity} ({position})')

            for i, axis in enumerate(['x', 'y', 'z']):
                col = f'{sensor}_{axis}'
                sns.histplot(data=df, x=col, ax=axes[i], kde=True)
                axes[i].set_title(f'{axis}-axis')

            plt.tight_layout()
            plt.savefig(plot_dir / f'{sensor}_distributions.png')
            plt.close()

    def plot_correlations(self, df: pd.DataFrame, activity: str,
                          position: str, plot_dir: Path) -> None:
        """ Plot correlation matrix."""
        sensor_cols = [col for col in df.columns if any(
            s in col for s in ['accel', 'gyro', 'mag'])]
        corr_matrix = df[sensor_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Sensor Correlations - {activity} ({position})')
        plt.tight_layout()
        plt.savefig(plot_dir / 'correlations.png')
        plt.close()

    def plot_frequency_analysis(self, df: pd.DataFrame, activity: str,
                                position: str, plot_dir: Path) -> None:
        """ Plot frequency domain analysis."""
        sampling_rate = 100  # Hz

        for sensor in ['accel', 'gyro', 'mag']:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{sensor.capitalize()} Frequency Analysis - {activity} ({position})')

            for i, axis in enumerate(['x', 'y', 'z']):
                col = f'{sensor}_{axis}'
                signal_data = df[col].values

                # Compute FFT
                fft_values = np.abs(np.fft.fft(signal_data))
                freqs = np.fft.fftfreq(len(signal_data), 1 / sampling_rate)

                # Plot positive frequencies only
                pos_mask = freqs > 0
                axes[i].plot(freqs[pos_mask], fft_values[pos_mask])
                axes[i].set_title(f'{axis}-axis')
                axes[i].set_xlabel('Frequency (Hz)')
                axes[i].set_ylabel('Magnitude')

            plt.tight_layout()
            plt.savefig(plot_dir / f'{sensor}_frequency.png')
            plt.close()

    def analyze_activity(self, df: pd.DataFrame) -> Dict:
        """ Analyze sensor patterns for an activity."""
        if df is None:
            return {}

        try:
            results = {
                'basic_stats': self.calculate_statistics(df),
                'patterns': self.analyze_patterns(df),
                'correlations': self.calculate_correlations(df),
                'activity_patterns': self.analyze_detailed_patterns(df)  # New analysis
            }
            return results
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {}

    def analyze_detailed_patterns(self, df: pd.DataFrame) -> Dict:
        """ Analyze distinctive patterns for activity recognition."""
        features = {}

        # Magnitude Statistics
        for sensor in ['accel', 'gyro', 'mag']:
            magnitude = np.sqrt(
                df[f'{sensor}_x'] ** 2 +
                df[f'{sensor}_y'] ** 2 +
                df[f'{sensor}_z'] ** 2
            )

            features[f'{sensor}_stats'] = {
                'mean': magnitude.mean(),
                'std': magnitude.std(),
                'max': magnitude.max(),
                'range': magnitude.max() - magnitude.min(),
                'rms': np.sqrt(np.mean(magnitude ** 2))
            }

        # Frequency Domain Features

        for sensor in ['accel', 'gyro', 'mag']:
            for axis in ['x', 'y', 'z']:
                # Convert pandas Series to numpy array before FFT
                signal = df[f'{sensor}_{axis}'].to_numpy()
                # Ensure the signal is float type
                signal = signal.astype(float)
                # Compute FFT
                fft_vals = np.abs(fft(signal))
                features[f'{sensor}_{axis}_freq_power'] = np.sum(fft_vals) / len(fft_vals)

        # Movement Periodicity
        from scipy.signal import find_peaks
        for sensor in ['accel', 'gyro', 'mag']:
            magnitude = np.sqrt(
                df[f'{sensor}_x'] ** 2 +
                df[f'{sensor}_y'] ** 2 +
                df[f'{sensor}_z'] ** 2
            )
            # Convert to numpy array
            magnitude = magnitude.to_numpy()
            peaks, _ = find_peaks(magnitude, distance=20)
            features[f'{sensor}_periodicity'] = len(peaks) / len(magnitude) if len(magnitude) > 0 else 0

        return features

    def plot_wavelet_analysis(self, df: pd.DataFrame, activity: str,
                              position: str, plot_dir: Path) -> None:
        """ Plot wavelet analysis for each sensor."""
        wavelet = 'db4'
        level = 3

        for sensor in ['accel', 'gyro', 'mag']:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'{sensor.capitalize()} Wavelet Analysis - {activity} ({position})')

            for i, axis in enumerate(['x', 'y', 'z']):
                signal = df[f'{sensor}_{axis}'].values

                # Compute wavelet decomposition
                coeffs = pywt.wavedec(signal, wavelet, level=level)

                # Plot original signal
                axes[i, 0].plot(signal)
                axes[i, 0].set_title(f'Original {axis}-axis Signal')
                axes[i, 0].set_ylabel('Amplitude')

                # Plot wavelet coefficients
                for j, coef in enumerate(coeffs):
                    axes[i, 1].plot(coef, label=f'Level {j}')
                axes[i, 1].set_title(f'Wavelet Coefficients {axis}-axis')
                axes[i, 1].legend()

            plt.tight_layout()
            plt.savefig(plot_dir / f'{sensor}_wavelet_analysis.png')
            plt.close()

    def plot_sensor_correlations(self, df: pd.DataFrame, activity: str,
                                 position: str, plot_dir: Path) -> None:
        """ Plot cross-sensor correlation analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Cross-Sensor Correlations - {activity} ({position})')

        # Accelerometer vs Gyroscope
        data_ag = pd.DataFrame({
            'Accelerometer': df['accel_magnitude'],
            'Gyroscope': df['gyro_magnitude']
        })
        sns.scatterplot(data=data_ag, x='Accelerometer', y='Gyroscope', ax=axes[0])
        axes[0].set_title('Accelerometer vs Gyroscope')

        # Accelerometer vs Magnetometer
        data_am = pd.DataFrame({
            'Accelerometer': df['accel_magnitude'],
            'Magnetometer': df['mag_magnitude']
        })
        sns.scatterplot(data=data_am, x='Accelerometer', y='Magnetometer', ax=axes[1])
        axes[1].set_title('Accelerometer vs Magnetometer')

        # Gyroscope vs Magnetometer
        data_gm = pd.DataFrame({
            'Gyroscope': df['gyro_magnitude'],
            'Magnetometer': df['mag_magnitude']
        })
        sns.scatterplot(data=data_gm, x='Gyroscope', y='Magnetometer', ax=axes[2])
        axes[2].set_title('Gyroscope vs Magnetometer')

        plt.tight_layout()
        plt.savefig(plot_dir / 'sensor_correlations.png')
        plt.close()

    def plot_enhanced_spectral(self, df: pd.DataFrame, activity: str, position: str, plot_dir: Path) -> None:
        """Plot enhanced spectral analysis including power bands."""
        sampling_rate = 100  # Hz

        for sensor in ['accel', 'gyro', 'mag']:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'{sensor.capitalize()} Enhanced Spectral Analysis - {activity} ({position})')

            # Define consistent frequency bands
            freq_bands = {
                'low': (0.5, 3),
                'medium': (3, 10),
                'high': (10, 20)
            }

            for i, axis in enumerate(['x', 'y', 'z']):
                signal_data = df[f'{sensor}_{axis}'].values

                # Adapt window size to data length
                nperseg = min(256, len(signal_data))
                if nperseg % 2 != 0:  # Ensure even length
                    nperseg -= 1
                noverlap = nperseg // 2  # Ensure noverlap is less than nperseg

                # Compute PSD with adapted parameters
                freqs, psd = signal.welch(x=signal_data,
                                          fs=sampling_rate,
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          scaling='density',
                                          detrend='constant')

                # Plot PSD with frequency bands highlighted
                axes[i, 0].semilogy(freqs, psd)
                axes[i, 0].set_title(f'{axis}-axis Power Spectral Density')
                axes[i, 0].set_xlabel('Frequency [Hz]')
                axes[i, 0].set_ylabel('PSD [V**2/Hz]')
                axes[i, 0].grid(True)

                # Highlight frequency bands
                for band_name, (low, high) in freq_bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    axes[i, 0].fill_between(freqs[mask], psd[mask], alpha=0.3)

                # Adapt spectrogram parameters
                nperseg_spec = min(128, len(signal_data))
                if nperseg_spec % 2 != 0:
                    nperseg_spec -= 1
                noverlap_spec = nperseg_spec // 2

                # Compute spectrogram with adapted parameters
                try:
                    f, t, Sxx = signal.spectrogram(signal_data,
                                                   fs=sampling_rate,
                                                   nperseg=nperseg_spec,
                                                   noverlap=noverlap_spec,
                                                   scaling='density',
                                                   mode='magnitude')

                    # Plot spectrogram with consistent colormap
                    im = axes[i, 1].pcolormesh(t, f, 10 * np.log10(Sxx),
                                               shading='gouraud',
                                               cmap='viridis',
                                               vmin=-60,
                                               vmax=0)
                    axes[i, 1].set_title(f'{axis}-axis Spectrogram')
                    axes[i, 1].set_xlabel('Time [sec]')
                    axes[i, 1].set_ylabel('Frequency [Hz]')
                except Exception as e:
                    print(f"Warning: Could not generate spectrogram for {sensor}_{axis}: {str(e)}")
                    axes[i, 1].text(0.5, 0.5, 'Insufficient data for spectrogram',
                                    horizontalalignment='center',
                                    verticalalignment='center')

            # Add colorbar (only if spectrograms were generated)
            if 'im' in locals():
                plt.colorbar(im, ax=axes[:, 1].ravel().tolist(),
                             label='Power/Frequency [dB/Hz]')

            # Adjust layout with specific spacing
            plt.subplots_adjust(top=0.92, bottom=0.08,
                                left=0.10, right=0.90,
                                hspace=0.25, wspace=0.35)

            # Save figure
            plt.savefig(plot_dir / f'{sensor}_enhanced_spectral.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    def validate_sensor_data(self, df: pd.DataFrame) -> bool:
        """ Validate loaded sensor data."""
        required_columns = [
            'timestamp',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z'
        ]

        # Check required columns
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing columns: {missing}")
            return False

        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            print("Null values found:")
            print(null_counts[null_counts > 0])
            return False

        # Check sampling rate
        time_diff = df['timestamp'].diff().mean()
        actual_rate = 1 / time_diff if time_diff > 0 else 0
        if not (95 <= actual_rate <= 105):  # Allow 5% deviation from 100Hz
            print(f"Warning: Irregular sampling rate detected: {actual_rate:.2f} Hz")
            return False

        return True

    def analyze_activity_patterns(self, df: pd.DataFrame, activity: str, position: str) -> Dict:
        """ Analyze patterns specific to each activity and position."""
        patterns = {}

        # Basic signal patterns
        for sensor in ['accel', 'gyro', 'mag']:
            # Calculate magnitude patterns
            magnitude = np.sqrt(
                df[f'{sensor}_x'] ** 2 +
                df[f'{sensor}_y'] ** 2 +
                df[f'{sensor}_z'] ** 2
            )

            patterns[f'{sensor}_stats'] = {
                'mean': magnitude.mean(),
                'std': magnitude.std(),
                'max': magnitude.max(),
                'min': magnitude.min(),
                'rms': np.sqrt(np.mean(magnitude ** 2))
            }

            # Calculate periodicity
            peaks, _ = signal.find_peaks(magnitude, distance=20)
            patterns[f'{sensor}_periodicity'] = len(peaks) / len(magnitude) if len(magnitude) > 0 else 0

            # Activity-specific features
            if activity == 'falling':
                # Peak impact detection
                patterns[f'{sensor}_impact'] = magnitude.max()
                patterns[f'{sensor}_impact_time'] = df['timestamp'][magnitude.argmax()]

            elif activity in ['walking', 'running']:
                # Stride analysis
                peaks, _ = signal.find_peaks(magnitude, distance=50)  # Adjust distance based on activity
                if len(peaks) > 1:
                    stride_times = np.diff(df['timestamp'].iloc[peaks])
                    patterns[f'{sensor}_stride_time'] = np.mean(stride_times)
                    patterns[f'{sensor}_stride_regularity'] = np.std(stride_times)

            elif activity == 'standing':
                # Stability analysis
                patterns[f'{sensor}_stability'] = magnitude.std()
                # Calculate sway
                patterns[f'{sensor}_sway'] = np.mean(np.abs(np.diff(magnitude)))

        # Add frequency domain analysis
        freq_features = self.analyze_frequency_domain(df)
        patterns.update(freq_features)

        # Add correlation analysis
        correlation_matrix = self.calculate_correlations(df)
        patterns['sensor_correlations'] = correlation_matrix.values.tolist()

        return patterns

    def plot_signal_processing_comparison(self, df: pd.DataFrame, activity: str, position: str, plot_dir: Path) -> None:
        """ Plot comparison between raw and processed signals."""
        try:
            # Set style for better visualization
            plt.style.use('seaborn-v0_8')

            # Create figure with 2 rows (raw vs processed)
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            fig.suptitle(f'Signal Processing Comparison - {activity.capitalize()} ({position})',
                         fontsize=14, y=1.02)

            # Calculate y-axis limits
            y_min = min(df['accel_x'].min(), df['accel_y'].min(), df['accel_z'].min()) - 2
            y_max = max(df['accel_x'].max(), df['accel_y'].max(), df['accel_z'].max()) + 2

            # Color scheme matching time series
            colors = {'x': '#1f77b4', 'y': '#2ca02c', 'z': '#ff7f0e'}  # Blue, Green, Orange

            # Plot raw data
            axes[0].set_title('Raw Accelerometer Signal', fontsize=12)
            for axis in ['x', 'y', 'z']:
                axes[0].plot(df['timestamp'], df[f'accel_{axis}'],
                             label=f'{axis}-axis',
                             color=colors[axis],
                             linewidth=1.5,
                             alpha=0.9)

            axes[0].set_xlabel('Time (s)', fontsize=10)
            axes[0].set_ylabel('Acceleration (raw)', fontsize=10)
            axes[0].legend(loc='upper right', framealpha=0.9)
            axes[0].grid(True, alpha=0.2)
            axes[0].set_ylim(y_min, y_max)

            # Process the data
            processed_df = self.process_data(df.copy())

            # Plot processed data
            axes[1].set_title('Processed Signal (After Filtering and Outlier Removal)',
                              fontsize=12)

            # Plot axes first
            for axis in ['x', 'y', 'z']:
                axes[1].plot(processed_df['timestamp'], processed_df[f'accel_{axis}'],
                             label=f'{axis}-axis',
                             color=colors[axis],
                             linewidth=1.5,
                             alpha=0.9)

            # Plot magnitude last so it's on top
            magnitude = processed_df['accel_magnitude']
            axes[1].plot(processed_df['timestamp'], magnitude,
                         label='magnitude',
                         color='black',
                         linewidth=2.0,
                         linestyle='--',
                         alpha=1.0)

            axes[1].set_xlabel('Time (s)', fontsize=10)
            axes[1].set_ylabel('Acceleration (processed)', fontsize=10)
            axes[1].legend(loc='upper right', framealpha=0.9)
            axes[1].grid(True, alpha=0.2)
            axes[1].set_ylim(y_min, y_max)

            # Enhance overall appearance
            for ax in axes:
                ax.tick_params(labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # Adjust layout
            plt.tight_layout()

            # Save with high quality
            plt.savefig(plot_dir / 'signal_processing_comparison.png',
                        dpi=300,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none')
            plt.close()

        except Exception as e:
            print(f"Error in plot_signal_processing_comparison: {str(e)}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns}")
            print(f"Sample of data:")
            print(df.head())


def main():
    analysis = SensorAnalysis()
    # analysis.verify_data_structure() Check if the data files exists
    activities = [d.name for d in analysis.data_dir.glob("*") if d.is_dir()]

    if not activities:
        print("\nNo activity directories found!")
        return

    results = {}
    comparative_analysis = {}

    for activity in activities:
        activity_path = analysis.data_dir / activity
        positions = [d.name for d in activity_path.glob("*") if d.is_dir()]

        activity_results = {}
        for position in positions:
            try:
                print(f"\nAnalyzing {activity} - {position}")

                if activity == 'falling':
                    # Process each falling sample
                    for is_second_sample in [False, True]:
                        sample_num = 2 if is_second_sample else 1
                        df = analysis.load_data(activity, position, is_second_sample)

                        if df is not None:
                            # Create sample-specific directory
                            plot_dir = analysis.output_dir / 'plots' / activity / position / f'sample{sample_num}'
                            plot_dir.mkdir(parents=True, exist_ok=True)
                            print(f"Generating plots for falling sample {sample_num}")

                            processed_df = analysis.process_data(df)

                            # Generate plots for this sample
                            analysis.plot_time_series(processed_df, activity, position, plot_dir)
                            analysis.plot_distributions(processed_df, activity, position, plot_dir)
                            analysis.plot_correlations(processed_df, activity, position, plot_dir)
                            analysis.plot_frequency_analysis(processed_df, activity, position, plot_dir)
                            analysis.plot_wavelet_analysis(processed_df, activity, position, plot_dir)
                            analysis.plot_sensor_correlations(processed_df, activity, position, plot_dir)
                            analysis.plot_enhanced_spectral(processed_df, activity, position, plot_dir)
                            analysis.plot_signal_processing_comparison(processed_df, activity, position, plot_dir)

                            # Store results for this sample
                            sample_results = analysis.analyze_activity(processed_df)
                            if sample_results:
                                activity_results[f"{position}_sample{sample_num}"] = sample_results
                else:
                    # Handle non-falling activities
                    df = analysis.load_data(activity, position)
                    if df is not None:
                        plot_dir = analysis.output_dir / 'plots' / activity / position
                        plot_dir.mkdir(parents=True, exist_ok=True)

                        processed_df = analysis.process_data(df)

                        # Generate plots
                        analysis.plot_time_series(processed_df, activity, position, plot_dir)
                        analysis.plot_distributions(processed_df, activity, position, plot_dir)
                        analysis.plot_correlations(processed_df, activity, position, plot_dir)
                        analysis.plot_frequency_analysis(processed_df, activity, position, plot_dir)
                        analysis.plot_wavelet_analysis(processed_df, activity, position, plot_dir)
                        analysis.plot_sensor_correlations(processed_df, activity, position, plot_dir)
                        analysis.plot_enhanced_spectral(processed_df, activity, position, plot_dir)
                        analysis.plot_signal_processing_comparison(processed_df, activity, position, plot_dir)

                        # Store results
                        analysis_results = analysis.analyze_activity(processed_df)
                        if analysis_results:
                            activity_results[position] = analysis_results

                            if 'activity_patterns' in analysis_results:
                                patterns = analysis_results['activity_patterns']
                                if patterns and 'gyro_stats' in patterns:
                                    key_metrics = {
                                        'gyro_intensity': patterns['gyro_stats']['rms'],
                                        'movement_variability': patterns['gyro_stats']['std'],
                                        'periodicity': patterns.get('gyro_periodicity', 0)
                                    }
                                    comparative_analysis[f"{activity}-{position}"] = key_metrics

            except Exception as e:
                print(f"Error processing {activity} - {position}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if activity_results:
            results[activity] = activity_results

    # Print comparative analysis
    print("\nActivity Comparison:")
    print("===================")
    for key, metrics in comparative_analysis.items():
        print(f"\n{key}:")
        print(f"Gyro Intensity: {metrics['gyro_intensity']:.4f}")
        print(f"Movement Variability: {metrics['movement_variability']:.4f}")
        print(f"Periodicity: {metrics['periodicity']:.4f}")

    print("\nAnalysis complete!")
    return results

if __name__ == "__main__":
    results = main()