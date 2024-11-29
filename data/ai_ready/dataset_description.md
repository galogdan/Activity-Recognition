# AI-Ready Dataset Description

## Dataset Overview
- Total Samples: 599
- Training Samples: 479
- Testing Samples: 120
- Selected Features: 20

## Selected Features and Their Importance Scores
- mag_x_wavelet_level0_entropy (Score: 0.9284, Category: wavelet)
- mag_y_wavelet_level0_entropy (Score: 0.6198, Category: wavelet)
- mag_y_dominant_freq (Score: 0.5703, Category: time_domain)
- mag_x_dominant_freq (Score: 0.5589, Category: time_domain)
- accel_x_zero_crossings (Score: 0.5030, Category: time_domain)
- mag_z_wavelet_level0_entropy (Score: 0.4969, Category: wavelet)
- mag_z_dominant_freq (Score: 0.4958, Category: time_domain)
- accel_y_zero_crossings (Score: 0.4610, Category: time_domain)
- gyro_x_dominant_freq (Score: 0.4052, Category: time_domain)
- accel_x_wavelet_level0_entropy (Score: 0.3982, Category: wavelet)
- accel_y_wavelet_level0_entropy (Score: 0.3889, Category: wavelet)
- accel_y_dominant_freq (Score: 0.3764, Category: time_domain)
- gyro_y_wavelet_level0_entropy (Score: 0.3761, Category: wavelet)
- gyro_z_dominant_freq (Score: 0.3690, Category: time_domain)
- accel_y_wavelet_level1_mean (Score: 0.3525, Category: time_domain)
- accel_y_power_medium (Score: 0.3439, Category: power_bands)
- accel_y_wavelet_level3_mean (Score: 0.3404, Category: time_domain)
- mag_z_spectral_centroid (Score: 0.3360, Category: frequency_domain)
- accel_z_wavelet_level0_entropy (Score: 0.3332, Category: wavelet)
- gyro_z_wavelet_level0_entropy (Score: 0.3325, Category: wavelet)

## Features by Category

### Frequency_Domain
- mag_z_spectral_centroid (Score: 0.3360)

### Power_Bands
- accel_y_power_medium (Score: 0.3439)

### Time_Domain
- mag_y_dominant_freq (Score: 0.5703)
- mag_x_dominant_freq (Score: 0.5589)
- accel_x_zero_crossings (Score: 0.5030)
- mag_z_dominant_freq (Score: 0.4958)
- accel_y_zero_crossings (Score: 0.4610)
- gyro_x_dominant_freq (Score: 0.4052)
- accel_y_dominant_freq (Score: 0.3764)
- gyro_z_dominant_freq (Score: 0.3690)
- accel_y_wavelet_level1_mean (Score: 0.3525)
- accel_y_wavelet_level3_mean (Score: 0.3404)

### Wavelet
- mag_x_wavelet_level0_entropy (Score: 0.9284)
- mag_y_wavelet_level0_entropy (Score: 0.6198)
- mag_z_wavelet_level0_entropy (Score: 0.4969)
- accel_x_wavelet_level0_entropy (Score: 0.3982)
- accel_y_wavelet_level0_entropy (Score: 0.3889)
- gyro_y_wavelet_level0_entropy (Score: 0.3761)
- accel_z_wavelet_level0_entropy (Score: 0.3332)
- gyro_z_wavelet_level0_entropy (Score: 0.3325)

## Class Distribution
### Activities
- standing: 158 (33.0%)
- walking: 128 (26.7%)
- running: 107 (22.3%)
- falling: 86 (18.0%)

### Positions
- right_pocket: 187 (39.0%)
- left_pocket: 158 (33.0%)
- hand: 134 (28.0%)
