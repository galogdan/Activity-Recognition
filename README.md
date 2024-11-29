# Human Activity Recognition System 🏃‍♂️🔍



## 📋 Overview
An advanced Human Activity Recognition (HAR) system that classifies human activities using smartphone sensor data. The system employs innovative signal processing techniques and machine learning algorithms to analyze data from multiple sensor positions, making it highly robust and position-invariant.

### 🎯 Key Features
- Multi-position activity recognition
- Fall detection capability
- Position-invariant classification
- Signal processing pipeline
- Feature engineering

## 🌟 System Components

### 📊 Multi-Sensor Data Collection
- **Accelerometer**: Linear acceleration measurement
- **Gyroscope**: Angular velocity detection
- **Magnetometer**: Orientation tracking

### 🎯 Activities
- Walking
- Running
- Standing
- Falling

### 📱 Sensor Positions
- Right Pocket
- Left Pocket
- Hand

## 🏗️ System Architecture

```
Raw Sensor Data → Preprocessing → Feature Extraction → AI Model → Activity Classification
```

### 📂 Project Structure
```
├── src/
│   ├── eda.py               # Exploratory Data Analysis
│   ├── preprocessor.py      # Signal Processing Pipeline
│   ├── feature_selection.py # Feature Engineering
│   ├── activity_classifier.py # ML Model Implementation
│   └── ai_dataset.py        # Dataset Creation
│
├── data/
│   ├── raw_data/
│   │   ├── walking/
│   │   │   ├── right_pocket/
│   │   │   │   ├── Accelerometer.csv
│   │   │   │   ├── Gyroscope.csv
│   │   │   │   └── Magnetometer.csv
│   │   │   ├── left_pocket/
│   │   │   └── hand/
│   │   ├── running/
│   │   ├── standing/
│   │   └── falling/
│   ├── processed/          # Processed features
│   └── ai_ready/          # Training datasets
│
├── docs/                   # Documentation
└── results/               # Analysis outputs
```

## 🛠️ Technical Implementation

### Signal Processing Pipeline
- **Preprocessing**: 
  - Z-score normalization
  - Low-pass filtering (3Hz cutoff)
  - Outlier removal

- **Window Segmentation**:
  - Falling: 80 samples (0.8s)
  - Standing: 100 samples (1.0s)
  - Walking/Running: 150 samples (1.5s)

### Feature Engineering
- **Time Domain Features**
  - Statistical measures
  - Signal characteristics
  - Zero-crossing rate
  - Peak-to-peak amplitude

- **Frequency Domain Features**
  - Spectral analysis (0.5-20Hz range)
  - Power band analysis
  - Dominant frequency
  - Spectral entropy

- **Wavelet Features**
  - DB4 wavelet decomposition
  - 3-level analysis
  - Wavelet energy
  - Coefficient statistics

### Power Band Analysis
- Low: 0.5-3Hz (postural changes)
- Medium: 3-10Hz (normal motion)
- High: 10-20Hz (rapid movements)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages:
```bash
pip install -r requirements.txt
```

### Quick Start

The system should be run in the following order:

Exploratory Data Analysis (eda.py)

Initial data visualization and analysis
Output: Plots and analysis in results/


Data Preprocessing (preprocessor.py)

Signal processing and feature extraction
Output: Processed features in data/processed/


Feature Selection (feature_selection.py)

Identifies most important features
Output: Feature rankings in results/feature_selection/


AI Dataset Creation (ai_dataset.py)

Creates train/test datasets
Output: Final datasets in data/ai_ready/


Model Training (activity_classifier.py)

Trains and evaluates the model
Output: Model results in results/model/

## 📦 Dependencies
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=0.24.2
- pywavelets>=1.1.1
- scipy>=1.7.0
- matplotlib>=3.4.2
- seaborn>=0.11.1


## 📫 Contact
Created by Gal Ogdan - feel free to contact me!
