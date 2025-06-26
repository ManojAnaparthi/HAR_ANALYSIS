
# Human Activity Recognition (HAR) Analysis

This project performs Human Activity Recognition using a modified version of the UCI-HAR dataset and real-world accelerometer data. It includes traditional modeling (Decision Tree) and prompt-based classification using LLaMA3 via LangChain.

---

## Tasks Overview

### Task 1: Exploratory Data Analysis (EDA)

- Loaded sensor CSVs for 6 activity classes
- Plotted time-series waveforms for each axis (`accx`, `accy`, `accz`)
- Computed and visualized `acc_net = sqrt(x² + y² + z²)`
- Visualized data separability using PCA on raw signal features, tsfel features and dataset-provided features

---

### Task 2: Decision Tree Modeling

- Trained a `DecisionTreeClassifier` each on raw signal features, tsfel features and dataset-provided features
- Evaluation included:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

### Task 3: Prompt Engineering with LLM (LLaMA3)

- Integrated `langchain_groq` to use LLaMA3 model
- Two modes implemented:
  - **Zero-Shot**: Direct test input → activity label
  - **Few-Shot**: Prompt includes 3 labeled examples before test input
- Compared predictions against true labels:
  - Accuracy
  - Confusion Matrix
  - Classification Report

> ⚠️ Before running LLM code, set your GROQ API key

---

### Task 4: Real-World Data Collection & Testing

- We collected raw accelerometer data from four individuals performing various activities, including walking upstairs, walking downstairs, standing, sitting, and laying
- The data was recorded using the Physics Toolbox Sensor app
- Real-world data collected per activity class in CSV format
- Each file contains 3-axis acceleration (`accx`, `accy`, `accz`)
- Used previously trained Decision Tree model(on raw data) to classify real-world samples
- Labels inferred from folder names

---

## Notes

- PCA used only for visualization, not for model training
- LLM was tested on UCI-style test samples, not real-world data

---

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- `langchain`, `groq`

---

## Author

- **A.V.S Manoj** (23110025) – [manoj.anaparthi@iitgn.ac.in](mailto:manoj.anaparthi@iitgn.ac.in)
