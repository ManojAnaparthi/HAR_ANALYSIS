
# Human Activity Recognition (HAR) Analysis

This project performs Human Activity Recognition using a modified version of the UCI-HAR dataset and real-world accelerometer data. It includes traditional modeling (Decision Tree) and prompt-based classification using LLaMA3 via LangChain.

---

## Tasks Overview

### Task 1: Exploratory Data Analysis (EDA)

- Loaded sensor CSVs for 6 activity classes
- Plotted time-series waveforms for each axis (`accx`, `accy`, `accz`)
- Computed and visualized `acc_net = sqrt(x² + y² + z²)`
- Visualized data separability using PCA on raw signals

---

### Task 2: Decision Tree Modeling

- Extracted 6 statistical features from each CSV:
  - `mean_x`, `std_x`, `mean_y`, `std_y`, `mean_z`, `std_z`
- Trained a `DecisionTreeClassifier` on the derived features
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

> ⚠️ Before running LLM code, set your GROQ API key:

```bash
export GROQ_API_KEY=your_key_here
```

---

### Task 4: Real-World Data Collection & Testing

- Real-world data collected per activity class in CSVs
- Each file contains 3-axis acceleration (`accx`, `accy`, `accz`)
- Preprocessed to extract same 6 statistical features
- Used previously trained Decision Tree model to classify real-world samples
- Labels inferred from folder names

---

## Notes

- Data structure under `Combined/Train/` and `Combined/Test/` mimics activity-based UCI-HAR splits
- PCA used only for visualization, not for model training
- LLM was tested on UCI-style test samples, not real-world data

---

## Dependencies

- Python ≥ 3.8
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- `langchain`, `groq`

---

## Author

Project based on coursework in Human Activity Recognition and AI interpretability.
