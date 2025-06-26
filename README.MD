# Human Activity Recognition (HAR) Analysis

This project performs Human Activity Recognition using data from the **UCI-HAR dataset** and **real-world accelerometer data**. The workflow involves Exploratory Data Analysis (EDA), Decision Tree modeling, LLM prompt engineering, and real-world testing with custom data.

---

---

## Tasks Overview

### Task 1: Exploratory Data Analysis (EDA)

- Loaded and visualized UCI HAR dataset
- Analyzed activity distribution
- Plotted sample signals from inertial sensors (X, Y, Z axes)

---

### Task 2: Decision Tree Modeling

- Used `DecisionTreeClassifier` from `scikit-learn`
- Trained using **6 statistical features**:
  - `mean_x`, `std_x`, `mean_y`, `std_y`, `mean_z`, `std_z`
- Derived from raw inertial signals in UCI dataset
- Evaluated on UCI test data (with same feature structure)
- Reported:
  - Accuracy
  - Confusion matrix
  - Classification report

---

### Task 3: Prompt Engineering with LLM (LLaMA3 via LangChain)

- Integrated `langchain_groq` for LLaMA3 LLM
- Designed prompt-based classification in two modes:
  - **Zero-Shot:** Direct prediction from test features
  - **Few-Shot:** Included 3 labeled examples to guide prediction
- Compared LLM predictions with ground truth
- Reported:
  - Accuracy
  - Classification report
  - Confusion matrix

---

### Task 4: Real-World Data Collection & Testing

#### QN-4.1: Use Decision Tree on Real Data

- Real-world CSVs collected per activity
- Each CSV file contains 3-axis accelerometer data
- Applied same statistical featurization (mean/std per axis)
- Used **Decision Tree model trained on UCI data (6 features)** to predict on real data

#### QN-4.2: Preprocessing and Evaluation

- Extracted:
  - `mean_x`, `std_x`, `mean_y`, `std_y`, `mean_z`, `std_z` from each CSV
- Label inferred from folder name

---

**Also set your GROQ API key in environment before using LLM**
