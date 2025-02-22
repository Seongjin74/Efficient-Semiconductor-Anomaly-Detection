# Efficient Semiconductor Anomaly Detection

**Note:** This project has been developed using a Jupyter Notebook (.ipynb) to enhance project visibility.  
For details on the research paper, please refer to the **ESAD_Eng.pdf** file.  
For the full code and implementation details, please check the **ML_Pipeline.ipynb** file.

This repository implements a machine learning pipeline for anomaly detection in semiconductor manufacturing processes using sensor signals. The project is based on the SECOM dataset and integrates various preprocessing and optimization techniques including SMOTE, PCA, hierarchical PCA, statistical feature selection, and hyperparameter tuning via GridSearchCV.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Proposed Method](#proposed-method)
4. [Experiments](#experiments)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Training and Evaluation](#model-training-and-evaluation)
5. [Experimental Evaluation](#experimental-evaluation)
6. [Conclusion and Future Work](#conclusion-and-future-work)
7. [References](#references)
8. [Usage](#usage)
9. [Contact](#contact)

---

## Introduction

Manufacturing processes, especially in semiconductor production, generate large volumes of sensor data that are often incomplete, noisy, and highly imbalanced. This project addresses these challenges by:

- Preprocessing raw sensor data to manage missing values, noise, and multicollinearity.
- Augmenting minority class data using SMOTE to counteract imbalanced data.
- Reducing dimensionality through PCA and hierarchical PCA to extract key features.
- Optimizing model performance using statistical feature selection and hyperparameter tuning via GridSearchCV.

The primary goal is to develop a reliable anomaly detection system that improves quality control in semiconductor manufacturing.

---

## Related Work

Several established methodologies have inspired this project:

- **SMOTE (Synthetic Minority Over-sampling Technique):** Proposed by Chawla et al., SMOTE addresses class imbalance by generating synthetic examples of the minority class.
- **Principal Component Analysis (PCA):** As described by Jolliffe, PCA is used for dimensionality reduction by preserving the most significant information while eliminating noise.
- **Ensemble and Hybrid Methods:** Research by Breiman and others has demonstrated the effectiveness of combining multiple preprocessing and modeling techniques. These methods improve model robustness by mitigating weaknesses inherent to single techniques.
- **Statistical Feature Selection:** Complementing PCA, statistical feature selection is employed to retain only the most relevant features, further enhancing model performance.

---

## Proposed Method

The project introduces an integrated pipeline designed for the SECOM dataset. The main stages of the pipeline include:

1. **Data Preprocessing:**
   - Removal of columns with over 50% missing values.
   - Imputation of missing values using forward and backward fill methods.
   - Elimination of constant features and irrelevant columns (e.g., the 'Time' column).
   - Handling multicollinearity by removing features with high correlation (correlation coefficient > 0.7).
   - Normalization using StandardScaler to ensure all features are on the same scale.

2. **Class Imbalance Resolution:**
   - Application of SMOTE to artificially augment the minority (defective) class.

3. **Dimensionality Reduction and Feature Selection:**
   - Use of PCA to reduce dimensionality and noise.
   - Introduction of hierarchical PCA combined with statistical feature selection to extract and retain key information from the data.

4. **Model Training and Optimization:**
   - Evaluation of six classifiers (Decision Tree, Naive Bayes, Logistic Regression, K-NN, SVM, Neural Network) at different stages.
   - Optimization of classifier hyperparameters using GridSearchCV, leading to the final selection of a hybrid pipeline with an MLPClassifier.

---

## Experiments

### Data Preprocessing

- **Missing Values:** Columns with more than 50% missing values were removed. Remaining missing entries were imputed using sequential fill methods to maintain data continuity.
- **Noise Reduction:** Constant features and highly correlated features (threshold > 0.7) were eliminated to reduce redundancy.
- **Normalization:** StandardScaler was applied to standardize feature scales across the dataset.
- **Data Visualization:** Distribution plots confirmed the severe class imbalance (~93.36% normal vs. ~6.64% defective).

### Model Training and Evaluation

The experiments were conducted in multiple steps:

1. **Step 1:** Training on preprocessed imbalanced data.
   - High accuracy (~82.64%) but poor detection of the minority class (low TPR and F1 Score).
2. **Step 2:** Application of SMOTE.
   - Improved TPR (~36.81%) at the cost of increased false positive rate (FPR).
3. **Step 3:** Incorporation of PCA.
   - Restoration of accuracy (~83.65%) and significant reduction in FPR, albeit with a slight decrease in TPR.
4. **Step 4:** Use of statistical feature selection and hierarchical PCA.
   - Enhanced TPR and F1 Score, with a trade-off of a modest increase in FPR.
5. **Step 5:** Optimization using GridSearchCV.
   - Final model achieved high accuracy (89.81%), a very low FPR (4.14%), with acceptable TPR and improved F1 Score.

---

## Experimental Evaluation

The evaluation of each stage of the pipeline highlights several key points:

- **Overcoming Imbalanced Data:** SMOTE effectively increased the detection rate of defects but required careful handling to avoid a high false positive rate.
- **Dimensionality Reduction:** PCA reduced noise and computational cost, leading to improvements in overall accuracy and model stability.
- **Feature Selection and Hierarchical PCA:** By concentrating on the most meaningful features, this stage further improved minority class detection, although it introduced a slight increase in FPR.
- **Hyperparameter Optimization:** GridSearchCV fine-tuned model parameters, achieving a final model that balances the trade-off between false positives and defect detection.

---

## Conclusion and Future Work

This project demonstrates that an integrated pipeline combining data preprocessing, SMOTE, PCA/hierarchical PCA, statistical feature selection, and model optimization can substantially enhance anomaly detection in semiconductor manufacturing. Key outcomes include:

- A significant reduction in false positives, critical for quality control.
- Improved overall performance metrics such as Accuracy and F1 Score.

**Future Work:**

- Explore alternative imbalance handling techniques such as ADASYN or undersampling.
- Investigate non-linear dimensionality reduction methods (e.g., autoencoders) to preserve complex feature interactions.
- Enhance the feature selection process by incorporating non-linear metrics like Mutual Information.
- Extend the system to handle real-time data streaming and online learning for immediate anomaly detection in production environments.

---

## References

1. McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16*, 321-357.
3. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
4. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
5. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284.
6. Van Hulse, J., Khoshgoftaar, T. M., & Napolitano, A. (2007). Experimental perspectives on learning from imbalanced data. In *Proceedings of the 2007 ACM symposium on Applied computing* (pp. 984-988).
7. Blagus, R., & Lusa, L. (2013). SMOTE for high-dimensional class-imbalanced data. *BMC Bioinformatics, 14*(1), 106.

---

## Usage

To get started with this project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Seongjin74/Efficient-Semiconductor-Anomaly-Detection.git
   cd Efficient-Semiconductor-Anomaly-Detection
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Clone the Repository:**
   ```bash
   python main.py

   ```
   
4. **View Results:

Check the output logs for model performance metrics and visualizations generated during each experiment stage.
