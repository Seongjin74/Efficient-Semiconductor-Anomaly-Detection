# Anomaly Detection in Modern Manufacturing Processes

In modern manufacturing processes, data of limited quantity and varying quality are mixed, and the importance of analysis and prediction that effectively utilizes these data is growing. In particular, data in manufacturing processes include vast amounts of information collected from sensors along with various issues such as missing values, noise, and multicollinearity, and the preprocessing and transformation of data have a profound impact on the performance of the final predictive model. Accordingly, how efficiently the given data are preprocessed and which appropriate analytical techniques are applied play a key role in product quality management and anomaly detection.

Semiconductor manufacturing processes are especially noteworthy from this perspective. In semiconductor manufacturing, it is essential to manage product quality and detect defects and abnormal signals early based on numerous sensor data. However, sensor data have several limitations, such as class imbalance, high dimensionality, missing values and noise, and multicollinearity; if these data are used as they are, the model’s performance may degrade and interpretation may become difficult.

Existing studies have proposed methods to address these issues by supplementing the minority class data through oversampling techniques such as SMOTE or by introducing dimensionality reduction techniques such as PCA to remove noise while preserving essential information. In addition, there are attempts to improve predictive performance by combining various preprocessing techniques and classifiers through hybrid and ensemble methods.

This study synthesizes the aforementioned previous research and aims to improve a machine learning model for anomaly detection using signal data. Specifically, the study applies preprocessing techniques such as missing value imputation, removal of constant values and noise, and alleviation of multicollinearity to the secom dataset and addresses class imbalance issues through SMOTE. Then, after enhancing data efficiency via hierarchical PCA and statistics-based feature selection, six classifiers are individually trained and evaluated to check the basic performance, followed by an optimal classifier selection process using GridSearchCV.

This approach is expected to maximize the efficiency of data utilization in quality management and anomaly detection in manufacturing processes, and contribute to the development of more reliable predictive models. In this study, we compare and analyze the performance at each stage of the proposed integrated pipeline—from Stage 0 (raw data) to Stage 1 (SMOTE applied), Stage 2 (SMOTE + PCA), and Stage 3 (SMOTE + statistics-based feature selection + PCA + GridSearchCV)—to comprehensively evaluate the impact of preprocessing and optimization on the final model performance.

The SECOM dataset is semiconductor manufacturing process data provided on November 18, 2008, consisting of 1,567 examples and 591 features. Each example contains sensor signals corresponding to a production unit along with a simple Pass/Fail label (including 104 Fail cases). This dataset includes various signals collected from actual processes, making meaningful feature selection and noise removal important.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
   - [Missing Value Treatment](#missing-value-treatment)
   - [Removal of Constant Values and Noise](#removal-of-constant-values-and-noise)
   - [Removal of Time Data and Multicollinearity](#removal-of-time-data-and-multicollinearity)
   - [Normalization](#normalization)
   - [Visualization of Imbalanced Distribution](#visualization-of-imbalanced-distribution)
3. [Model Training and Evaluation](#model-training-and-evaluation)
   - [Step 1: Original Data (Imbalanced)](#step-1-original-data-imbalanced)
   - [Step 2: Oversampling Using SMOTE](#step-2-oversampling-using-smote)
   - [Step 3: SMOTE + PCA (Dimensionality Reduction)](#step-3-smote--pca-dimensionality-reduction)
   - [Step 4: SMOTE + Statistics-Based Feature Selection + Hierarchical PCA + GridSearchCV](#step-4-smote--statistics-based-feature-selection--hierarchical-pca--gridsearchcv)
4. [Final Evaluation and Comparative Analysis](#final-evaluation-and-comparative-analysis)
5. [Conclusions and Future Research Directions](#conclusions-and-future-research-directions)
6. [References](#references)

---

## Introduction

In modern manufacturing processes, sensor data are often fraught with issues such as missing values, noise, and multicollinearity. Preprocessing these data and applying the right analytical techniques are essential steps for achieving high performance in predictive models used for product quality management and anomaly detection. Semiconductor manufacturing, in particular, relies heavily on numerous sensor data to manage quality and detect defects early, though challenges such as class imbalance and high dimensionality make this task nontrivial.

---

## Data Preprocessing

Data preprocessing is a critical stage that directly influences model performance. Below are the techniques applied:

### Missing Value Treatment

- **Problem Identification:**  
  Missing values frequently occur in manufacturing process data during sensor measurements and can act as noise during analysis and model training.

- **Treatment Method:**  
  - Remove columns containing more than 50% missing values.  
  - For the remaining missing values, apply forward fill and backward fill methods, considering the order of the data to maintain continuity.

- **Improvement Effects and Limitations:**  
  - Reduces unnecessary noise by eliminating features with excessive missing values.  
  - Minimizes data loss with appropriate imputation, though the sequential fill method may be less effective when data lack a clear temporal structure.

### Removal of Constant Values and Noise

- **Problem Identification:**  
  Columns that have the same value across all samples provide no distinguishing information and unnecessarily increase computational load.

- **Treatment Method:**  
  Remove columns that contain constant values across all rows.

- **Improvement Effects and Limitations:**  
  Enhances model training efficiency and reduces computational costs, though rarely some constant features might hold important information.

### Removal of Time Data and Multicollinearity

- **Time Data Removal:**  
  Remove the 'Time' column, as it is not directly related to the analysis, to reduce dimensions and avoid introducing unnecessary noise.

- **Multicollinearity Removal:**  
  Identify features with a correlation coefficient of 0.7 or above and remove redundant features by keeping only one representative feature.

- **Improvement Effects and Limitations:**  
  Enhances model interpretability and reduces the risk of overfitting, but setting the correlation threshold requires careful consideration to avoid losing meaningful information.

### Normalization

- **Problem Identification:**  
  Different feature scales can cause certain features to overly influence distance-based or probability-based algorithms.

- **Treatment Method:**  
  Use StandardScaler to normalize all features, transforming them to have a mean of 0 and a standard deviation of 1.

- **Improvement Effects and Limitations:**  
  Ensures each feature is compared on the same scale, improving model stability and convergence speed, although normalization might have minimal effect on some tree-based algorithms.

### Visualization of Imbalanced Distribution

- **Problem Identification:**  
  The dataset is highly imbalanced (approximately 93.36% normal vs. 6.64% defective), risking bias toward the majority class.

- **Treatment Method:**  
  Visualize the target variable distribution to clearly understand the imbalance and motivate the use of oversampling techniques such as SMOTE.

- **Improvement Effects and Limitations:**  
  Helps in early detection of imbalance, though visualization alone does not resolve the issue and requires subsequent corrective measures.

---

## Model Training and Evaluation

After preprocessing, the dataset is split into training and test sets. Six classifiers (Decision Tree, Naive Bayes, Logistic Regression, K-NN, SVM, and Neural Network) are evaluated at each stage.

### Step 1: Original Data (Imbalanced)

- **Method Description:**  
  Evaluate the classifiers using the preprocessed original data without addressing class imbalance.

- **Evaluation Metrics:**  
  Accuracy, TPR (True Positive Rate), and FPR (False Positive Rate).

- **Observations and Limitations:**  
  - Accuracy appears high, but there is a bias toward the majority class.  
  - TPR is low (around 20%), indicating insufficient detection of defects.

### Step 2: Oversampling Using SMOTE

- **Purpose:**  
  Address class imbalance by artificially augmenting the minority class using SMOTE.

- **Improvement Effects:**  
  - Increased TPR for minority class detection (rising to approximately 37%).  
  - However, FPR increases (around 27%), causing overall Accuracy to drop to the 70% range.

- **Observations and Limitations:**  
  Some models, such as K-NN, may show extremely high TPR but also an excessive FPR.

### Step 3: SMOTE + PCA (Dimensionality Reduction)

- **Purpose:**  
  Apply PCA to the SMOTE-augmented data to reduce dimensionality and noise, addressing the “curse of dimensionality.”

- **Improvement Effects:**  
  - Reduces the feature set (e.g., from 204 to 50 components) and lowers noise.  
  - Accuracy recovers to approximately 83.65% with FPR reduced to 11.38%.  
  - TPR decreases slightly compared to the SMOTE-only stage, but the overall balance between precision and recall improves, leading to a higher F1 Score (approximately 0.5428).

- **Observations and Limitations:**  
  Some important information might be lost during dimensionality reduction, and performance improvements vary across models.

### Step 4: SMOTE + Statistics-Based Feature Selection + Hierarchical PCA + GridSearchCV

- **Purpose:**  
  Combine statistics-based feature selection with hierarchical PCA for more effective feature extraction and dimensionality reduction. Optimize classifier selection using GridSearchCV.

- **Hybrid Feature Selection:**  
  - Calculate Pearson correlation coefficients between features and the target variable.  
  - Apply VarianceThreshold to remove features with low correlation or variance, retaining only the most significant features.

- **Hierarchical PCA:**  
  - Cluster features and apply PCA within each cluster to effectively capture the most important information.

- **Optimization Using GridSearchCV:**  
  Perform cross-validation for multiple classifiers and parameters to select the optimal model.

- **Improvement Effects and Limitations:**  
  - Overall predictive performance (Accuracy, TPR, FPR, and F1 Score) is enhanced compared to using simple preprocessing or single techniques alone.  
  - The increased computational cost and complexity of parameter optimization are notable challenges.

---

## Final Evaluation and Comparative Analysis

- **Step 1 (Original Data):**  
  - High Accuracy but low TPR and F1 Score due to class imbalance.
  
- **Step 2 (SMOTE Applied):**  
  - Improved TPR through data augmentation, but at the cost of increased FPR and lower overall Accuracy.
  
- **Step 3 (SMOTE + PCA):**  
  - Dimensionality reduction reduced noise and improved Accuracy and F1 Score, though TPR remained slightly low.
  
- **Step 4 (Final Hybrid Model):**  
  - The integrated pipeline (SMOTE + statistics-based feature selection + hierarchical PCA + GridSearchCV) achieved high Accuracy (89.81%) and extremely low FPR (4.14%), with a stable F1 Score (0.5728), making it well-suited for quality management in manufacturing processes.

---

## Conclusions and Future Research Directions

- **Key Achievements:**  
  - Improved data quality through preprocessing (missing value treatment, removal of constant/multicollinear features, normalization).  
  - Addressed class imbalance using SMOTE.  
  - Enhanced model performance through PCA and hierarchical PCA for dimensionality reduction and noise removal.  
  - Achieved further performance optimization using statistics-based feature selection and GridSearchCV.

- **Future Research Directions:**  
  - Compare performance with deep neural networks or deep learning models.  
  - Explore applications in real-time data streaming environments.  
  - Investigate further combinations of various feature selection and dimensionality reduction techniques for more refined anomaly detection models.

---

## References

[1] McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305.

