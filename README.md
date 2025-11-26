# Anti-Money Laundering (AML) Implementation: XGBoost and GNN

This repository contains the implementation and comparative analysis of two machine learning approaches**XGBoost** (Extreme Gradient Boosting) and **Graph Neural Networks** (GNN)to detect money laundering transactions using the IBM Transactions for Anti Money Laundering (AML) dataset.

> **Note:** This project was developed as part of the **Introduction to Data Mining and Knowledge Discovery course (01204465)** at **Kasetsart University**, Academic Year 2025.

##  Project Overview

Money laundering detection is a complex problem characterized by high data volume and extreme class imbalance. This project aims to compare **Feature-based learning**  and **Relation-based learning** (Graph data) to identify suspicious financial activities.

## Repository Structure

```text
Anti-Money-Laundering-Datamining/
├── GNN/                  # Graph Neural Network implementation and experiments
├── XGBoost/              # XGBoost model implementation and experiments
├── requirements.txt      # List of Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── report.pdf            # Project report
```

### Key Objectives:
* Explore and visualize complex financial transaction data.
* Implement **XGBoost** to classify transactions based on tabular features.
* Implement **Graph Neural Networks (GNN)** to classify accounts based on transaction network topology.
* Compare performance metrics, specifically focusing on Recall and Precision due to the imbalanced nature of the data.

##  Dataset

The project uses the **IBM Transactions for Anti Money Laundering (AML)** dataset (Secondary data from Kaggle).

* **Data Size:** 5.07 million transactions.
* **Files Used:**
    * `HI-Small_Trans.csv`: Transaction details (Timestamp, Currency, Amount, etc.).
    * `HI-Small_accounts.csv`: Account details (Bank ID, Entity type).
* **Class Distribution:**
    * **Normal Transactions:** 99.89%.
    * **Money Laundering:** 0.10% (Extremely Imbalanced).

##  Methodology

### 1. Data Preprocessing & Feature Engineering
* **Data Cleaning:** Removed duplicates and filtered out banks with zero history of laundering patterns.
* **Feature Engineering:**
    * Time-based features: `weekday`, `hour`.
    * Transaction patterns: `sender_txn_count`, `receiver_txn_count`, `daily_counts`.
* **Encoding:** OneHotEncoding/BinaryEncoding for categorical data; StandardScaler/MinMax for numerical data.
* **Imbalance Handling:**
    * **XGBoost:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) and Undersampling.
    * **GNN:** Due to hardware (GPU) limitations on the large graph, imbalance handling techniques were limited in the graph experiment.

### 2. Models

#### A. XGBoost (Feature-based)
Utilizes Gradient Boosted Decision Trees to classify individual transactions].
* **Hyperparameters:** `max_depth=16`, `eta=0.1`, `objective='binary:logistic'`.
* **Focus:** Detecting fraud based on individual transaction attributes and aggregated history.

#### B. Graph Neural Network (Relation-based)
Utilizes **Graph Attention Networks (GNN)** via **PyTorch Geometric**.
* **Graph Construction:**
    * **Nodes:** Bank Accounts.
    * **Edges:** Transactions (Directed).
    * **Node Features:** Average amount sent/received, Bank ID.
* **Focus:** Detecting suspicious accounts by learning from the structure of the transaction network and neighbor behaviors.

##  Evaluation Results

### XGBoost Performance
XGBoost served as a strong baseline, identifying individual fraudulent transactions effectively after threshold tuning.

| Metric | Value (Threshold 0.70) |
| :--- | :--- |
| **Accuracy** | 97.0% |
| **Precision** | 3.0% |
| **Recall (TPR)** | **79.0%** |
| **F1-Score** | 0.05 |
*(Data sourced from project report analysis )*

*Observation: High Recall indicates the model catches most fraud cases, but Low Precision results in a higher number of false alarms, which is acceptable for an "Early Warning System." *

### GNN Performance
The GNN model demonstrated high accuracy but struggled with identifying the minority class due to the lack of imbalance handling (e.g., GraphSMOTE) under hardware constraints.

| Metric (Validation) | Value |
| :--- | :--- |
| **Accuracy** | 94.4% |
| **Recall** | Low |
*(Data sourced from project report analysis )*

*Observation: While GNNs are theoretically superior for detecting rings/networks, the extreme class imbalance requires specialized graph-sampling techniques to be effective.*

##  Technologies Used

* **Python**
* **Pandas / NumPy:** Data manipulation.
* **XGBoost:** Gradient Boosting model.
* **PyTorch Geometric:** Graph Neural Network implementation.
* **Scikit-learn:** Preprocessing (SMOTE, LabelEncoder) and Metrics.
* **Matplotlib / Seaborn:** Visualization.

##  Developer

1.Jiraphat Sritawee			   StudentID	6610502005
2.Napatsanan Damaporn		   StudentID	6610502102
3.Jarukit  Phonwattananuwong	StudentID   6610505306



