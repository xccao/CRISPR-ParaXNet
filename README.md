# CRISPR-ParaXNet

Overview
------------
CRISPR-ParaXNet is a novel parallel deep learning architecture tailored for predicting CRISPR/Cas9 off-target cleavage activities. With an emphasis on both high predictive accuracy and uncertainty estimation, it provides a trustworthy and computationally efficient solution to a key bottleneck in CRISPR genome editing: off-target effect assessment.

This repository provides the source code, models, and benchmarking pipelines for reproducibility and extension.

Key Features & Contributions
------------
We think CRISPR-ParaXNet represents a step forward — not just incrementally, but in architectural philosophy.

**Parallel Neural Network Architecture**
Efficiently processes gRNA–DNA pairs to achieve state-of-the-art performance on multiple evaluation metrics (Pearson, Spearman, MAE, MAPE, Gini). Its lightweight design also offers a lower computational footprint, enabling broader usability.

**Best-in-Class Performance**
Outperforms all current baselines on ranking tasks, especially for high-cleavage activity regions (top 1% and 5%), which are often of greatest concern in therapeutic settings. Maintains stability even on imbalanced datasets — a common issue in genomic prediction tasks.

**Uncertainty-Aware Predictions with Dropout as Bayesian Approximation (DBA)**
Empowers downstream decisions with transparent confidence scores, enabling risk-aware interpretations — essential for clinical and experimental genomics.

PREREQUISITE
------------
The off-target prediction models were conducted using Python 3.8.18, Keras 2.13.1, and TensorFlow v2.13.0. The following Python packages should be installed:

scipy 1.10.1

numpy 1.24.3

pandas 1.5.3

scikit-learn > 1.1.3

TensorFlow > 2.13.0

EXAMPLE USAGE
------------
An example of CRISPR-ParaXNet can be run as: 

```
python CRISPR-ParaXNet.py crisprSQL.csv
```
**Make sure to place both the .py and .csv files in the same folder; otherwise, you will encounter a 'file not found' error.**

CONTAINS
------------
models/CRISPR-ParaXNet.py : Python script for running CRISPR-ParaXNet to predict off-target activities involving indels and mismatches. 

data/crisprSQL.csv : A clean dataset that can be directly accessed by the script. 
