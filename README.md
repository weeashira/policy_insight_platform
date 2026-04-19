# Topic Sector Classification Module

This module implements the Topic Sector Classification component for the Policy Insight Platform.

## Objective
To classify each parliamentary debate into the most relevant policy sector using:
- Debate Title
- Speaker Turn 1
- Speaker Turn 2

## Method
We use a supervised text classification pipeline:
- TF-IDF vectorisation
- Logistic Regression classifier

Two versions are evaluated:
1. Unweighted baseline model
2. Confidence-weighted model using annotation confidence (HIGH / MEDIUM / LOW)

## Input
Place the labelled training dataset at:

data/sector_training_data.csv

The CSV should contain the following columns:
- debate_id
- title
- turn1
- turn2
- sector_label
- confidence
- combined_text

## How to Run
Install dependencies:

pip install -r requirements.txt

Run the pipeline:

python src/sector_classification_pipeline.py

## Outputs
The script will generate the following files under `outputs/`:
- sector_distribution.png
- confusion_matrix_cv_unweighted.png
- confusion_matrix_cv_weighted.png
- metrics_summary.csv
- classification_report_unweighted.csv
- classification_report_weighted.csv
- top_keywords_unweighted.txt
- error_analysis_unweighted.csv
- error_pairs_unweighted.csv

## Main Model
The unweighted TF-IDF + Logistic Regression model is used as the main model because it achieved better overall performance.