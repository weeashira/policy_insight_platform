# 🏛️ Policy Insight Platform

## 📌 Overview
The Policy Insight Platform processes parliamentary debate transcripts (Hansards) and converts them into structured insights.

Users can explore key discussions through an interactive dashboard, where each debate is summarised and enriched with sentiment analysis and sector classification. Topic modelling is used to group related debates into broader themes.


---

## 🚀 Quick Start (View Dashboard)
To view the dashboard with existing data:

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app.py
```

This will launch the dashboard and display available dates.


---

## 📊 Generate Data for a New Date
This step runs the full pipeline to fetch, process, and analyse parliamentary debates for a given sitting date, generating new insights for the dashboard.

### ⚙️ One-time Setup (Required)

#### 1. Download NLTK data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### 2. Install Ollama

**Windows:**
```powershell
irm https://ollama.com/install.ps1 | iex
```

**macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Alternatively, download from: https://ollama.com/download

#### 3. Pull summarisation model
```bash
ollama pull qwen3.5:9b
```


### ▶️ Run the Pipeline

To process and generate insights for a specific sitting date:

```bash
python backend/run_pipeline.py DD-MM-YYYY
```

**Example:**
```bash
python backend/run_pipeline.py 08-04-2026
```

The output will be generated in:
```bash
data/final/combined_<date>.json
```

After the pipeline completes, the dashboard can be launched using:
```bash
streamlit run frontend/streamlit_app.py
```


---

## 🧠 Model Training

This project includes pre-trained models so training is **not required** to run the platform. The training process can be reproduced using the resources below.


### 📂 Training Data
Training datasets are stored in:

```bash
backend/training/data/
```

- topic_sentiment_classification.ipynb uses both "final_train_weights.csv" and "inference_set.csv"
- topic_modelling_explaination.ipynb uses "speaker_turn_data.csv"
- sector_classification_exploration.ipynb uses "sector_training_data.csv"


### 📓 Training Notebooks
The following notebooks were used to train the models. To reproduce them, the notebooks can be executed, and the resulting models saved for use in the pipeline.

```bash
backend/training/notebooks/
```

- topic_sentiment_classification.ipynb  
- topic_modelling_explaination.ipynb  
- sector_classification_exploration.ipynb  


### 🏗️ Model Outputs
After training, the resulting models should be saved and placed in the following directory:

```bash
backend/models/
```

Organise them as follows:

```bash
backend/models/
├── longformer_sentiment/
├── topic_modelling/
└── sector_classification/
```
