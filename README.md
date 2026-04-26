# 🏛️ Policy Insight Platform

## 📌 Overview

The Policy Insight Platform processes parliamentary debate transcripts (Hansards) and converts them into structured insights.

Users can explore key discussions through an interactive dashboard, where each debate is summarised and enriched with sentiment and sector classification. Topic modelling groups related debates into broader themes.

---

## 🚀 Quick Start (View Dashboard)

To view the dashboard with existing data:

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app.py
```

This will open the dashboard and display the available dates.

---

## 📊 Generate Data for a New Date

### ⚙️ One-time Setup (Required)

#### 1. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### 2. Install Ollama

For Windows users:

```powershell
irm https://ollama.com/install.ps1 | iex
```

For macOS users:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Alternatively, download from: [https://ollama.com/download](https://ollama.com/download)

#### 3. Pull summarisation model

```bash
ollama pull qwen3.5:9b
```

---

### ▶️ Run the Pipeline

To process and generate insights for a specific sitting date:

```bash
python backend/run_pipeline.py DD-MM-YYYY
```

Example:

```bash
python backend/run_pipeline.py 08-04-2026
```

This will generate a new output file in:

```text
data/final/combined_<date>.json
```

After the pipeline completes, launch the dashboard to view the results:

```bash
streamlit run frontend/streamlit_app.py
