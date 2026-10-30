# PG Finder: Smart PG Review Analyst

Compare paying guest (PG) accommodations in Bengaluru using review text instead of broker claims.

Lakhs of students and freshers arrive in Bengaluru every year and pick a PG in Marathahalli, BTM Layout or Electronic City based on what a broker tells them. Reviews about food, safety, Wi-Fi and owner behaviour are scattered and hard to compare. This app reads the reviews, scores each PG on what matters, and answers questions like *"Which PG is best for a vegetarian girl working in Whitefield?"*

![screenshot](docs/screenshot.png)

## Features

- **Compare PGs.** VADER sentiment (classic NLP) shown next to SLM aspect scores for food, safety, Wi-Fi, cleanliness and owner. A radar chart compares up to 3 PGs.
- **Search reviews.** Semantic search with sentence embeddings and cosine similarity. Type a need like "quiet place with good Wi-Fi" and get the top 5 matching reviews.
- **Find my PG.** Rule-based shortlist by budget, gender, office area and food preference. A small language model then recommends one PG plus a backup and explains why.
- **Bring your own data.** Upload a CSV with your own PG reviews.
- **Transparent prompts.** Every SLM result has an expander showing the exact system and user prompt used.
- **Classic NLP vs SLM.** Both results are shown side by side wherever they apply, so you can compare them.

## Tech stack

| Part | Tool |
| --- | --- |
| UI | Streamlit, Plotly |
| Data | pandas, numpy |
| Classic NLP | NLTK VADER |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Small language model | Qwen2.5-0.5B-Instruct or Qwen2.5-1.5B-Instruct (Hugging Face transformers) |

Everything runs locally. No paid APIs, no API keys, no vector databases.

## Getting started

### Prerequisites

- Python 3.10 or newer
- Git
- About 3 GB of free disk space for model downloads

### Install and run

```bash
git clone https://github.com/abhishekgdangi/pg-finder.git
cd pg-finder

python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (Git Bash):   source .venv/Scripts/activate
# Mac / Linux:          source .venv/bin/activate

pip install torch
pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`.

For an NVIDIA GPU, install the matching torch build from [pytorch.org](https://pytorch.org/get-started/locally/) instead of plain `pip install torch`.

### First run

The models are downloaded from Hugging Face the first time you use them, about 1 to 3 GB in total, then cached on disk. The first SLM click can take a few minutes. A GPU is optional but makes scoring much faster. On CPU, use the 0.5B model and compare one or two PGs at a time.

## How to use

1. **Compare PGs tab.** Pick up to 3 PGs and press *Score selected PGs*. The SLM reads each review and produces aspect scores, shown as a radar chart and a table next to the VADER sentiment.
2. **Search reviews tab.** Describe what you need in plain English and press *Run search*.
3. **Find my PG tab.** Set budget, gender, office location and food preference, then press *Run recommendation*.

### Example inputs

| Tab | Input | What to expect |
| --- | --- | --- |
| Compare PGs | Sri Sai Comforts + Green Nest PG | Sri Sai scores higher on safety, Wi-Fi and cleanliness. Green Nest is weaker on Wi-Fi and safety. |
| Search reviews | `quiet place with good Wi-Fi` | Reviews from TechNest Co-Living, Lakshmi Ladies Residency and Sunrise Premium PG |
| Find my PG | 10000, Female, Whitefield, Vegetarian | Shortlist of Sri Sai Comforts and Lakshmi Ladies Residency, with an SLM pick and a backup |

Exact scores and wording vary with the model.

## Using your own data

Upload a CSV from the sidebar with these columns:

| Column | Example |
| --- | --- |
| `pg_name` | Sri Sai Comforts |
| `area` | Marathahalli |
| `rent` | 9500 |
| `gender` | ladies, gents or unisex |
| `review` | Food is homely and there is always a pure veg option. |

Each row is one review. Rent can be repeated on every row of the same PG. If the file cannot be read, the app falls back to the built-in sample data.

## How it works

```
reviews ──► VADER ───────────────────────► sentiment per PG
   │
   ├──► Qwen SLM (JSON prompt) ──► aspect scores ──► radar chart
   │
   ├──► MiniLM embeddings ──► cosine similarity ──► semantic search
   │
   └──► rule filters ──► shortlist ──► Qwen SLM ──► recommendation + backup
```

- **Classic NLP** uses rules and word lists (VADER), so it is fast but only sees how positive or negative the words are.
- **The SLM** understands meaning and follows instructions, so it can tell that "Wi-Fi keeps disconnecting" is a Wi-Fi problem and explain a choice in plain English.
- SLM output is parsed defensively: JSON is extracted from the reply, retried once if it fails, and every field is type-checked so a bad answer never crashes the app.

## Project structure

```
pg-finder/
├── app.py              Streamlit app (data, NLP functions, UI)
├── requirements.txt    Python packages (torch is installed separately)
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml     Dark theme
```

## Limitations

- The sample PGs, reviews and people are all made up.
- Small models sometimes return unreadable JSON. The 1.5B model is more reliable than the 0.5B one.
- The recommendation step scores at most the 5 best shortlisted PGs to keep it fast.
- The office-to-area mapping is a simple hard-coded rule, not a real commute calculation.

## Roadmap

- [ ] Weighted ranking sliders (set how much you care about food, safety, Wi-Fi)
- [ ] Price vs quality scatter plot
- [ ] Red-flag detector (deposit refusal, hidden charges, hygiene issues)
- [ ] Ask-the-reviews chat using retrieval plus the SLM
- [ ] Evaluation tab comparing VADER and the SLM against hand-labelled reviews
- [ ] Disk cache for SLM scores
- [ ] Export report as CSV or PDF

## License

Add a license of your choice, for example MIT, via **Add file > Create new file** on GitHub and type `LICENSE`.
