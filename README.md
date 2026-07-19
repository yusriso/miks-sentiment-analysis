# Miks Sentiment Analysis

Comparing VADER, a trained TF-IDF + Logistic Regression classifier, and LLM-based zero-shot classification on 698 community comments about Miks, a VALORANT agent, to test how much sentiment analysis accuracy you actually lose by using an off-the-shelf tool on slang-heavy gaming text.

**Headline result:**

| Method | Accuracy | Accuracy (excl. Mixed) |
|---|---|---|
| VADER (lexicon baseline) | 36.0% | 39.9% |
| TF-IDF + Logistic Regression (cross-validated) | 56.4% | 62.4% |
| LLM zero-shot classification | 74.6% | 79.3% |

Full write-up with methodology, findings, and limitations: [`miks_sentiment_case_study.md`](./miks_sentiment_case_study.md)
Full analysis with all code and charts: [`miks_sentiment_analysis.ipynb`](./miks_sentiment_analysis.ipynb)

---

## Repo structure

```
├── README.md                              <- you are here
├── miks_sentiment_case_study.md           <- full write-up
├── miks_sentiment_analysis.ipynb          <- main analysis notebook (run this)
│
├── data/
│   ├── combined_comments.csv              <- 698 comments, raw/unscored
│   ├── combined_comments_scored.csv       <- same data + VADER/LogReg predictions
│   ├── gold_set_v2_full_comparison.csv    <- 236 hand-labeled comments w/ all 3 methods' predictions
│   └── threads_metadata.csv               <- original post text per thread (context, not scored)
│
├── scripts/
│   ├── parse_comments.py                  <- Reddit parser, markdown-link username format
│   ├── parse_comments_v2.py               <- Reddit parser, avatar/flair/badge format
│   ├── parse_comments_v3.py               <- Reddit parser, OP-marker/hashtag-flair format
│   ├── parse_youtube_comments.py          <- YouTube @username/timestamp/body/likes format
│   ├── vader_sentiment_scoring.py         <- standalone VADER scorer
│   ├── train_logreg_classifier.py         <- trains + cross-validates the LogReg model
│   ├── score_with_logreg.py               <- applies the trained model to new comments
│   └── evaluate_gold_set.py               <- accuracy/confusion matrix for VADER vs. LLM labels
│
├── models/
│   └── logreg_sentiment_model.joblib      <- trained classifier (TF-IDF + LogReg pipeline)
│
└── charts/
    ├── accuracy_comparison_chart.png
    ├── sentiment_by_platform_chart.png
    └── sentiment_over_time_chart.png
```

---

## Setup

```bash
pip install pandas numpy scikit-learn vaderSentiment joblib matplotlib scipy openpyxl jupyter
```

## How to reproduce

**Easiest:** open `miks_sentiment_analysis.ipynb` and run all cells. It loads `data/combined_comments.csv` and `data/gold_set_v2_full_comparison.csv`, runs all three methods, regenerates every chart, and re-scores the full dataset.

**Or run the pieces individually:**

```bash
# score any CSV with a comment_text column using VADER
python scripts/vader_sentiment_scoring.py data/combined_comments.csv output.csv

# train + cross-validate the LogReg model on the gold set
python scripts/train_logreg_classifier.py data/gold_set_v2_full_comparison.csv

# apply the trained model to new/unlabeled comments
python scripts/score_with_logreg.py data/combined_comments.csv output.csv

# compare VADER vs. LLM-zero-shot labels against the human gold-set labels
python scripts/evaluate_gold_set.py data/gold_set_v2_full_comparison.csv
```

The Reddit/YouTube parsers (`parse_comments*.py`) were used to turn raw copy-pasted comment threads into structured CSVs during data collection. They're included for transparency but aren't needed to reproduce the analysis, the already-parsed output is `data/combined_comments.csv`.

---

## Data sources

698 comments collected from r/ValorantCompetitive, r/VALORANT, VLR.gg forums, and one YouTube video, spanning Miks's launch week through roughly a month later. See the case study for full collection methodology, known limitations, and platform breakdown.

## Methodology notes

- The LogReg model is evaluated with 5-fold stratified cross-validation, not a single train/test split, given how small the labeled gold set is (236 comments).
- The "LLM zero-shot" method is disclosed as not fully independent/blind (see case study, Method 3), it's a ceiling estimate, not a directly comparable deployable baseline.
- Full limitations section is in the case study.

## Tools

Python, pandas, scikit-learn, VADER (vaderSentiment), scipy, matplotlib, openpyxl.
