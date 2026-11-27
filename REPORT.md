# ScamShield Project Report

**Group members:** [Member 1], [Member 2], [Member 3]  
**Course / instructor / date:** [fill in]

## Abstract

ScamShield is a supervised machine-learning prototype that flags SMS and WhatsApp messages as scam or legitimate. We transform message text into TF-IDF features, compare four classifiers on a held-out test set, and deploy the model with the best F1 score in a local browser application. The system is designed as a learning prototype, not a financial-security product.

## 1. Problem statement

Scam messages often create urgency, promise rewards, or ask recipients to visit a link or share payment information. The task is binary classification: `text -> scam / legit`. A useful model needs to catch scams while avoiding excessive warnings on ordinary messages. We therefore report precision, recall, and F1 for the scam class alongside accuracy.

## 2. Data and variables

English SMS examples come from the UCI SMS Spam Collection: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>. We supplement them with 151 group-written Roman-Urdu-style examples based on scam patterns, clearly labelled as synthetic.

| Item | Description |
| --- | --- |
| Input | `text`: a user’s message |
| Target | `label`: `scam` or `legit` |
| Encoded target | scam = 1, legit = 0 |
| Train/test split | created in code with stratified 80/20 split, random state 42: 4,254 / 1,064 messages |

## 3. Data preprocessing

First, `load_data` checks that the single CSV contains `text` and `label`. It removes null/blank text, invalid labels, and duplicate text. Next, `load_and_split` makes a stratified 80/20 training/test partition with `random_state=42`. Labels become numerical binary targets.

`clean_text` lowercases text and preserves important patterns as special tokens: URLs become `urltoken`, phone numbers become `phonetoken`, and numeric strings become `numtoken`. It removes punctuation and normalises selected Roman-Urdu spelling variants. TF-IDF converts one-word and two-word phrases into numbers. Its built-in L2 normalisation means additional feature scaling is not needed. All text transformations are inside each sklearn pipeline, preventing data leakage.

## 4. Exploratory data analysis

Run `python analysis.py` and insert the two generated images below in the submitted PDF:

1. `artifacts/class_distribution.png` shows class imbalance. Legitimate messages are much more common, so accuracy alone could be misleading.
2. `artifacts/message_length.png` compares character counts between labels. Length may help, but vocabulary and short phrase patterns carry most of the classification signal.

The matching `data_quality_summary.csv` records the exact row counts, missing values, duplicates, and mean lengths. This makes the EDA repeatable rather than manually reported.

## 5. Models and evaluation

| Model | Why it was included |
| --- | --- |
| Multinomial Naive Bayes | Fast, conventional baseline for count/TF-IDF text features. |
| Logistic Regression | Interpretable linear baseline that can estimate probabilities. |
| Linear SVM | Strong high-dimensional text classifier; calibration converts its scores to probabilities for the UI. |
| Random Forest | Non-linear tree ensemble comparison, though sparse text often favours linear models. |

All models train only on the training partition and are tested once on the held-out testing partition. Accuracy is the overall correct fraction. Precision asks, “of messages flagged as scams, how many were really scams?” Recall asks, “of all real scams, how many did we catch?” F1 is the harmonic balance of precision and recall. We choose the model with the highest test F1, computed automatically by the code.

Paste the generated `artifacts/model_comparison.csv` table here after your final run. This avoids reporting stale or invented metrics.

## 6. Best-model rationale

The program ranks models by scam-class F1. The top row of `model_comparison.csv` is the deployment model. F1 is more appropriate than accuracy for this imbalanced dataset: a model that calls every message legitimate would appear accurate but would have zero scam recall. Explain the selected model using the terms above and read its actual precision/recall trade-off from the CSV.

For this type of sparse, high-dimensional TF-IDF data, linear models commonly work well because each word or phrase can provide an additive signal. The selected pipeline first converts text to weighted features, then learns a classifier boundary; it does not understand meaning or verify sender identity.

## 7. Web-app integration and demo

`main.py` trains and selects the model once when it starts. The browser page sends the textarea content as JSON to `/predict`. The server caps the request size, predicts a label and calibrated probability, then returns JSON. The page displays Scam in red and Not Scam in green.

Demo sequence: (1) run `python main.py`; (2) open `http://127.0.0.1:8000`; (3) enter a benign message and show the response; (4) enter a suspicious prize/link/payment message; (5) show the terminal’s model-comparison table; (6) state the limitation that it is a prototype.

## 8. Limitations and future work

The Roman-Urdu examples are small and synthetic, labels may contain error, and the English corpus is old. Real scams evolve and may evade keyword patterns. Future work should collect ethically sourced, consented, current multilingual examples; perform cross-validation and threshold tuning; evaluate fairness and false positives; and deploy with monitoring, privacy controls, and human review.
