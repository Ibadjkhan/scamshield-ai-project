# ScamShield — AI Project Prototype

ScamShield classifies an SMS or WhatsApp message as **Scam** or **Not Scam**. It supports English plus a small Roman-Urdu extension, compares four machine-learning models, and exposes the selected model through a simple local web app.

## 1. Problem statement

People receive misleading prize, payment, wallet, and link messages. Manually checking every message is slow and inconsistent. The project is a binary text-classification prototype: given a message, predict `scam` (positive class) or `legit` (negative class) and show a calibrated scam probability. It is decision support only, not a replacement for verifying senders or financial-safety advice.

## 2. Dataset

The English messages originate from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The group added 151 short Roman-Urdu examples modelled on common wallet-scam phrasing; they are synthetic and must not be presented as real victim messages.

The single `data.csv` file contains 5,318 rows. At runtime the code makes a stratified 80/20 split using `random_state=42`: 4,254 training and 1,064 testing messages. `text` is the raw message input. `label` is the target, either `scam` or `legit`; code encodes these as 1 and 0.

## 3. Install and run

```powershell
cd "C:\Users\WIN11-PRO\OneDrive\Documents\ChatGPT\ai lab project"
python -m pip install -r requirements.txt
python analysis.py       # creates EDA charts and comparison CSV in artifacts/
python main.py           # browse to http://127.0.0.1:8000
```

Use `Ctrl+C` in the terminal to stop the web server. Do not commit a Python virtual environment; the dependencies file is the reproducible setup.

## 4. Preprocessing

`program.load_data()` validates the two expected columns, drops missing/empty/invalid-label rows, and removes duplicate message text. It encodes labels (`scam = 1`, `legit = 0`). `clean_text()` lowercases text, changes URLs, phone numbers, and numbers into meaningful tokens, removes punctuation, and normalises a few common Roman-Urdu spellings. `TfidfVectorizer` turns words and adjacent word pairs into numeric TF-IDF features. TF-IDF uses L2 normalisation, so a separate scaler is not required. The supplied files are already the stratified train/test split; production code never fits on the test set.

## 5. EDA and models

Run `analysis.py` before the presentation. It writes:

- `artifacts/data_quality_summary.csv` — rows, labels, missing values, duplicates, and average message length;
- `artifacts/class_distribution.png` — class balance by train/test split;
- `artifacts/message_length.png` — length distributions; and
- `artifacts/model_comparison.csv` — the final test metrics.

The classifiers are Multinomial Naive Bayes, Logistic Regression, Linear SVM (calibrated to provide probabilities), and Random Forest. They are evaluated on untouched test data with accuracy, precision, recall, and F1. F1 is used to select the deployment model because the dataset is imbalanced and both missed scams (low recall) and false alarms (low precision) matter. The console and CSV state the actual winner; do not replace those values with hand-written scores.

## 6. Code map / explanation

| File | What it does |
| --- | --- |
| `program.py` | Core ML logic: cleaning, validation, TF-IDF/model pipelines, metrics, and one-message prediction. A pipeline prevents the vectoriser being fitted on test data. |
| `analysis.py` | Reproducible report evidence. It inspects the input files, saves charts/CSVs, trains all models, and prints the recommended model. |
| `main.py` | Starts the local HTTP server. It trains once at startup; `GET /` serves the page and `POST /predict` validates a small JSON request then returns a prediction. |
| `interface.html` | Browser UI: reads textarea text, sends it with `fetch`, and presents a colour-coded response. It does not contain a model or training data. |
| `data.csv` | The single source dataset. Code splits it into unseen training and test partitions at runtime. |
| `REPORT.md` | Presentation-ready report narrative and speaker prompts. |

## 7. GitHub submission

Create a GitHub repository, then from this directory:

```powershell
git add .
git commit -m "Build ScamShield AI project prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/scamshield.git
git push -u origin main
```

Replace the placeholder URL with your own repository. Submit that link, this dataset-source link, `REPORT.md` (or export it to PDF), the generated charts, and a short screen recording of `main.py` plus two example predictions.

## Limitations and responsible use

The Roman-Urdu set is small and synthetic; the English source is an old UK SMS collection. Results may be worse for current, multilingual, adversarial, or context-dependent scams. A 50% probability is a demonstration threshold, not proof. Never paste personal banking information into a demo and always independently verify suspicious requests.
