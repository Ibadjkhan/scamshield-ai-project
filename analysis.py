"""Create reproducible EDA charts and the model-comparison CSV for the report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import program

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "artifacts"


def raw_summary() -> pd.DataFrame:
    """Summarise the single source CSV before preprocessing and splitting."""
    frame = pd.read_csv(HERE / "data.csv")
    return pd.DataFrame([{"dataset": "data.csv", "messages": len(frame),
                          "scam": (frame.label == "scam").sum(), "legit": (frame.label == "legit").sum(),
                          "missing_values": int(frame.isna().sum().sum()),
                          "duplicate_texts": int(frame.duplicated("text").sum()),
                          "mean_characters": round(frame.text.astype(str).str.len().mean(), 1)}])


def make_eda_charts(X_train, y_train, X_test, y_test, model) -> None:
    """Save EDA plots plus final-model evaluation plots."""
    all_data = pd.concat([pd.DataFrame({"text": X_train, "label": y_train.map({0: "legit", 1: "scam"}), "split": "Train"}),
                          pd.DataFrame({"text": X_test, "label": y_test.map({0: "legit", 1: "scam"}), "split": "Test"})])
    counts = pd.crosstab(all_data["split"], all_data["label"])[["legit", "scam"]]
    counts.plot(kind="bar", color=["#3b82f6", "#ef4444"], rot=0)
    plt.title("Class distribution by split")
    plt.ylabel("Number of messages")
    plt.xlabel("")
    plt.tight_layout(); plt.savefig(OUTPUT / "class_distribution.png", dpi=180); plt.close()

    all_data["characters"] = all_data.text.astype(str).str.len()
    for label, colour in (("legit", "#3b82f6"), ("scam", "#ef4444")):
        plt.hist(all_data.loc[all_data.label == label, "characters"], bins=35, alpha=.6, label=label, color=colour)
    plt.title("Message-length distribution")
    plt.xlabel("Characters per message"); plt.ylabel("Number of messages"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTPUT / "message_length.png", dpi=180); plt.close()

    vectorizer = model.named_steps["tfidfvectorizer"]
    matrix = vectorizer.transform(X_train)
    terms = vectorizer.get_feature_names_out()
    scam_mean = matrix[y_train.to_numpy() == 1].mean(axis=0).A1
    legit_mean = matrix[y_train.to_numpy() == 0].mean(axis=0).A1
    top = (scam_mean - legit_mean).argsort()[-12:][::-1]
    plt.barh(terms[top][::-1], (scam_mean - legit_mean)[top][::-1], color="#ef4444")
    plt.title("Terms relatively associated with scam messages")
    plt.xlabel("Difference in mean TF-IDF weight")
    plt.tight_layout(); plt.savefig(OUTPUT / "scam_terms.png", dpi=180); plt.close()

    predicted = model.predict(X_test)
    confusion = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(predicted, name="Predicted"))
    confusion = confusion.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion.values, cmap="Blues")
    for row in range(2):
        for col in range(2): ax.text(col, row, str(confusion.iloc[row, col]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Legit", "Scam"]); ax.set_yticks([0, 1], ["Legit", "Scam"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label"); ax.set_title("Best-model confusion matrix")
    fig.colorbar(image, ax=ax); plt.tight_layout(); plt.savefig(OUTPUT / "confusion_matrix.png", dpi=180); plt.close()


def build_artifacts():
    """Run preprocessing, EDA, and comparison; return the production model."""
    OUTPUT.mkdir(exist_ok=True)
    summary = raw_summary(); summary.to_csv(OUTPUT / "data_quality_summary.csv", index=False)
    X_train, X_test, y_train, y_test, split_audit = program.load_and_split(HERE / "data.csv")
    results, best_name, model = program.train_and_compare(X_train, y_train, X_test, y_test)
    signal_profiles = program.build_signal_profiles(model, X_train, y_train)
    results.to_csv(OUTPUT / "model_comparison.csv")
    make_eda_charts(X_train, y_train, X_test, y_test, model)
    dashboard = {"best_model": best_name, "metrics": results.reset_index().to_dict(orient="records"), "summary": summary.to_dict(orient="records"), "audit": split_audit}
    (OUTPUT / "dashboard.json").write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    return results, best_name, model, dashboard, signal_profiles


if __name__ == "__main__":
    results, best_name, _, dashboard, _ = build_artifacts()
    print(pd.DataFrame(dashboard["summary"]).to_string(index=False))
    print("\nPreprocessing audit:", dashboard["audit"])
    print("\nModel comparison:\n", results.to_string())
    print(f"\nRecommended model: {best_name} (highest F1).")
