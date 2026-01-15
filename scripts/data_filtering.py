import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= CONFIGURATION =================
# We download directly from Hugging Face Hub
DATASET_ID = "AigizK/bashkir-russian-parallel-corpora"
OUTPUT_FILE = "../data/train_with_scores.parquet"
METRIC_MODEL = "slone/bert-base-multilingual-cased-bak-rus-similarity"
BATCH_SIZE = 128


# =================================================

def prepare_dataset():
    print(f">>> Downloading dataset: {DATASET_ID}...")
    # Load the specific subset if needed, usually 'default' works
    try:
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Total samples: {len(dataset)}")

    # Identify column names automatically
    cols = dataset.column_names
    print(f"Columns found: {cols}")

    # Map columns to 'ru' and 'ba' for consistency
    ru_col = next((c for c in cols if c in ['ru', 'russian', 'rus']), None)
    ba_col = next((c for c in cols if c in ['ba', 'bashkir', 'bak']), None)

    if not ru_col or not ba_col:
        print(f"Error: Could not identify language columns in {cols}")
        return

    print(f">>> Loading Metric Model: {METRIC_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(METRIC_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(METRIC_MODEL).to("cuda")
    model.eval()

    def compute_similarity(batch):
        # Tokenize pairs
        inputs = tokenizer(
            batch[ru_col],
            batch[ba_col],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            # Class 1 corresponds to "semantically similar"
            probs = torch.softmax(outputs.logits, dim=1)
            scores = probs[:, 1].cpu().numpy()

        return {"similarity_score": scores}

    print(">>> Computing semantic similarity scores (this may take a while)...")
    # Apply scoring
    dataset_scored = dataset.map(compute_similarity, batched=True, batch_size=BATCH_SIZE)

    # Rename columns to standard 'ru'/'ba' for training scripts
    if ru_col != 'ru': dataset_scored = dataset_scored.rename_column(ru_col, 'ru')
    if ba_col != 'ba': dataset_scored = dataset_scored.rename_column(ba_col, 'ba')

    print(f">>> Saving processed dataset to {OUTPUT_FILE}...")
    dataset_scored.to_parquet(OUTPUT_FILE)
    print("Done. Ready for training.")


if __name__ == "__main__":
    prepare_dataset()