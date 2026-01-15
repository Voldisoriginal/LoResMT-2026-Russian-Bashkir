import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, DataCollatorForSeq2Seq
)

# ================= CONFIGURATION =================
MODEL_ID = "Helsinki-NLP/opus-mt-en-trk"
OUTPUT_DIR = "../experiments/marian_custom_vocab"
TRAIN_FILE = "../data/train_with_scores.parquet"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5


# =================================================

def train():
    print(f">>> Loading MarianMT Data...")
    dataset = load_dataset("parquet", data_files=TRAIN_FILE, split="train")

    # Broad filtering for max vocabulary exposure (Score >= 0.10)
    dataset = dataset.filter(lambda x: x["similarity_score"] >= 0.10)
    dataset = dataset.shuffle(seed=42).select(range(900000))  # Cap at 900k
    dataset = dataset.train_test_split(test_size=1000)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 1. Vocabulary Expansion (Bashkir Cyrillic)
    new_tokens = ["Ҙ", "ҙ", "Ҫ", "ҫ", "Ҡ", "ҡ", "Ғ", "ғ", "Һ", "һ"]
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]

    if tokens_to_add:
        print(f"Adding tokens to vocab: {tokens_to_add}")
        tokenizer.add_tokens(tokens_to_add)

    def preprocess_function(examples):
        # Adding prefix for transfer learning
        inputs = [f">>bak<< {str(ex)}" for ex in examples['ru']]
        targets = [str(ex) for ex in examples['ba']]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    print(">>> Initializing MarianMT (Full Fine-Tuning)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to("cuda")

    # Resize embeddings for new tokens
    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps", eval_steps=2000,
        save_strategy="steps", save_steps=2000,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        fp16=False,  # Important: FP32 for stability on small custom vocab models
        logging_steps=100,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()