import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

# ================= CONFIGURATION =================
MODEL_ID = "facebook/nllb-200-1.3B"  # Base model
OUTPUT_DIR = "experiments/nllb_1.3b_qlora"
TRAIN_FILE = "dataset/train_with_scores.parquet"

# Training hyperparameters
# We select the top high-quality samples based on semantic similarity
MAX_TRAIN_SAMPLES = 486517
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8
EPOCHS = 1
LEARNING_RATE = 2e-4


# =================================================

def prepare_data():
    """
    Loads, filters, and tokenizes the dataset.
    Filters data based on BERT-based semantic similarity score >= 0.80.
    """
    print(f">>> Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset("parquet", data_files=TRAIN_FILE, split="train")

    # 1. Semantic Filtering (High Precision Subset)
    dataset = dataset.filter(lambda x: x["similarity_score"] >= 0.80)

    # 2. Sort by quality
    dataset = dataset.sort("similarity_score", reverse=True)

    # 3. Create splits
    valid_dataset = dataset.select(range(2000))
    train_dataset = dataset.select(range(2000, 2000 + MAX_TRAIN_SAMPLES))
    train_dataset = train_dataset.shuffle(seed=42)

    print(f"Dataset Stats:\nTrain: {len(train_dataset)}\nValidation: {len(valid_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="rus_Cyrl", tgt_lang="bak_Cyrl")
    cols_to_remove = dataset.column_names

    def preprocess_function(examples):
        keys = list(examples.keys())
        ru_col = "ru" if "ru" in keys else "russian"
        ba_col = "ba" if "ba" in keys else "bashkir"

        inputs = [str(ex) for ex in examples[ru_col]]
        targets = [str(ex) for ex in examples[ba_col]]

        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(text_target=targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(">>> Tokenizing...")
    tokenized_datasets = {}
    tokenized_datasets["train"] = train_dataset.map(preprocess_function, batched=True, remove_columns=cols_to_remove)
    tokenized_datasets["test"] = valid_dataset.map(preprocess_function, batched=True, remove_columns=cols_to_remove)

    return tokenized_datasets, tokenizer


def train(tokenized_datasets, tokenizer):
    """
    Fine-tunes the NLLB-1.3B model using QLoRA.
    """
    print(">>> Initializing Model with QLoRA config...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Gradient checkpointing disabled for stability in this specific setup
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.config.use_cache = False

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=False,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        bf16=True,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(">>> Starting Training...")
    trainer.train()

    print(f">>> Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    tokenized_data, tokenizer = prepare_data()
    train(tokenized_data, tokenizer)