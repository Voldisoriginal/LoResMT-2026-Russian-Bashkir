import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= CONFIGURATION =================
MODEL_ID = "facebook/m2m100_418M"
OUTPUT_DIR = "../experiments/m2m100_lora"
TRAIN_FILE = "../data/train_with_scores.parquet"
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5


# =================================================

def train():
    print(f">>> Loading M2M-100 Data...")
    dataset = load_dataset("parquet", data_files=TRAIN_FILE, split="train")

    # Moderate filtering (Score >= 0.60)
    dataset = dataset.filter(lambda x: x["similarity_score"] >= 0.60)
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.train_test_split(test_size=1000)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="ru", tgt_lang="ba")

    def preprocess_function(examples):
        inputs = [str(ex) for ex in examples['ru']]
        targets = [str(ex) for ex in examples['ba']]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    print(">>> Initializing M2M Model with LoRA...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    # Adding LoRA adapters
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # M2M specific configuration
    model.config.forced_bos_token_id = tokenizer.get_lang_id("ba")
    model.config.use_cache = False

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        num_train_epochs=EPOCHS,
        bf16=True,
        logging_steps=50,
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