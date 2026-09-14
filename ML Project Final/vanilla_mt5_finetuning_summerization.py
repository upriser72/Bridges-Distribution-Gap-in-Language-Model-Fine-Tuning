!pip install -q torch datasets pandas transformers sentencepiece accelerate

import torch
from datasets import Dataset
import pandas as pd
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def main():

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    try:
        df = pd.read_csv(
            "Summarization_dataset.csv",
            engine="python",
            on_bad_lines="skip"
        )
    except FileNotFoundError:
        print("Error: 'Summarization_dataset.csv' not found.")
        print("Please upload the dataset to your Colab environment.")
        return

    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    dataset_split = dataset.train_test_split(
        train_size=9000,
        test_size=1000,
        seed=42
    )

    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # ---------------------------------------------------------
    # 3. Load pretrained mT5 model and tokenizer
    # ---------------------------------------------------------
    model_name = "google/mt5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # ---------------------------------------------------------
    # 4. Tokenization
    # ---------------------------------------------------------
    prefix = "summarize: "
    max_input_length = 512
    max_target_length = 150

    def preprocess_function(examples):

        # Input: article
        inputs = [
            prefix + str(article)
            for article in examples["article"]
        ]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        # Target: summary
        labels = tokenizer(
            text_target=examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

        # Ignore padding tokens when calculating loss
        labels["input_ids"] = [
            [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in label
            ]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    # ---------------------------------------------------------
    # 6. Training configuration
    # ---------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_mt5_summarization",

        num_train_epochs=3,

        
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,

        warmup_steps=50,
        weight_decay=0.01,

        logging_dir="./logs_summarization",
        logging_steps=10,

        save_total_limit=2,

        predict_with_generate=True,

        report_to="none",

        # Evaluation after each epoch
        eval_strategy="epoch",
        save_strategy="epoch",

        # Keep the best checkpoint based on evaluation loss
        load_best_model_at_end=True,

        fp16=torch.cuda.is_available()
    )

    # ---------------------------------------------------------
    # 7. Seq2Seq Trainer
    # ---------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,

        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ---------------------------------------------------------
    # 8. Vanilla Full Fine-Tuning
    # ---------------------------------------------------------
    print("Starting Vanilla Full Fine-Tuning...")

    trainer.train()

    print("Fine-tuning complete.")

    # ---------------------------------------------------------
    # 9. Save fine-tuned model
    # ---------------------------------------------------------
    final_model_path = "./fine_tuned_mt5_summarization"

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"Model saved to: {final_model_path}")

    # ---------------------------------------------------------
    # 10. Inference
    # ---------------------------------------------------------
    print("\n--- Running Inference ---")

    trained_model = MT5ForConditionalGeneration.from_pretrained(
        final_model_path
    )

    trained_tokenizer = AutoTokenizer.from_pretrained(
        final_model_path
    )

    article_to_summarize = (
        "Scientists have discovered a new species of glowing frog "
        "in a remote rainforest. The frog, which emits a soft blue "
        "light, has unique bioluminescent properties that are not "
        "fully understood. Researchers believe this could lead to "
        "new advancements in medical imaging. The ecosystem where "
        "the frog was found is incredibly delicate and under threat "
        "from deforestation."
    )

    prompt = f"summarize: {article_to_summarize}"

    inputs = trained_tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Move inputs to the same device as the model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    trained_model = trained_model.to(device)
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("Generating summary...")

    outputs = trained_model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    generated_text = trained_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nOriginal Article:")
    print(article_to_summarize)

    print("\nGenerated Summary:")
    print(generated_text)


if __name__ == "__main__":
    main()
