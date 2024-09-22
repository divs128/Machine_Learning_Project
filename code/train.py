import os
import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, \
    DataCollatorWithPadding
import sys
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str):
    """
    Load data from a CSV file and return it as a Hugging Face Dataset.
    """
    try:
        df = pd.read_csv(file_path)
        # Check if 'claim' column exists
        if 'claim' not in df.columns:
            logger.error(f"Error: 'claim' column not found in {file_path}.")
            sys.exit(1)

        # Remove rows where 'claim' is null or empty
        df = df.dropna(subset=['claim'])
        df = df[df['claim'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

        dataset = Dataset.from_pandas(df)
        return dataset
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} does not exist. Please provide a valid path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading the dataset: {e}")
        sys.exit(1)


def tokenize_data(dataset, tokenizer, max_length=512):
    """
    Tokenize the dataset using the provided tokenizer.
    """
    logger.info("Tokenizing data...")

    def tokenize_fn(examples):
        claims = examples['claim']
        if not isinstance(claims, list):
            claims = [claims]  # Ensure claims is always a list

        valid_claims = []
        for claim in claims:
            if isinstance(claim, str) and claim.strip() != '':
                valid_claims.append(claim)
            else:
                valid_claims.append("[UNK]")  # Replace invalid claims with a placeholder

        return tokenizer(valid_claims, truncation=True, padding='max_length', max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    return tokenized_dataset


def train_model(train_file: str, val_file: str, epochs: int = 3, batch_size: int = 2, use_gpu=False):
    try:
        # Load datasets
        train_dataset = load_data(train_file)
        val_dataset = load_data(val_file)

        # Reduce dataset size for testing/debugging
        train_dataset = train_dataset.select(range(100))  # Reduce size to first 100 samples
        val_dataset = val_dataset.select(range(50))  # Reduce size to first 50 samples

        # Load tokenizer and model
        logger.info("Loading model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

        # Tokenize datasets
        train_dataset = tokenize_data(train_dataset, tokenizer, max_length=512)  # Reduced max length to 512
        val_dataset = tokenize_data(val_dataset, tokenizer, max_length=512)

        # Set format for PyTorch tensors
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # Define data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Device setup: Force CPU training for debugging purposes
        device = torch.device("cpu")
        model.to(device)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,  # Reduced batch size to 2
            per_device_eval_batch_size=batch_size,  # Matching eval batch size
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",  # Save model at the end of each epoch
            logging_dir='./logs',
            logging_steps=200,  # Log every 200 steps
            save_total_limit=2,  # Keep the 2 most recent models
            load_best_model_at_end=True,  # Load the best model at the end
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
            no_cuda=True,  # Force CPU usage
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,  # Use data collator for batch processing
            tokenizer=tokenizer,  # Include tokenizer to handle text properly
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the trained model
        logger.info("Saving the model...")
        model.save_pretrained('./trained_model')
        tokenizer.save_pretrained('./trained_model')

        logger.info("Training completed successfully!")

    except ImportError as e:
        logger.error(f"Error: Required library not found. {e}")
        logger.error("Please ensure that PyTorch, TensorFlow, or the necessary transformers are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Hardcode the file paths and parameters here
    train_file = "./processed_data/train_processed.csv"
    val_file = "./processed_data/val_processed.csv"
    epochs = 3
    batch_size = 2  # Reduced batch size to prevent memory issues
    use_gpu = False  # Force CPU training

    # Ensure files exist
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        logger.error("Error: One or both of the dataset files do not exist.")
        sys.exit(1)

    # Run the training
    train_model(train_file, val_file, epochs, batch_size, use_gpu)
