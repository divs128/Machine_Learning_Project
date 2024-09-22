import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def prepare_data(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load train data
    train_data = pd.read_csv(os.path.join(input_dir, 'train.csv'))

    # Remove unnecessary columns if necessary
    train_data = train_data[['claim', 'label', 'explanation']]

    # Split into training and validation datasets
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Save the processed data
    train_data.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val_processed.csv'), index=False)
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and prepare data for model training.")
    parser.add_argument('--input_dir', type=str, default="./data", help="Directory containing the dataset")
    parser.add_argument('--output_dir', type=str, default="./processed_data", help="Directory to save the processed dataset")
    args = parser.parse_args()

    prepare_data(args.input_dir, args.output_dir)
