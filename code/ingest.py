from datasets import load_dataset
import argparse
import os


def download_data(dataset_name: str, save_dir: str):
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    os.makedirs(save_dir, exist_ok=True)
    dataset['train'].to_csv(os.path.join(save_dir, 'train.csv'))
    dataset['test'].to_csv(os.path.join(save_dir, 'test.csv'))
    print(f"Dataset downloaded and saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data and save to folder.")
    parser.add_argument('--save_dir', type=str, default="./data", help='Directory to save the dataset')
    args = parser.parse_args()

    DATASET_NAME = "ImperialCollegeLondon/health_fact"
    download_data(DATASET_NAME, args.save_dir)
