# Machine_Learning_Project


# **ML Fact-Checking with MLOps: Public Health Claims**

## **Overview**
This project builds a fact-checking solution to verify public health claims using ML and MLOps. It processes the **PUBHEALTH** dataset and fine-tunes an NLP model for veracity predictions (true, false, unproven, mixture).


## **Workflow**
1. **Data Collection**: Downloads the dataset.
   ```bash
   python code/ingest.py
   ```
2. **Data Processing**: Prepares data for training.
   ```bash
   python code/prepare.py
   ```
3. **Model Training**: Fine-tunes the model.
   ```bash
   python code/train.py
   ```
4. **Deployment**: Runs the FastAPI service.
   ```bash
   uvicorn code.serve:app --host 0.0.0.0 --port 8000
   ```
