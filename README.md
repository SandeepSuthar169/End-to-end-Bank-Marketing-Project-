## End-to-End Bank Marketing Project
📌 Overview
This project aims to develop a machine learning pipeline to predict whether a client will subscribe to a term deposit based on data from direct marketing campaigns of a Portuguese banking institution. The pipeline encompasses data ingestion, preprocessing, model training, evaluation, and deployment, ensuring reproducibility and scalability.



### 📁 Project Structure
```bash
End-to-end-Bank-Marketing-Project/
│
├── data/                      # Data files (raw, processed)
│   └── bank.csv
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── DVC_Pipeline.py
│   ├── Excrement_Tracking.ipynb
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── dvc.yaml
│   ├── model_building.py
│   ├── model_evaluation.py
│   │
│   │
│   │                          # Main pipeline script
│   └── utils.py               # Utility functions (optional)
│
├── tests/                     # Unit and integration tests
│   └── test_pipeline.py
│
├── .github/                   # GitHub actions workflows
│   └── ci.yml                 # CI workflow
│
├── .dvc/                      # DVC configuration folder
│
├── models/                    # Saved models or checkpoints
│
├── params.yaml                # Parameters config file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package install file (optional)
├── README.md                  # Project overview and instructions
└── LICENSE                   # License file (optional)


```

### 🧰 Technologies Used
- Programming Language:`Python`

- Data Version Control: `DVC`

- Workflow Automation: `GitHub Actions`

- Machine Learning Libraries: `scikit-learn, pandas, NumPy`

- Visualization: `Matplotlib, Seaborn`

- Notebook Environment: `Jupyter Notebook` 
 ### 📊 Dataset

The dataset is sourced from the UCI Machine Learning Repository. It contains information on direct marketing campaigns (phone calls) conducted by a Portuguese banking institution. The goal is to predict if a client will subscribe to a term deposit.

- Total Instances: 41,188

- Features: 20 input variables (categorical and numerical)

- Target Variable: y (binary: 'yes' or 'no')
### 🚀 Getting Started
Prerequisites
Ensure you have the following installed:

- Python 3.13.3

- pip

- DVC


### Installation
1. Clone the repository:
```
git clone https://github.com/SandeepSuthar169/End-to-end-Bank-Marketing-Project-.git
cd End-to-end-Bank-Marketing-Project-
```
2. Create a virtual environment and activate it:

```
python -m venv venv
venv\Scripts\activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
4. Set up DVC and pull the data:
```
dvc pull
```


### ⚙️ Usage
1. Data Exploration and Preprocessing:

Navigate to the `Notebook/` directory and open the Jupyter notebooks to explore and preprocess the data.

2. Model Training:

Use the scripts in the `src/` directory to train machine learning models. You can configure hyperparameters in the params.yaml file.

3. Evaluation:

After training, evaluation metrics are stored in `metrics.json`. Use visualization tools to analyze model performance.

4. Pipeline Execution:

DVC pipelines are defined in `dvc.yaml`. To reproduce the entire pipeline:

```
dvc repro
```
📈 Model Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics are logged in the `metrics.json` file for easy tracking and comparison.

🔄 Continuous Integration

GitHub Actions are set up to automate testing and ensure code quality. Workflows are defined in the `.github/workflows/` directory.
