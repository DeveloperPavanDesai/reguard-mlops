# Reguard MLOps

MNIST digit classification pipeline with experiment tracking (MLflow), data versioning (DVC), and a FastAPI inference API.

## Project structure

```
reguard-mlops/
├── src/
│   ├── data/
│   │   └── load_data.py    # MNIST data loaders
│   ├── model/
│   │   ├── model.py        # ANN architecture
│   │   └── train.py        # Training + MLflow logging
│   └── api/
│       └── app.py          # FastAPI prediction endpoint
├── data/raw/               # MNIST data (DVC-tracked)
├── models/                 # Saved model weights (mnist_model.pth)
├── mlflow.db               # MLflow tracking DB (SQLite)
├── mlruns/                 # MLflow artifacts (when using server)
├── requirements.txt
├── dvc.yaml
└── docker-compose.yml
```

## Setup

1. **Clone and create a virtual environment**

   ```bash
   cd reguard-mlops
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   For the API you also need FastAPI and Uvicorn:

   ```bash
   pip install fastapi uvicorn
   ```

3. **Optional: fix SSL certificates (macOS)**

   If MNIST download fails with `SSL: CERTIFICATE_VERIFY_FAILED`:

   ```bash
   export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
   export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
   ```

   Or run the system certificate installer:

   ```bash
   open "/Applications/Python 3.12/Install Certificates.command"
   ```

## Data

- MNIST is downloaded automatically on first run into `data/raw/`.
- To track raw data with DVC: `dvc add data/raw`, then commit `data/raw.dvc` and `.gitignore` changes.

## Training

1. **Start the MLflow server** (so artifacts and UI work):

   ```bash
   mlflow server \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns \
     --host 0.0.0.0 \
     --port 5000
   ```

2. **In another terminal**, set the tracking URI and run training:

   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   python -m src.model.train
   ```

   Training uses the default experiment, logs params (dropout, L1/L2) and metrics (train loss), and saves the model to `models/mnist_model.pth` and to MLflow.

3. **View runs**: open [http://localhost:5000](http://localhost:5000) in your browser.

## Inference API

After training, a model file is saved at `models/mnist_model.pth`. Start the API:

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

- **GET /** — Health check.
- **POST /predict** — Body: `{"image": [list of 784 floats]}` (28×28 grayscale). Returns `predicted_digit` and `confidence`.

## Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

Ports: 5000 (MLflow), 8000 (API). Override the default command to run the server or training as needed.

## Requirements (summary)

- **Python 3.10+** (3.12 used in development)
- **PyTorch** + **torchvision** — model and MNIST
- **MLflow** — experiment tracking and model registry
- **DVC** — data versioning
- **FastAPI** + **uvicorn** — inference API (add to `requirements.txt` if you use the API)