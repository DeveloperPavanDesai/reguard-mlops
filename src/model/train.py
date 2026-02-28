import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.model.model import ANN
from src.data.load_data import get_data_loaders

def train_model(dropout=0.0, l1_lambda=0.0, l2_lambda=0.0, epochs=5):

    train_loader, test_loader = get_data_loaders()

    model = ANN(dropout_rate=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=l2_lambda)

    mlflow.start_run()

    mlflow.log_param("dropout", dropout)
    mlflow.log_param("l1_lambda", l1_lambda)
    mlflow.log_param("l2_lambda", l2_lambda)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # L1 Regularization
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        mlflow.log_metric("train_loss", total_loss, step=epoch)

    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    train_model()