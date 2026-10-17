import copy
import time

import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.best_weights = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False

    def check(self, val_loss, model, epoch):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = running_loss / total_samples
    train_accuracy = correct_predictions / total_samples
    return train_loss, train_accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    val_loss = running_loss / total_samples
    val_accuracy = correct_predictions / total_samples
    return val_loss, val_accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5,
                device=None, scheduler=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"train_loss {train_loss:.4f} | train_acc {train_accuracy:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_accuracy:.4f} | "
              f"{epoch_duration:.1f}s")

        early_stopping.check(val_loss, model, epoch)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}. Best epoch: {early_stopping.best_epoch}")
            break

    early_stopping.restore_best_weights(model)
    history["best_epoch"] = early_stopping.best_epoch
    history["stopped_epoch"] = epoch
    return model, history