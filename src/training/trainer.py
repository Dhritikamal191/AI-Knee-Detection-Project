from pathlib import Path
import torch


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler,
        device,
        checkpoint_dir="artifacts/checkpoints"
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.loss_fn(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)

        epoch_accuracy = correct / max(total, 1)

        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def validate(self, dataloader):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.loss_fn(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        val_loss = running_loss / max(total, 1)

        val_accuracy = correct / max(total, 1)

        return val_loss, val_accuracy

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        filename="best_model.pt"
    ):

        checkpoint_path = (
            self.checkpoint_dir / filename
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "val_loss": val_loss
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=10
    ):

        history = []

        for epoch in range(1, epochs + 1):

            train_loss, train_accuracy = (
                self.train_epoch(train_loader)
            )

            val_loss, val_accuracy = (
                self.validate(val_loader)
            )

            if self.scheduler is not None:

                self.scheduler.step(
                    val_loss
                )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }
            )

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                self.save_checkpoint(
                    epoch,
                    val_loss
                )

        return history