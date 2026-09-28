import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .tools import ReconstructDataset, my_kl_loss
# import stumpy


def MARE(preds, target):
    return (preds - target).abs().mean() / (target.abs().mean() + 1e-8)



class Trainer:
    """
    Minimal trainer for deep learning reconstruction models on time series.

    Model contract:
      forward(x): [B, L, D] -> [B, L, D] (reconstruction)
    """

    def __init__(
        self,
        batch_size=32,
        lr=1e-3,
        device='cuda',
        win_size=100,
        validation_size=0.1,
        lr_scheduler="none",
        lr_scheduler_kwargs=None,
    ):
        print(f"Trainer device: {device}")
        # Config
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.win_size = win_size
        self.validation_size = validation_size
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # losses / optimizer factory
        self.criterion = nn.MSELoss()
        self.optimizer_fn = optim.Adam
        self.mask_value = 0.0
        self.train_losses = []

    def _build_scheduler(self, optimizer, total_epochs):
        if self.lr_scheduler in (None, "none"):
            return None

        if self.lr_scheduler == "step":
            step_size = self.lr_scheduler_kwargs.get(
                "step_size", max(1, total_epochs // 10))
            gamma = self.lr_scheduler_kwargs.get("gamma", 0.5)
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )

        if self.lr_scheduler == "plateau":
            factor = self.lr_scheduler_kwargs.get("factor", 0.5)
            patience = self.lr_scheduler_kwargs.get("patience", 30)
            min_lr = self.lr_scheduler_kwargs.get("min_lr", 1e-5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        raise ValueError(
            f"Unknown lr_scheduler '{self.lr_scheduler}'. "
            "Use one of: 'none', 'step', 'plateau'."
        )

    def _split(self, data):
        cut = int((1.0 - self.validation_size) * len(data))
        return data[:cut], data[cut:]

    def _loader(self, ts, win_size, shuffle=False):
        ds = ReconstructDataset(ts, window_size=win_size,
                                stride=1, normalize=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _fit_epochs(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        scheduler=None,
    ):
        for _ in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch in train_loader:
                inputs = batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = self.criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                # epoch_train_relative_loss += relative_loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            # avg_train_relative_loss = epoch_train_relative_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            # self.train_relative_losses.append(avg_train_relative_loss)

            model.eval()
            epoch_val_loss = 0.0
            # epoch_val_relative_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch.to(self.device)
                    outputs = model(inputs)
                    loss = self.criterion(outputs, inputs)
                    # relative_loss = MARE(outputs, inputs)
                    epoch_val_loss += loss.item()
                    # epoch_val_relative_loss += relative_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            # avg_val_relative_loss = epoch_val_relative_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            # self.val_relative_losses.append(avg_val_relative_loss)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            self.lrs.append(optimizer.param_groups[0]['lr'])

    def train_with_checkpoints(self, model, data, checkpoint_epochs):
        model = model.to(self.device)

        self.train_losses = []
        self.train_relative_losses = []
        self.val_losses = []
        self.val_relative_losses = []
        self.lrs = []

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        scheduler = self._build_scheduler(
            optimizer, total_epochs=max(checkpoint_epochs))
        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        val_loader = self._loader(
            data_val, win_size=self.win_size, shuffle=False)

        previous_epoch = 0
        for checkpoint_epoch in checkpoint_epochs:
            delta_epochs = checkpoint_epoch - previous_epoch
            if delta_epochs <= 0:
                continue
            self._fit_epochs(
                model,
                train_loader,
                val_loader,
                optimizer,
                delta_epochs,
                scheduler=scheduler,
            )
            previous_epoch = checkpoint_epoch

            # Yield current checkpoint to let caller run evaluation.
            yield checkpoint_epoch

    def train(self, model, data, epochs, patience=20):
        model = model.to(self.device)

        self.train_losses = []
        self.train_relative_losses = []
        self.val_losses = []
        self.val_relative_losses = []
        self.lrs = []

        # Initialize early stopping variables
        self.early_stop_epoch = epochs  # Default to max epochs if no early stop occurs

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        scheduler = self._build_scheduler(optimizer, total_epochs=epochs)
        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        val_loader = self._loader(
            data_val, win_size=self.win_size, shuffle=False)

        best_val_loss = float("inf")
        best_params = None

        self._fit_epochs(
            model,
            train_loader,
            val_loader,
            optimizer,
            epochs,
            scheduler=scheduler,
        )

            # --- Early Stopping Logic ---
            # Using your 0.5% improvement threshold
            # if avg_val_loss < best_val_loss * 0.995:
            #     best_val_loss = avg_val_loss
            #     best_params = {k: v.cpu()
            #                    for k, v in model.state_dict().items()}
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1

            # if epochs_no_improve >= patience:
            #     # print(f"Early stopping triggered at epoch {epoch}")
            #     self.early_stop_epoch = epoch
            #     break

        # Load best params at the end of training
        if best_params is not None:
            model.load_state_dict(best_params)

    def evaluate_loss(self, model, data):
        """Return mean reconstruction MSE over all windows in ``data``."""
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        loader = self._loader(data, win_size=self.win_size, shuffle=False)
        if len(loader) == 0:
            raise ValueError(
                f"need at least {self.win_size} values to evaluate loss"
            )

        model = model.to(self.device)
        model.eval()
        total_loss = 0.0
        total_windows = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch.to(self.device)
                outputs = model(inputs)
                total_loss += self.criterion(outputs, inputs).item() * len(batch)
                total_windows += len(batch)
        return total_loss / total_windows

    def train_low_freq(self, model, data, epochs):
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 10), gamma=0.5)

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                #  remove high freq components
                # Remove high freq components using moving average
                kernel_size = max(1, inputs.shape[1] // 10)
                if kernel_size > 1:
                    inputs_low_freq = torch.nn.functional.avg_pool1d(
                        inputs.transpose(1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size//2
                    ).transpose(1, 2)
                else:
                    inputs_low_freq = inputs

                outputs = model(inputs_low_freq)

                loss = self.criterion(outputs, inputs_low_freq)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
