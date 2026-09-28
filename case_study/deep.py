"""Deep-model configuration, training, and reconstruction utilities."""

from types import SimpleNamespace

import numpy as np

from anomaly_detection.utils import Evaluator, Trainer, train_and_evaluate
from anomaly_detection.models.AutoEncoder import Model as AutoEncoder
from anomaly_detection.models.DLinear import Model as DLinear
from anomaly_detection.models.TimesNet import Model as TimesNet


DEEP_HYPERPARAMETERS = {
    "AutoEncoder": {
        "model": {"hidden_ratios": [.5, .25]},
        "training": {"epochs": 50, "batch_size": 1024, "lr": 1e-2},
    },
    "DLinear": {
        "model": {"moving_avg_ratio": .1},
        "training": {"epochs": 50, "batch_size": 1024, "lr": 1e-2},
    },
    "TimesNet": {
        "model": {
            "d_model": 16,
            "d_ff": 32,
            "e_layers": 2,
            "top_k": 3,
            "num_kernels": 6,
            "embed": "timeF",
            "freq": "d",
            "dropout": .1,
        },
        "training": {"epochs": 50, "batch_size": 1024, "lr": 1e-2},
    },
}
DEEP_SEQUENCE_LENGTH = 96

MODEL_CLASSES = {
    "AutoEncoder": AutoEncoder,
    "DLinear": DLinear,
    "TimesNet": TimesNet,
}


def resolved_model_hyperparameters(name, subsequence_length):
    """Resolve derived parameters without changing model implementations."""
    model_hp = dict(DEEP_HYPERPARAMETERS[name]["model"])
    if name == "DLinear":
        ratio = model_hp.pop("moving_avg_ratio")
        if not 0 < ratio <= 1:
            raise ValueError("DLinear moving_avg_ratio must be in (0, 1]")
        kernel = max(1, int(ratio * subsequence_length))
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        model_hp["moving_avg"] = kernel
    return model_hp


def build_models():
    base = {
        "task_name": "anomaly_detection",
        "seq_len": DEEP_SEQUENCE_LENGTH,
        "label_len": 0,
        "pred_len": 0,
        "enc_in": 1,
        "c_out": 1,
    }
    return {
        name: model_class(SimpleNamespace(
            **base,
            **resolved_model_hyperparameters(name, DEEP_SEQUENCE_LENGTH),
        ))
        for name, model_class in MODEL_CLASSES.items()
    }


def train_models(train, test):
    """Standardize, construct, train, and score every configured deep model."""
    mean, std = train.mean(), max(train.std(), 1e-8)
    standardized_train = (train - mean) / std
    standardized_test = (test - mean) / std
    models = build_models()
    scores = {}
    losses = {}
    for name, model in models.items():
        hp = DEEP_HYPERPARAMETERS[name]["training"]
        trainer = Trainer(
            batch_size=hp["batch_size"],
            lr=hp["lr"],
            win_size=DEEP_SEQUENCE_LENGTH,
        )
        scores[name] = train_and_evaluate(
            standardized_train, standardized_test, model,
            DEEP_SEQUENCE_LENGTH, hp["epochs"], trainer,
            batch_size=hp["batch_size"],
        )
        losses[name] = {
            "train_loss": trainer.train_losses[-1],
            "val_loss": trainer.val_losses[-1],
            "test_loss": trainer.evaluate_loss(model, standardized_test),
        }
    return models, scores, standardized_test, mean, std, losses


def reconstruct_models(models, standardized_test, mean, std):
    """Return overlapping reconstructions mapped back to physical units."""
    reconstructions = {}
    for name, model in models.items():
        batch_size = DEEP_HYPERPARAMETERS[name]["training"]["batch_size"]
        reconstruction = Evaluator(batch_size=batch_size).overlapping_reconstruct(
            standardized_test[:, None].astype(np.float32), model,
            DEEP_SEQUENCE_LENGTH, 1)
        reconstructions[name] = reconstruction.cpu().numpy() * std + mean
    return reconstructions


def reconstruction_error_summary(test, reconstructions):
    """Summarize absolute errors of overlapping reconstructions."""
    test = np.asarray(test, dtype=float).reshape(-1)
    rows = {}
    for name, reconstruction in reconstructions.items():
        reconstruction = np.asarray(reconstruction, dtype=float).reshape(-1)
        if len(reconstruction) != len(test):
            raise ValueError(
                f"{name} reconstruction has {len(reconstruction)} values; "
                f"expected {len(test)}"
            )
        absolute_error = np.abs(test - reconstruction)
        rows[name] = {
            "absolute_error_median": float(np.median(absolute_error)),
            "absolute_error_mean": float(np.mean(absolute_error)),
            "absolute_error_max": float(np.max(absolute_error)),
        }
    return rows
