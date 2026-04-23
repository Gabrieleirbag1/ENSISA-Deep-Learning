import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

def _load_matrix(path: str) -> np.ndarray:
    for delim in [None, "\t", " "]:
        try:
            data = np.loadtxt(path, delimiter=delim)
            if data.ndim == 2 and data.shape[1] >= 2:
                return data
        except Exception:
            pass
    raise ValueError(f"Impossible de lire correctement {path}")


def load_ecg200(data_dir: str):
    train_path = os.path.join(data_dir, "ECG200_TRAIN.txt")
    test_path = os.path.join(data_dir, "ECG200_TEST.txt")

    train = _load_matrix(train_path)
    test = _load_matrix(test_path)

    y_train_raw = train[:, 0]
    X_train = train[:, 1:].astype(np.float32)
    y_test_raw = test[:, 0]
    X_test = test[:, 1:].astype(np.float32)

    uniq = sorted(np.unique(np.concatenate([y_train_raw, y_test_raw])).tolist())
    label_map = {lab: i for i, lab in enumerate(uniq)}
    y_train = np.array([label_map[v] for v in y_train_raw], dtype=np.int32)
    y_test = np.array([label_map[v] for v in y_test_raw], dtype=np.int32)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def build_mlp(input_length: int, n_classes: int) -> Model:
    inp = layers.Input(shape=(input_length,), name="ecg_input")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out, name="MLP_ECG200")
    return model


def build_cnn(input_length: int, n_classes: int) -> Model:
    inp = layers.Input(shape=(input_length, 1), name="ecg_input")
    x = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out, name="CNN1D_ECG200")
    return model


def build_rnn(input_length: int, n_classes: int) -> Model:
    inp = layers.Input(shape=(input_length, 1), name="ecg_input")
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out, name="RNN_LSTM_ECG200")
    return model


def compile_model(model: Model, lr: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


def ensure_dirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)


def prepare_dataset(data_dir: str):
    X_train, y_train, X_test, y_test = load_ecg200(data_dir)

    n_classes = len(np.unique(y_train))
    y_train_onehot = to_categorical(y_train, num_classes=n_classes)

    X_train_seq = X_train[..., np.newaxis]
    X_test_seq = X_test[..., np.newaxis]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_train_onehot": y_train_onehot,
        "n_classes": n_classes,
        "input_length": X_train.shape[1],
        "X_train_seq": X_train_seq,
        "X_test_seq": X_test_seq,
    }


def plot_history(history, model_name: str, fig_dir: str):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Courbe de perte")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{model_name}_history.png"))
    plt.close()

def plot_confusion_matrix_figure(cm: np.ndarray, model_name: str, fig_dir: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name} - Matrice de confusion")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, [f"Classe {i}" for i in range(cm.shape[0])])
    plt.yticks(tick_marks, [f"Classe {i}" for i in range(cm.shape[0])])
    plt.ylabel("Vrai label")
    plt.xlabel("Prédiction")
    plt.tight_layout()

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )
    
    plt.savefig(os.path.join(fig_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
def estimate_model_size_mb(model: Model) -> float:
    return (model.count_params() * 4) / (1024 ** 2)


def summarize_and_plot_comparison(results_df: pd.DataFrame, out_dir: str):
    fig_dir = os.path.join(out_dir, "figures")

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(results_df["model"]))
    width = 0.2

    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(metric_cols):
        plt.bar(x + (idx - 1.5) * width, results_df[metric].values, width=width, label=metric)

    plt.xticks(x, results_df["model"].values)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Comparaison des performances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "comparison_metrics.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.bar(results_df["model"], results_df["model_size_mb_est"], color="tab:purple")
    plt.ylabel("Taille estimée (MB)")
    plt.title("Taille des modèles")
    plt.xticks(rotation=15)

    plt.subplot(1, 3, 2)
    plt.bar(results_df["model"], results_df["train_time_s"], color="tab:orange")
    plt.ylabel("Temps d'entraînement (s)")
    plt.title("Complexité temporelle")
    plt.xticks(rotation=15)

    plt.subplot(1, 3, 3)
    plt.bar(results_df["model"], results_df["infer_time_s"], color="tab:green")
    plt.ylabel("Temps d'inférence (s)")
    plt.title("Temps d'inférence")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "comparison_complexity.png"), dpi=150)
    plt.close()


def train_and_evaluate_model(
    model_name: str,
    model: Model,
    X_train,
    y_train_onehot,
    X_test,
    y_test,
    epochs: int,
    batch_size: int,
    out_dir: str,
):
    ckpt_path = os.path.join(out_dir, "checkpoints", f"{model_name}.keras")
    fig_dir = os.path.join(out_dir, "figures")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train_onehot,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    train_time_s = time.perf_counter() - start

    start_infer = time.perf_counter()
    y_pred_proba = model.predict(X_test, verbose=0)
    infer_time_s = time.perf_counter() - start_infer
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plot_history(history, model_name, fig_dir)
    plot_confusion_matrix_figure(cm, model_name, fig_dir)

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
        "params": int(model.count_params()),
        "model_size_mb_est": estimate_model_size_mb(model),
        "train_time_s": train_time_s,
        "infer_time_s": infer_time_s,
    }

    if os.path.exists(ckpt_path):
        result["checkpoint_size_mb"] = os.path.getsize(ckpt_path) / (1024 ** 2)
    else:
        result["checkpoint_size_mb"] = np.nan

    print(f"\n===== {model_name} =====")
    print(pd.Series(result))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(cm)

    return result


def build_model_configs():
    return [
        {
            "name": "MLP",
            "builder": build_mlp,
            "train_key": "X_train",
            "test_key": "X_test",
        },
        {
            "name": "CNN1D",
            "builder": build_cnn,
            "train_key": "X_train_seq",
            "test_key": "X_test_seq",
        },
        {
            "name": "RNN_LSTM",
            "builder": build_rnn,
            "train_key": "X_train_seq",
            "test_key": "X_test_seq",
        },
    ]


def run_all_models(dataset: dict, epochs: int, batch_size: int, out_dir: str):
    input_length = dataset["input_length"]
    n_classes = dataset["n_classes"]
    model_configs = build_model_configs()

    results = []
    for config in model_configs:
        model = config["builder"](input_length=input_length, n_classes=n_classes)
        compile_model(model)

        result = train_and_evaluate_model(
            model_name=config["name"],
            model=model,
            X_train=dataset[config["train_key"]],
            y_train_onehot=dataset["y_train_onehot"],
            X_test=dataset[config["test_key"]],
            y_test=dataset["y_test"],
            epochs=epochs,
            batch_size=batch_size,
            out_dir=out_dir,
        )
        results.append(result)

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_reproducibility(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    set_reproducibility(args.seed)
    ensure_dirs(args.data_dir)

    dataset = prepare_dataset(args.data_dir)
    results = run_all_models(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dir=args.data_dir,
    ).sort_values("f1", ascending=False)

    summarize_and_plot_comparison(results, args.data_dir)

    print("\n===== COMPARAISON FINALE =====")
    print(results)

    results.to_csv(os.path.join(args.data_dir, "ecg200_model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()