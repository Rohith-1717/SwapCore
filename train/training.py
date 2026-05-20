import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from collect.preprocess import load_trace, make_sequences, preprocess
from .model import GRUModel


DEFAULT_SEQ_LEN = 32
DEFAULT_FEATURE_DIM = 5
DEFAULT_HIDDEN_SIZE = 32


class TrainState(train_state.TrainState):
    pass


def create_train_state(
    rng,
    learning_rate: float = 1e-3,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    feature_dim: int = DEFAULT_FEATURE_DIM,
):
    model = GRUModel(hidden_size=hidden_size)
    variables = model.init(rng, jnp.zeros((1, seq_len, feature_dim), dtype=jnp.float32))
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def loss_fn(apply_fn, params, batch, labels):
    logits = apply_fn({"params": params}, batch)
    assert logits.shape == labels.shape, f"logits shape {logits.shape} != labels shape {labels.shape}"
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def train_epoch(state, features, labels, batch_size: int = 256, rng: np.random.Generator | None = None):
    assert features.ndim == 3, f"features must be (batch, seq_len, feature_dim), got {features.shape}"
    assert labels.ndim == 1, f"labels must be (batch,), got {labels.shape}"
    assert features.shape[0] == labels.shape[0], f"features/labels length mismatch: {features.shape[0]} vs {labels.shape[0]}"

    data_size = features.shape[0]
    rng = rng or np.random.default_rng()
    perms = rng.permutation(data_size)
    losses = []

    for start in range(0, data_size, batch_size):
        idx = perms[start:start + batch_size]
        batch = jnp.asarray(features[idx], dtype=jnp.float32)
        label_batch = jnp.asarray(labels[idx], dtype=jnp.float32)

        def batch_loss(params):
            return loss_fn(state.apply_fn, params, batch, label_batch)

        loss, grads = jax.value_and_grad(batch_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        losses.append(float(loss))

    return state, float(np.mean(losses))


def load_dataset(path: str, seq_len: int = DEFAULT_SEQ_LEN, stride: int = 1, max_accesses: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)

    if "features" in data and "labels" in data:
        features = data["features"]
        labels = data["labels"].astype(np.float32)
        if features.ndim == 3:
            return features.astype(np.float32), labels
        sequences, sequence_labels = make_sequences(features, labels, window_size=seq_len, stride=stride)
        return sequences.astype(np.float32), sequence_labels.astype(np.float32)

    trace = load_trace(path)
    if max_accesses is not None:
        trace = {key: value[:max_accesses] for key, value in trace.items()}
    processed = preprocess(trace)
    sequences, sequence_labels = make_sequences(processed["features"], processed["labels"], window_size=seq_len, stride=stride)
    return sequences.astype(np.float32), sequence_labels.astype(np.float32)


def validate_shapes(features, labels, hidden_size: int = DEFAULT_HIDDEN_SIZE):
    assert features.ndim == 3, f"expected sequence features (batch, seq_len, feature_dim), got {features.shape}"
    assert labels.ndim == 1, f"expected labels (batch,), got {labels.shape}"
    assert features.shape[0] == labels.shape[0], f"batch mismatch: {features.shape[0]} vs {labels.shape[0]}"

    rng = jax.random.PRNGKey(0)
    model = GRUModel(hidden_size=hidden_size)
    params = model.init(rng, jnp.asarray(features[:1], dtype=jnp.float32))
    logits = model.apply(params, jnp.asarray(features[: min(4, len(features))], dtype=jnp.float32))
    assert logits.shape == (min(4, len(features)),), f"expected logits shape (batch,), got {logits.shape}"
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"model output shape: {logits.shape}")


def save_params(params, output_path: str):
    state_dict = serialization.to_state_dict(params)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, state_dict, allow_pickle=True)


def train_from_file(
    data_path: str,
    output_path: str,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = 1,
    max_accesses: int | None = None,
):
    features, labels = load_dataset(data_path, seq_len=seq_len, stride=stride, max_accesses=max_accesses)
    validate_shapes(features, labels, hidden_size=hidden_size)

    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        rng,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        seq_len=features.shape[1],
        feature_dim=features.shape[2],
    )
    np_rng = np.random.default_rng(0)

    for epoch in range(epochs):
        start = time.time()
        state, mean_loss = train_epoch(state, features, labels, batch_size=batch_size, rng=np_rng)
        duration = time.time() - start
        print(f"Epoch {epoch + 1}/{epochs} loss={mean_loss:.6f} duration={duration:.2f}s")

    save_params(state.params, output_path)
    return state.params
