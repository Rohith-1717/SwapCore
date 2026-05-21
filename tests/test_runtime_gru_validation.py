import re
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = ROOT / "native" / "build"
sys.path.insert(0, str(NATIVE_BUILD))


def parse_array(header: str, name: str) -> np.ndarray:
    pattern = rf"static const float {name}\[[^\]]+\] = \{{(.*?)\}};"
    match = re.search(pattern, header, flags=re.S)
    if not match:
        raise KeyError(f"could not find {name} in runtime_gru_params.hpp")
    values = re.findall(r"[-+]?\d+\.\d+e[-+]?\d+f?|[-+]?\d+\.\d+f?|[-+]?\d+f?", match.group(1))
    return np.asarray([float(value.rstrip("f")) for value in values], dtype=np.float32)


def load_params():
    header = (ROOT / "native" / "runtime_gru_params.hpp").read_text(encoding="utf-8")
    input_size = int(re.search(r"GRU_INPUT_SIZE = (\d+)", header).group(1))
    hidden_size = int(re.search(r"GRU_HIDDEN_SIZE = (\d+)", header).group(1))
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "wir": parse_array(header, "GRU_WIR").reshape(hidden_size, input_size),
        "wiz": parse_array(header, "GRU_WIZ").reshape(hidden_size, input_size),
        "win": parse_array(header, "GRU_WIN").reshape(hidden_size, input_size),
        "whr": parse_array(header, "GRU_WHR").reshape(hidden_size, hidden_size),
        "whz": parse_array(header, "GRU_WHZ").reshape(hidden_size, hidden_size),
        "whn": parse_array(header, "GRU_WHN").reshape(hidden_size, hidden_size),
        "bir": parse_array(header, "GRU_BIR"),
        "biz": parse_array(header, "GRU_BIZ"),
        "bin": parse_array(header, "GRU_BIN"),
        "bir_hidden": parse_array(header, "GRU_BIR_HIDDEN"),
        "biz_hidden": parse_array(header, "GRU_BIZ_HIDDEN"),
        "bhn": parse_array(header, "GRU_BHN"),
        "wo": parse_array(header, "GRU_WO"),
        "bo": parse_array(header, "GRU_BO"),
    }


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def python_gru_logit(sequence: np.ndarray, params) -> float:
    hidden = jnp.zeros((params["hidden_size"],), dtype=jnp.float32)
    for timestep in range(sequence.shape[0]):
        x = jnp.asarray(sequence[timestep], dtype=jnp.float32)
        reset = sigmoid(params["wir"] @ x + params["whr"] @ hidden + params["bir"] + params["bir_hidden"])
        update = sigmoid(params["wiz"] @ x + params["whz"] @ hidden + params["biz"] + params["biz_hidden"])
        candidate = jnp.tanh(params["win"] @ x + params["whn"] @ (reset * hidden) + params["bin"] + params["bhn"])
        hidden = (1.0 - update) * candidate + update * hidden
    return float(params["wo"] @ hidden + params["bo"][0])


def deterministic_sequence(seq_len: int = 32, feature_dim: int = 5) -> np.ndarray:
    rng = np.random.default_rng(20260521)
    sequence = rng.normal(loc=0.0, scale=0.25, size=(seq_len, feature_dim)).astype(np.float32)
    sequence[:, 3] = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    sequence[:, 2] = (np.arange(seq_len) % 3 == 0).astype(np.float32)
    return np.ascontiguousarray(sequence)


def main():
    import swapcore_native as scn

    params = load_params()
    sequence = deterministic_sequence(32, params["input_size"])
    python_output = python_gru_logit(sequence, params)
    cpp_output = scn.RuntimeGRU().predictLogit(sequence.reshape(-1).astype(np.float32).tolist(), sequence.shape[0])
    diff = abs(python_output - cpp_output)

    print(f"Python output: {python_output:.9f}")
    print(f"C++ output: {cpp_output:.9f}")
    print(f"Absolute difference: {diff:.9e}")
    assert diff < 1e-4, f"runtime GRU mismatch: {diff}"


if __name__ == "__main__":
    main()
