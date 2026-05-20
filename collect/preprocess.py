import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

FEATURE_KEYS = ['vpn', 'delta', 'access', 'ts', 'phase']


def load_trace(path: str):
    path_obj = Path(path)
    if path_obj.suffix == '.npz':
        data = np.load(path_obj)
        return {key: data[key] for key in data.files}

    with path_obj.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def compute_future_reuse_labels(traces: List[Dict[str, Any]]) -> List[bool]:
    seen_vpns = set()
    labels = [False] * len(traces)
    for index in range(len(traces) - 1, -1, -1):
        vpn = traces[index]['vpn']
        labels[index] = vpn in seen_vpns
        seen_vpns.add(vpn)
    return labels


def compute_future_reuse_labels_array(vpns: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(vpns), dtype=np.float32)
    seen_vpns = set()
    for index in range(len(vpns) - 1, -1, -1):
        vpn = int(vpns[index])
        labels[index] = 1.0 if vpn in seen_vpns else 0.0
        seen_vpns.add(vpn)
    return labels


def preprocess(traces) -> Dict[str, Any]:
    if isinstance(traces, dict):
        return preprocess_arrays(traces)

    dataset = {k: [] for k in FEATURE_KEYS}
    labels = []
    has_labels = any('future_reuse' in record for record in traces)
    generated_labels = compute_future_reuse_labels(traces) if not has_labels else None

    for idx, record in enumerate(traces):
        dataset['vpn'].append(float(record['vpn']))
        dataset['access_delta'].append(float(record['access_delta']))
        dataset['access_type'].append(1.0 if record['access_type'] == 'write' else 0.0)
        dataset['timestamp'].append(float(record['timestamp']))
        dataset['reuse_distance'].append(float(record['reuse_distance']))
        if has_labels:
            labels.append(1.0 if record.get('future_reuse', False) else 0.0)
        else:
            labels.append(1.0 if generated_labels[idx] else 0.0)

    return {'features': [dataset[k] for k in FEATURE_KEYS], 'labels': labels}


def preprocess_arrays(trace: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    access_type = trace['access'] if 'access' in trace else trace['access_type']
    if access_type.dtype.kind in {'U', 'S', 'O'}:
        access_type = np.array([1.0 if value == 'write' else 0.0 for value in access_type], dtype=np.float32)
    else:
        access_type = access_type.astype(np.float32)

    delta = trace['delta'] if 'delta' in trace else trace['access_delta']
    timestamp = trace['ts'] if 'ts' in trace else trace['timestamp']
    phase = trace['phase'] if 'phase' in trace else np.zeros(len(trace['vpn']), dtype=np.int8)

    features = np.vstack([
        trace['vpn'].astype(np.float32),
        delta.astype(np.float32),
        access_type,
        timestamp.astype(np.float32),
        phase.astype(np.float32),
    ])

    if 'future_reuse' in trace:
        labels = trace['future_reuse'].astype(np.float32)
    else:
        labels = compute_future_reuse_labels_array(trace['vpn'])

    return {'features': features, 'labels': labels}


def make_sequences(features, labels, window_size: int = 32, stride: int = 1):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.shape[0] == len(FEATURE_KEYS):
        features = features.T
    if len(features) < window_size:
        raise ValueError('trace is shorter than window_size')

    starts = np.arange(0, len(features) - window_size + 1, stride)
    sequences = np.stack([features[start:start + window_size] for start in starts])
    sequence_labels = labels[starts + window_size - 1]
    return sequences, sequence_labels


def train_validation_split(features, labels, validation_ratio: float = 0.2):
    split = int(len(labels) * (1.0 - validation_ratio))
    return features[:split], labels[:split], features[split:], labels[split:]


def save_numpy(output_path: str, features, labels):
    import numpy as np
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=labels)


def load_numpy(path: str):
    import numpy as np
    data = np.load(path)
    return data['features'], data['labels']
