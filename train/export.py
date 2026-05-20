import json
from pathlib import Path

import numpy as np


def _flatten_matrix(matrix):
    return matrix.reshape(-1).tolist()


def _transpose_and_flatten(matrix):
    return matrix.T.reshape(-1).tolist()


def export_c_header(params, output_path: str):
    if 'params' in params:
        params = params['params']

    gate_kernel = params['gru_cell']['gates']['kernel']
    candidate_kernel = params['gru_cell']['candidate']['kernel']
    gate_bias = params['gru_cell']['gates']['bias']
    candidate_bias = params['gru_cell']['candidate']['bias']
    output_kernel = params['output_dense']['kernel']
    output_bias = params['output_dense']['bias']

    hidden_size = output_kernel.shape[0]
    input_size = gate_kernel.shape[0] - hidden_size
    if gate_kernel.shape != (input_size + hidden_size, 2 * hidden_size):
        raise ValueError(f"unexpected gate kernel shape: {gate_kernel.shape}")
    if candidate_kernel.shape != (input_size + hidden_size, hidden_size):
        raise ValueError(f"unexpected candidate kernel shape: {candidate_kernel.shape}")

    wir = _transpose_and_flatten(gate_kernel[:input_size, :hidden_size])
    wiz = _transpose_and_flatten(gate_kernel[:input_size, hidden_size:2 * hidden_size])
    whr = _transpose_and_flatten(gate_kernel[input_size:, :hidden_size])
    whz = _transpose_and_flatten(gate_kernel[input_size:, hidden_size:2 * hidden_size])

    win = _transpose_and_flatten(candidate_kernel[:input_size, :hidden_size])
    whn = _transpose_and_flatten(candidate_kernel[input_size:, :hidden_size])

    bir = gate_bias[:hidden_size].tolist()
    biz = gate_bias[hidden_size:2 * hidden_size].tolist()
    bin_bias = candidate_bias.tolist()

    weights = {
        'GRU_WIR': wir,
        'GRU_WIZ': wiz,
        'GRU_WIN': win,
        'GRU_WHR': whr,
        'GRU_WHZ': whz,
        'GRU_WHN': whn,
        'GRU_BIR': bir,
        'GRU_BIZ': biz,
        'GRU_BIN': bin_bias,
        'GRU_BIR_HIDDEN': [0.0] * hidden_size,
        'GRU_BIZ_HIDDEN': [0.0] * hidden_size,
        'GRU_BHN': [0.0] * hidden_size,
        'GRU_WO': output_kernel.reshape(-1).tolist(),
        'GRU_BO': output_bias.tolist(),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write('#pragma once\n')
        handle.write('#include <cstddef>\n\n')
        handle.write(f'constexpr size_t GRU_INPUT_SIZE = {input_size};\n')
        handle.write(f'constexpr size_t GRU_HIDDEN_SIZE = {hidden_size};\n')
        handle.write('constexpr size_t GRU_OUTPUT_SIZE = 1;\n\n')
        for name, values in weights.items():
            size = len(values)
            handle.write(f'static const float {name}[{size}] = {{\n')
            for i, value in enumerate(values):
                handle.write(f'    {value:.8e}f{"," if i + 1 < size else ""}\n')
            handle.write('};\n\n')


def export_json(params, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(params, path.open('w', encoding='utf-8'), indent=2)
