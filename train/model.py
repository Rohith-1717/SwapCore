from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class GRUCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry: jnp.ndarray, x: jnp.ndarray) -> Any:
        features = jnp.concatenate([x, carry], axis=-1)
        gates = nn.Dense(2 * self.hidden_size, name='gates')(features)
        reset_gate, update_gate = jnp.split(gates, 2, axis=-1)
        reset_gate = nn.sigmoid(reset_gate)
        update_gate = nn.sigmoid(update_gate)

        candidate_input = jnp.concatenate([x, reset_gate * carry], axis=-1)
        candidate = nn.tanh(nn.Dense(self.hidden_size, name='candidate')(candidate_input))

        new_carry = (1.0 - update_gate) * candidate + update_gate * carry
        return new_carry, new_carry


class GRUModel(nn.Module):
    hidden_size: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"GRUModel expects (batch, seq_len, features), got {x.shape}")

        batch_size = x.shape[0]
        carry = jnp.zeros((batch_size, self.hidden_size))
        cell = GRUCell(self.hidden_size, name='gru_cell')

        for timestep in range(x.shape[1]):
            carry, _ = cell(carry, x[:, timestep, :])

        logits = nn.Dense(1, name='output_dense')(carry)
        return logits.squeeze(-1)
