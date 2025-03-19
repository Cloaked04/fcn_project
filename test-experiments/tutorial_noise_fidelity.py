"""
Demonstrates quantum noise effects using NetSquid's depolarizing noise model.

This script creates a qubit in the |0> state, applies depolarizing noise with a 
5% probability, and calculates the resulting fidelity compared to the ideal state.
A basic script to test how quantum information degrades under noise.
"""

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates  # Import ketstates directly
from netsquid.components.models.qerrormodels import DepolarNoiseModel


def tutorial_noise_and_fidelity():
    # 1. Create one qubit in state |0>
    qubits = qapi.create_qubits(1)
    ideal_state = ketstates.s0  # |0> state

    # 2. Define a depolarizing noise model with a 5% depolarizing rate
    noise_model = DepolarNoiseModel(depolar_rate=0.05)

    # 3. Apply the noise model using the correct function
    qapi.depolarize(qubits[0], prob=noise_model.depolar_rate)

    # 4. Compute fidelity with respect to |0>
    # In NetSquid, density matrices are accessed differently
    # We can compute fidelity directly using the qubitapi
    F = qapi.fidelity(qubits[0], ketstates.s0)
    print(f"Fidelity with |0> after depolarizing noise: {F:.3f}")

if __name__ == "__main__":
    tutorial_noise_and_fidelity()
