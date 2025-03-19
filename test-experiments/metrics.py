"""
Quantum Error Correction (QEC) Performance Metrics

This script simulates and compares the performance of a three-qubit repetition code
against unprotected qubits in the presence of depolarizing noise. 

The three-qubit repetition code is a basic quantum error correction technique that
encodes a single logical qubit into three physical qubits. It can detect and correct
single bit-flip errors through majority voting.

The script:
1. Runs multiple trials with and without QEC protection
2. Applies depolarizing noise to simulate quantum channel errors
3. Measures the fidelity between the final state and the ideal |0⟩ state
4. Compares average fidelities and generates distribution histograms
5. Saves the results as a plot in the current directory

Parameters:
    num_runs (int): Number of simulation trials to perform
    noise_rate (float): Depolarization probability (0.0 to 1.0)

Output:
    - Prints average fidelity statistics to console
    - Saves histogram plot as 'qec_fidelity_comparison.png' in the current directory
"""

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates
from netsquid.components.models.qerrormodels import DepolarNoiseModel
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_metrics_logging(num_runs=50, noise_rate=0.05):
    """
    Demonstrates collecting fidelity data across multiple trials
    for a single qubit with vs. without 3-qubit repetition code.
    
    Args:
        num_runs (int): Number of simulation runs to perform
        noise_rate (float): Probability of depolarizing noise (0.0 to 1.0)
        
    Returns:
        None: Results are printed to console and saved as a plot file
    """
    fidelities_no_qec = []
    fidelities_qec = []

    # Define a simple depolarizing noise model
    noise_model = DepolarNoiseModel(depolar_rate=noise_rate)

    # A helper function to measure fidelity w.r.t. |0>
    def measure_fidelity_zero(qubit):
        return qapi.fidelity(qubit, ketstates.s0)

    for _ in range(num_runs):
        # --- NO QEC CASE ---
        qubit_noqec = qapi.create_qubits(1)
        # Apply depolarizing noise correctly
        qapi.depolarize(qubit_noqec[0], prob=noise_model.depolar_rate)
        f_noqec = measure_fidelity_zero(qubit_noqec[0])
        fidelities_no_qec.append(f_noqec)

        # --- QEC CASE (3-qubit repetition) ---
        qubits_qec = qapi.create_qubits(3)
        # Encode using proper CNOT syntax
        qapi.operate([qubits_qec[0], qubits_qec[1]], ns.CNOT)
        qapi.operate([qubits_qec[0], qubits_qec[2]], ns.CNOT)

        # Apply noise to all 3 qubits individually
        for qubit in qubits_qec:
            qapi.depolarize(qubit, prob=noise_model.depolar_rate)

        # Decode: measure all qubits in computational basis + majority vote
        m0 = qapi.measure(qubits_qec[0])[0]
        m1 = qapi.measure(qubits_qec[1])[0]
        m2 = qapi.measure(qubits_qec[2])[0]
        # Reconstruct a single qubit in the same basis as the measured logical bit
        # We'll define "decoded qubit" = |0> if majority is 0, else |1>
        majority = m0 + m1 + m2
        decoded_bit = 0 if majority < 2 else 1
        # Let's create a single qubit in that state for a fidelity check:
        final_qubit = qapi.create_qubits(1)
        if decoded_bit == 1:
            qapi.operate(final_qubit[0], ns.X)

        f_qec = measure_fidelity_zero(final_qubit[0])
        fidelities_qec.append(f_qec)

    # Log average results
    avg_no_qec = np.mean(fidelities_no_qec)
    avg_qec = np.mean(fidelities_qec)
    print(f"Average fidelity WITHOUT QEC: {avg_no_qec:.3f}")
    print(f"Average fidelity WITH QEC:    {avg_qec:.3f}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fidelities_no_qec, bins=10, alpha=0.5, label="No QEC")
    plt.hist(fidelities_qec, bins=10, alpha=0.5, label="3-qubit QEC")
    plt.title(f"Fidelity Distribution (Noise Rate = {noise_rate:.2f})")
    plt.xlabel("Fidelity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot instead of displaying it
    filename = f"qec_fidelity_comparison_noise{noise_rate:.2f}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")


if __name__ == "__main__":
    run_metrics_logging(num_runs=200, noise_rate=0.10)
