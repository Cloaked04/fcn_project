"""
Quantum Error Correction: Three-Qubit Repetition Code Implementation

This script demonstrates a fundamental quantum error correction technique known as the 
three-qubit repetition code, which protects against single bit-flip errors. 

The code works as follows:
1. Creates three physical qubits, with the first one representing our logical qubit
2. Encodes the logical qubit state across all three physical qubits using CNOT gates
   (|0⟩ → |000⟩ or |1⟩ → |111⟩)
3. Simulates a random bit-flip error on one of the qubits with 20% probability
4. Decodes the state by measuring all qubits and performing majority voting
5. Reports whether the error correction succeeded in recovering the original logical state

This implementation uses NetSquid's quantum simulation framework to accurately model
quantum operations and measurements. The three-qubit repetition code can only correct
bit-flip (X) errors, not phase-flip (Z) errors or more complex errors.
"""


import random
import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.operators import X

def encode_3qubit_repetition(qubits):
    """
    Encodes the logical qubit in qubits[0]
    across qubits[0], qubits[1], qubits[2].
    Logical |0> -> |000>, Logical |1> -> |111>.
    """
    # CNOT(q0 -> q1): where q0 is control and q1 is target
    qapi.operate([qubits[0], qubits[1]], ns.CNOT)
    # CNOT(q0 -> q2): where q0 is control and q2 is target
    qapi.operate([qubits[0], qubits[2]], ns.CNOT)

def decode_3qubit_repetition(qubits):
    """
    Decodes 3 physical qubits by measuring all in the computational basis
    and doing majority voting.
    Returns '0' or '1' as the decoded logical bit.
    """
    m0 = qapi.measure(qubits[0])[0]
    m1 = qapi.measure(qubits[1])[0]
    m2 = qapi.measure(qubits[2])[0]
    # Majority vote
    s = m0 + m1 + m2  # sum of measured bits
    logical_bit = 0 if s < 2 else 1
    return logical_bit

def run_3qubit_repetition_demo():
    # 1. Create 3 qubits for repetition code (assume logical qubit is |0>)
    qubits = qapi.create_qubits(3)
    # We'll treat qubits[0] as the "control" for encoding the logical state.

    # 2. (Optional) If you want to encode a logical |1>, you can flip qubits[0] first:
    # qapi.operate(qubits[0], X)

    # 3. Encode
    encode_3qubit_repetition(qubits)

    # 4. Introduce random bit-flip error on exactly one qubit 20% of the time
    # or no error if random >= 0.2
    if random.random() < 0.2:
        error_qubit = random.choice([0, 1, 2])
        qapi.operate(qubits[error_qubit], X)
        print(f"Bit-flip introduced on qubit index {error_qubit}")

    # 5. Decode
    decoded_bit = decode_3qubit_repetition(qubits)
    print(f"Decoded logical bit: {decoded_bit}")

    # 6. Compare with the original logical bit (0):
    result = "SUCCESS" if decoded_bit == 0 else "FAILURE"
    print(f"Result of QEC: {result}")

if __name__ == "__main__":
    run_3qubit_repetition_demo()
