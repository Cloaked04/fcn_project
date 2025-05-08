# qec_single_pauli_tests.py
import sys
from netsquid.qubits.qubitapi import operate
from netsquid.qubits import operators as ops

from single_qubit_experiments import QuantumErrorCorrectionExperiment

# ------------------------------------------------------------
# which codes to exercise and which Pauli errors they must fix
# ------------------------------------------------------------
TEST_MATRIX = {
    "three_qubit_bit_flip":   ["X"],
    "three_qubit_phase_flip": ["Z"],
    "shor_nine":              ["X", "Y", "Z"],
}

BASIS_STATES = ["0", "1", "+", "-"]
TOL          = 1e-6


def run_suite():
    results = {}
    for code, paulis in TEST_MATRIX.items():
        total = failures = 0

        # discover block length once
        dummy_exp = QuantumErrorCorrectionExperiment(
            initial_state="0", qec_method=code, error_type="bit_flip")
        block_len = len(dummy_exp.encode_qubit(dummy_exp.prepare_qubit()))

        for init in BASIS_STATES:
            for p in paulis:
                gate = getattr(ops, p)
                for phys_idx in range(block_len):
                    total += 1

                    exp = QuantumErrorCorrectionExperiment(
                        initial_state=init,
                        qec_method=code,
                        error_type="bit_flip")   # error model unused here

                    # encode logical qubit
                    logical = exp.prepare_qubit()
                    block   = exp.encode_qubit(logical)

                    # inject exactly one Pauli error
                    operate(block[phys_idx], gate)

                    # decode
                    decoded = exp.decode_qubit(block)

                    # fidelity vs initial state
                    fid = exp.calculate_fidelity(decoded)
                    if abs(fid - 1.0) > TOL:
                        failures += 1

        results[code] = (total, failures)
    return results


def print_table(res):
    print("\nQEC Method           | Tests | Failures | Pass %")
    print("-" * 44)
    ok = True
    for code, (tot, fail) in res.items():
        pct = 100.0 * (tot - fail) / tot if tot else 0.0
        print(f"{code:20s} | {tot:5d} | {fail:8d} | {pct:6.2f}%")
        if fail:
            ok = False
    if ok:
        print("\nAll single-Pauli tests passed!")
    else:
        print("\nOne or more codes failed at least one test.")
        sys.exit(1)


if __name__ == "__main__":
    print_table(run_suite())
