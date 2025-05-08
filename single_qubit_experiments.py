import netsquid as ns
import numpy as np
from netsquid.nodes import Node
from netsquid.components import QuantumChannel, ClassicalChannel
from netsquid.nodes import DirectConnection
from netsquid.qubits.qubitapi import *
from netsquid.qubits import operators as ops
from netsquid.components.models.qerrormodels import *
from netsquid.qubits import dmtools

from math import pi

class QuantumErrorCorrectionExperiment:
    """
    A class to run experiments testing different quantum error correction methods
    using single qubits over a network from node A to node B.
    """
    
    def __init__(self, 
                 initial_state="0", 
                 error_type="bit_flip",
                 error_params=None,
                 qec_method="three_qubit_bit_flip",
                 distance=1.0):
        """
        Initialize the experiment.
        
        Parameters:
        _________________________

        initial_state : str; default: 0.
            The initial state of the qubit before sending over the network.
            Options: "0", "1", "+", "-". The qubit can be compute

        error_type : str; default: bit_flip.
            The type of error to apply. Options: "bit_flip", "phase_flip", "bit_phase_flip",
            "depolarizing", "amplitude_damping", "phase_damping", "gen_amplitude_damping",
            "coherent_overrotation", "mixed_xz"

        error_params : dict
            Parameters for the error model. Here you can passing in the probabilities or other parameters that
            determine how, when or with what probability the error is applied. Required keys depend on error_type.

        qec_method : str
            The quantum error correction method to use. 
            Options: "three_qubit_bit_flip", "three_qubit_phase_flip", "shor_nine", "steane_seven",
             "none"

        distance : float
            Distance between nodes in km.
        """
        self.initial_state = initial_state
        self.error_type = error_type
        self.error_params = error_params if error_params is not None else {}
        self.qec_method = qec_method
        self.distance = distance
        
        # Initialize the network
        self._setup_network()
        
        print(f"Initialized QEC experiment with:")
        print(f"  Initial state: {initial_state}")
        print(f"  Error type: {error_type}")
        print(f"  Error parameters: {error_params}")
        print(f"  QEC method: {qec_method}")

    
    def _setup_network(self):
        """
        Private function.
        Sets up the quantum network with two nodes A and B.

        The whole process goes like follows:
            - create nodes
            - initialize quantum channels A -> B, B->A
            - connect the channels so that the network has the capability of sending information back and forth
            - connect the nodes: node A to node B.
        """
        # Create nodes
        self.node_a = Node("NodeA")
        self.node_b = Node("NodeB")
        
        # Create quantum and classical channels
        qchannel_a_to_b = QuantumChannel("QChannel_A2B", length=self.distance)
        qchannel_b_to_a = QuantumChannel("QChannel_B2A", length=self.distance)
        cchannel_a_to_b = ClassicalChannel("CChannel_A2B", length=self.distance)
        cchannel_b_to_a = ClassicalChannel("CChannel_B2A", length=self.distance)
        
        # Create connections
        qconnection = DirectConnection("QConn_A2B", 
                                       channel_AtoB=qchannel_a_to_b,
                                       channel_BtoA=qchannel_b_to_a)
        cconnection = DirectConnection("CConn_A2B",
                                       channel_AtoB=cchannel_a_to_b,
                                       channel_BtoA=cchannel_b_to_a)
        
        # Connect nodes
        self.node_a.connect_to(self.node_b, qconnection, local_port_name="qubit_port", 
                               remote_port_name="qubit_port")
        self.node_a.connect_to(self.node_b, cconnection, local_port_name="classical_port", 
                               remote_port_name="classical_port")
        
        print("Network setup complete with quantum and classical connections between nodes.")


    def prepare_qubit(self):
        """
        Prepare the initial qubit state as specified.
        
        Returns:
        --------
        Qubit: The prepared qubit in desired state.
        """
        # Create a qubit in the |0⟩ state
        qubit, = create_qubits(1)
        
        # Prepare the specified state
        if self.initial_state == "0":
            # Already in |0⟩ state
            pass
        elif self.initial_state == "1":
            #Apply X gate to change 0 to 1.
            operate(qubit, ops.X)
        elif self.initial_state == "+":
            #Apply hadamard gate to O to get +.
            operate(qubit, ops.H)
        elif self.initial_state == "-":
            #Apply X and then H gate to 0 to get -.
            operate(qubit, ops.X)
            operate(qubit, ops.H)
        
        print(f"Qubit prepared in {self.initial_state} state")
        return qubit


    def apply_error(self, qubits):
        """
        Apply the specified error to the qubits.
        
        Parameters:
        -----------
        qubits : list
            List of qubits to apply the error to
        """
        if self.error_type == "bit_flip":
            p = self.error_params.get("probability", 0.1)
            for qubit in qubits:
                if np.random.random() < p:
                    operate(qubit, ops.X)
            print(f"Applied bit-flip error with probability {p}")
        
        elif self.error_type == "phase_flip":
            p = self.error_params.get("probability", 0.1)
            for qubit in qubits:
                if np.random.random() < p:
                    operate(qubit, ops.Z)
            print(f"Applied phase-flip error with probability {p}")
        
        elif self.error_type == "bit_phase_flip":
            p = self.error_params.get("probability", 0.1)
            for qubit in qubits:
                if np.random.random() < p:
                    operate(qubit, ops.Y)
            print(f"Applied bit-phase-flip error with probability {p}")
        
        elif self.error_type == "depolarizing":
            p = self.error_params.get("probability", 0.1)
            for qubit in qubits:
                r = np.random.random()
                
                if r < p:  # Apply error with full probability p
                    if r < p/3:
                        operate(qubit, ops.X)
                    elif r < 2*p/3:
                        operate(qubit, ops.Y)
                    else:
                        operate(qubit, ops.Z)
                        
            print(f"Applied depolarizing error with probability {p}")
        
        elif self.error_type == "amplitude_damping":
            gamma = self.error_params.get("gamma", 0.1)
            for qubit in qubits:
                amplitude_dampen(qubit, gamma)
            print(f"Applied amplitude damping with gamma={gamma}")
        
        elif self.error_type == "phase_damping":
            gamma = self.error_params.get("gamma", 0.1)
            for qubit in qubits:
                dephase(qubit, gamma)
            print(f"Applied phase damping with gamma={gamma}")
        
        elif self.error_type == "gen_amplitude_damping":
            gamma = self.error_params.get("gamma", 0.1)
            p = self.error_params.get("probability", 0.5)
            for qubit in qubits:
                amplitude_dampen(qubit, gamma, prob=p)
            print(f"Applied generalized amplitude damping with gamma={gamma}, p={p}")
        
        elif self.error_type == "coherent_overrotation":
            theta = self.error_params.get("theta", 0.1)
            axis = self.error_params.get("axis", "X")
            
            if axis == "X":
                rotation = ops.create_rotation_op(theta, (1, 0, 0))
            elif axis == "Y":
                rotation = ops.create_rotation_op(theta, (0, 1, 0))
            elif axis == "Z":
                rotation = ops.create_rotation_op(theta, (0, 0, 1))
            
            for qubit in qubits:
                operate(qubit, rotation)
            print(f"Applied coherent overrotation around {axis}-axis by angle {theta}")
        
        elif self.error_type == "mixed_xz":
            p_x = self.error_params.get("p_x", 0.1)
            p_z = self.error_params.get("p_z", 0.1)
            
            for qubit in qubits:
                # Apply X error with probability p_x
                if np.random.random() < p_x:
                    operate(qubit, ops.X)
                
                # Apply Z error with probability p_z (independent of X)
                if np.random.random() < p_z:
                    operate(qubit, ops.Z)
            
            print(f"Applied mixed X-Z error with p_x={p_x}, p_z={p_z}")


    def encode_qubit(self, qubit):
        """
        Encode the qubit using the specified error correction method.
        
        Parameters:
       
            qubit : Qubit
                The qubit to encode
        
        Returns:
        
            list: List of encoded qubits
        """
        if self.qec_method == "none":
            print("No encoding (QEC disabled)")
            return [qubit]
        
        elif self.qec_method == "three_qubit_bit_flip":
            # Create two additional qubits in |0⟩ state
            anc1, anc2 = create_qubits(2)
            
            # Apply CNOT gates to entangle qubits
            operate([qubit, anc1], ops.CNOT)
            operate([qubit, anc2], ops.CNOT)
            
            print("Encoded using 3-qubit bit-flip code")
            return [qubit, anc1, anc2]
        
        elif self.qec_method == "three_qubit_phase_flip":
            # Create two additional qubits in |0⟩ state
            anc1, anc2 = create_qubits(2)

            # Apply CNOT gates to entangle qubits
            operate([qubit, anc1], ops.CNOT)
            operate([qubit, anc2], ops.CNOT)
            
            # Apply Hadamard gates to all qubits
            operate(qubit, ops.H)
            operate(anc1, ops.H)
            operate(anc2, ops.H)
            
            
            
            print("Encoded using 3-qubit phase-flip code")
            return [qubit, anc1, anc2]
        
        
        elif self.qec_method == "shor_nine":
            # Create eight additional qubits in |0⟩ state
            ancillas = create_qubits(8)
            
            operate([qubit, ancillas[2]], ops.CNOT)
            operate([qubit, ancillas[5]], ops.CNOT)
            operate(qubit, ops.H)
            operate(ancillas[2], ops.H)
            operate(ancillas[5], ops.H)
            operate([qubit, ancillas[0]], ops.CNOT)
            operate([qubit, ancillas[1]], ops.CNOT)
            operate([ancillas[2], ancillas[3]], ops.CNOT)
            operate([ancillas[2], ancillas[4]], ops.CNOT)
            operate([ancillas[5], ancillas[6]], ops.CNOT)
            operate([ancillas[5], ancillas[7]], ops.CNOT)

            return [qubit] + ancillas

                
        # elif self.qec_method == "steane_seven":
        #     # Create six additional qubits in |0⟩ state
        #     ancillas = create_qubits(6)
            
        #     # Prepare all ancillas in |+⟩ state
        #     for anc in ancillas:
        #         operate(anc, ops.H)
            
        #     # Apply CNOT gates to create the X-type stabilizer state
        #     operate([ancillas[0], ancillas[2]], ops.CNOT)
        #     operate([ancillas[1], ancillas[2]], ops.CNOT)
        #     operate([ancillas[0], ancillas[4]], ops.CNOT)
        #     operate([ancillas[3], ancillas[4]], ops.CNOT)
        #     operate([ancillas[1], ancillas[5]], ops.CNOT)
        #     operate([ancillas[3], ancillas[5]], ops.CNOT)
            
        #     # Encode the data qubit
        #     for anc in ancillas:
        #         operate([qubit, anc], ops.CNOT)
            
        #     # Apply Z-type half (global H, repeat pattern, undo H)
        #     for q in [qubit] + list(ancillas):
        #         operate(q, ops.H)
            
        #     operate([ancillas[0], ancillas[2]], ops.CNOT)
        #     operate([ancillas[1], ancillas[2]], ops.CNOT)
        #     operate([ancillas[0], ancillas[4]], ops.CNOT)
        #     operate([ancillas[3], ancillas[4]], ops.CNOT)
        #     operate([ancillas[1], ancillas[5]], ops.CNOT)
        #     operate([ancillas[3], ancillas[5]], ops.CNOT)
            
        #     for anc in ancillas:
        #         operate([qubit, anc], ops.CNOT)
            
        #     for q in [qubit] + list(ancillas):
        #         operate(q, ops.H)
            
        #     print("Encoded using Steane's [7,1,3] code")
        #     return [qubit] + list(ancillas)


    def decode_qubit(self, qubits):
        """
        Decode the encoded qubits and correct errors if possible.
        
        Parameters:
        -----------
        qubits : list
            List of encoded qubits
        
        Returns:
        --------
        Qubit: The decoded qubit
        """
        if self.qec_method == "none":
            print("No decoding (QEC disabled)")
            return qubits[0]
        
        elif self.qec_method == "three_qubit_bit_flip":
            # Perform syndrome measurement
            anc1, anc2 = create_qubits(2)
            
            # Compute parity between qubits 0 and 1
            operate([qubits[0], anc1], ops.CNOT)
            operate([qubits[1], anc1], ops.CNOT)
            
            # Compute parity between qubits 0 and 2
            operate([qubits[0], anc2], ops.CNOT)
            operate([qubits[2], anc2], ops.CNOT)
            
            # Measure ancillas to get syndrome
            s1, _ = measure(anc1)
            s2, _ = measure(anc2)
            
            print(f"Syndrome measurements: {s1}, {s2}")
            
            # Apply correction based on syndrome
            if s1 == 1 and s2 == 0:
                operate(qubits[1], ops.X)
                print("Corrected bit-flip error on qubit 1")
            elif s1 == 0 and s2 == 1:
                operate(qubits[2], ops.X)
                print("Corrected bit-flip error on qubit 2")
            elif s1 == 1 and s2 == 1:
                operate(qubits[0], ops.X)
                print("Corrected bit-flip error on qubit 0")
            else:
                print("No bit-flip error detected")

            operate([qubits[0], qubits[2]], ops.CNOT)
            operate([qubits[0], qubits[1]], ops.CNOT)
            #return qubits[0]
                
            print("Decoded using 3-qubit bit-flip code")
            return qubits[0]
        
        elif self.qec_method == "three_qubit_phase_flip":
            # Implement the decode circuit

            for q in qubits:
                operate(q, ops.H)

                
            
            anc1, anc2 = create_qubits(2)
            
            operate([qubits[0], anc1], ops.CNOT)
            operate([qubits[1], anc1], ops.CNOT)
            operate([qubits[1], anc2], ops.CNOT)
            operate([qubits[2], anc2], ops.CNOT)

            for q in qubits:
                operate(q, ops.H)


            
            # Measure ancillas to get syndrome
            s1, _ = measure(anc1)
            s2, _ = measure(anc2)
            
            print(f"Syndrome measurements: {s1}, {s2}")
            
            # Apply correction based on syndrome
            if s1 == 1 and s2 == 0:
                operate(qubits[0], ops.Z)
                print("Corrected phase-flip error on qubit 1")
            elif s1 == 0 and s2 == 1:
                operate(qubits[2], ops.Z)
                print("Corrected phase-flip error on qubit 2")
            elif s1 == 1 and s2 == 1:
                operate(qubits[1], ops.Z)
                print("Corrected phase-flip error on qubit 0")
            else:
                print("No phase-flip error detected")
            
            # Transform back to Z basis
            for q in qubits:
                operate(q, ops.H)

            operate([qubits[0], qubits[2]], ops.CNOT)
            operate([qubits[0], qubits[1]], ops.CNOT)
            # for q in qubits:
            #     operate(q, ops.H)                 # undo first H
                
            print("Decoded using 3-qubit phase-flip code")
            return qubits[0]
            
            
        elif self.qec_method == "shor_nine":
            #Decoding circuit for Shor's

            operate([qubits[0], qubits[1]], ops.CNOT)
            operate([qubits[0], qubits[2]], ops.CNOT)
            operate([qubits[2], qubits[1], qubits[0]], ops.CCX)
            operate([qubits[3], qubits[4]], ops.CNOT)
            operate([qubits[3], qubits[5]], ops.CNOT)
            operate([qubits[5], qubits[4], qubits[3]], ops.CCX)
            operate([qubits[6], qubits[7]], ops.CNOT)
            operate([qubits[6], qubits[8]], ops.CNOT)
            operate([qubits[8], qubits[7], qubits[6]], ops.CCX)

            operate(qubits[0], ops.H)
            operate(qubits[3], ops.H)
            operate(qubits[6], ops.H)

            operate([qubits[0], qubits[3]], ops.CNOT)
            operate([qubits[0], qubits[6]], ops.CNOT)
            operate([qubits[6], qubits[3], qubits[0]], ops.CCX)




            return qubits[0]
            
        # elif self.qec_method == "steane_seven":
        #     # Syndrome measurements for Steane code
        #     # X error detection (Z-type stabilizers)
        #     z_sets = [
        #         (0, 1, 2, 3), # Z Z Z Z I I I
        #         (0, 1, 4, 5), # Z Z I I Z Z I
        #         (0, 2, 4, 6), # Z I Z I Z I Z
        #     ]
            
        #     z_anc = create_qubits(3)
            
        #     # Measure Z-type stabilizers
        #     for s_idx, subset in enumerate(z_sets):
        #         for d_idx in subset:
        #             operate([qubits[d_idx], z_anc[s_idx]], ops.CNOT)
            
        #     z_syndrome = []
        #     for anc in z_anc:
        #         res, _ = measure(anc)
        #         z_syndrome.append(res)
            
        #     # Z error detection (X-type stabilizers)
        #     x_sets = z_sets # same index sets
        #     x_anc = create_qubits(3)
            
        #     # Measure X-type stabilizers
        #     for s_idx, subset in enumerate(x_sets):
        #         # Rotate indicated data qubits into X-basis
        #         for d_idx in subset:
        #             operate(qubits[d_idx], ops.H)
                
        #         # Accumulate Z-parity (which is X-parity of original state)
        #         for d_idx in subset:
        #             operate([qubits[d_idx], x_anc[s_idx]], ops.CNOT)
                
        #         # Rotate data qubits back
        #         for d_idx in subset:
        #             operate(qubits[d_idx], ops.H)
            
        #     x_syndrome = []
        #     for anc in x_anc:
        #         res, _ = measure(anc)
        #         x_syndrome.append(res)
            
        #     print(f"X syndrome: {x_syndrome}, Z syndrome: {z_syndrome}")
            
        #     # Create syndrome-to-qubit index mapping
        #     syndrome_to_idx = {
        #         (1, 1, 1): 0, # qubit 0
        #         (1, 1, 0): 1, # qubit 1
        #         (1, 0, 1): 2, # qubit 2
        #         (1, 0, 0): 3, # qubit 3
        #         (0, 1, 1): 4, # qubit 4
        #         (0, 1, 0): 5, # qubit 5
        #         (0, 0, 1): 6, # qubit 6
        #     }
            
        #     # Apply X correction if an X-syndrome is present
        #     z_syndrome_value = tuple(z_syndrome)
        #     x_idx = syndrome_to_idx.get(z_syndrome_value)
        #     if x_idx is not None:
        #         operate(qubits[x_idx], ops.X)
        #         print(f"Corrected X error on qubit {x_idx}")
            
        #     # Apply Z correction if a Z-syndrome is present
        #     x_syndrome_value = tuple(x_syndrome)
        #     z_idx = syndrome_to_idx.get(x_syndrome_value)
        #     if z_idx is not None:
        #         operate(qubits[z_idx], ops.Z)
        #         print(f"Corrected Z error on qubit {z_idx}")

        #     # Apply Z-type half (global H, pattern, global H) in reverse
        #     for q in qubits:
        #         operate(q, ops.H)
        #     # same CNOT pattern as encoder
        #     for ctrl,trg in ((0,2),(1,2),(0,4),(3,4),(1,5),(3,5)):
        #         operate([qubits[ctrl], qubits[trg]], ops.CNOT)
        #     for q in qubits:
        #         operate(q, ops.H)

        #     # Undo X-type half (CNOTs from data qubit 0 to ancillas)
        #     anc = range(1,7)
        #     for a in anc:
        #         operate([qubits[0], qubits[a]], ops.CNOT)

        #     for ctrl,trg in ((0,2),(1,2), (0,4),(3,4), (1,5),(3,5)):
        #         operate([qubits[ctrl], qubits[trg]], ops.CNOT)
            
        #     print("Decoded using Steane's [7,1,3] code")
        #     return qubits[0]
            

    def calculate_fidelity(self, qubit, reference_state=None):
        """
        Calculate the fidelity of the qubit with respect to the reference state.
        
        Parameters:
        -----------
        qubit : Qubit
            The qubit to measure fidelity of
        reference_state : numpy.array or Qubit, optional
            Reference state to compare against. If None, use the initial state.
        
        Returns:
        --------
        float: Fidelity between 0 and 1
        """
        # Check if qubit is None
        if qubit is None:
            print("Warning: Input qubit is None, returning fidelity 0.0")
            return 0.0
            
        try:
            # Import necessary classes
            from netsquid.qubits.qubitapi import reduced_dm
            from netsquid.qubits.dmtools import DenseDMRepr
            
            # Get density matrix of the measured qubit
            dm_qubit = reduced_dm(qubit)
            
            # Create a DMRepr instance for the qubit
            dm_qubit_repr = DenseDMRepr(dm=dm_qubit)
            
            if reference_state is None:
                # Create a reference qubit in the initial state
                from netsquid.qubits.qubitapi import create_qubits, operate
                from netsquid.qubits import operators as ops
                
                ref_qubit, = create_qubits(1)
                
                if self.initial_state == "0":
                    pass  # Already in |0⟩ state
                elif self.initial_state == "1":
                    operate(ref_qubit, ops.X)
                elif self.initial_state == "+":
                    operate(ref_qubit, ops.H)
                elif self.initial_state == "-":
                    operate(ref_qubit, ops.X)
                    operate(ref_qubit, ops.H)
                        
                # Get density matrix of reference qubit
                dm_ref = reduced_dm(ref_qubit)
                
            else:
                # Check if reference_state is a Qubit or already a density matrix
                if hasattr(reference_state, 'qstate'):  # It's a Qubit
                    dm_ref = reduced_dm(reference_state)
                else:  # Assume it's already a density matrix
                    dm_ref = reference_state
            
            # Create DMRepr for the reference state
            dm_ref_repr = DenseDMRepr(dm=dm_ref)
            
            # Compute and return fidelity
            return dm_qubit_repr.fidelity(dm_ref_repr)
            
        except Exception as e:
            print(f"Error calculating fidelity: {e}")
            return 0.0  # Return default value on error





    def run_experiment(self):
        """
        Run the quantum error correction experiment.
        
        Returns:
        --------
        dict: Results dictionary with fidelities and error mitigation factor
        """
        # Initialize results dictionary
        results = {
            "fidelity_with_qec": 0.0,
            "fidelity_without_qec": 0.0, 
            "error_mitigation_factor": 0.0
        }
        
        print("\n=== Running QEC Experiment ===")
        print(f"Initial state: {self.initial_state}")
        print(f"Error type: {self.error_type}")
        print(f"QEC method: {self.qec_method}")
        
        # Prepare the initial qubit
        original_qubit = self.prepare_qubit()
        
        # For random states, store the original state for later fidelity calculations
        if self.initial_state == "random":
            self.reference_state = reduced_dm(original_qubit)
        
        # Run with QEC
        if self.qec_method != "none":
            print("\n--- With QEC ---")
            # Make a copy of the qubit for QEC
            qec_qubit, = create_qubits(1)
            assign_qstate([qec_qubit], reduced_dm(original_qubit))
            
            # Encode the qubit
            encoded_qubits = self.encode_qubit(qec_qubit)
            
            # Apply error to encoded qubits
            self.apply_error(encoded_qubits)
            
            # Decode and correct
            decoded_qubit = self.decode_qubit(encoded_qubits)
            
            # Measure fidelity with correction
            fidelity_with_qec = self.calculate_fidelity(decoded_qubit)
            results["fidelity_with_qec"] = fidelity_with_qec
            print(f"Fidelity with QEC: {fidelity_with_qec:.4f}")
        
        # Run without QEC
        print("\n--- Without QEC ---")
        # Make a copy of the qubit without QEC
        no_qec_qubit, = create_qubits(1)
        assign_qstate([no_qec_qubit], reduced_dm(original_qubit))
        
        # Apply error directly
        self.apply_error([no_qec_qubit])
        
        # Measure fidelity without correction
        fidelity_without_qec = self.calculate_fidelity(no_qec_qubit)
        results["fidelity_without_qec"] = fidelity_without_qec
        print(f"Fidelity without QEC: {fidelity_without_qec:.4f}")
        
        # Calculate error mitigation factor
        if self.qec_method != "none" and fidelity_without_qec > 0:
            emf = results["fidelity_with_qec"] / results["fidelity_without_qec"]
            results["error_mitigation_factor"] = emf
            print(f"Error mitigation factor: {emf:.4f}")
            
        return results


    def main():
        """
        Example usage of the QuantumErrorCorrectionExperiment class.
        """
        # Example 1: Test bit-flip error with 3-qubit bit-flip code
        print("\n\n=== Example 1: Bit-flip error with 3-qubit bit-flip code ===")
        experiment1 = QuantumErrorCorrectionExperiment(
            initial_state="+",  # |+⟩ state is sensitive to phase errors
            error_type="bit_flip",
            error_params={"probability": 0.2},
            qec_method="three_qubit_bit_flip"
        )
        results1 = experiment1.run_experiment()
        
        # Example 2: Test phase-flip error with 3-qubit phase-flip code
        print("\n\n=== Example 2: Phase-flip error with 3-qubit phase-flip code ===")
        experiment2 = QuantumErrorCorrectionExperiment(
            initial_state="+",  # |+⟩ state is sensitive to phase errors
            error_type="phase_flip",
            error_params={"probability": 0.2},
            qec_method="three_qubit_phase_flip"
        )
        results2 = experiment2.run_experiment()
        
        


if __name__ == "__main__":
    main()






