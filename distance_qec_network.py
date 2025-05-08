import os
import time
import gc
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import netsquid as ns
from netsquid.nodes import Node, Network
from netsquid.components import QuantumChannel, ClassicalChannel
from netsquid.nodes.connections import DirectConnection
from netsquid.protocols import NodeProtocol, Signals
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel, DephaseNoiseModel, T1T2NoiseModel, FibreLossModel
)
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.qubits.qubit import Qubit # Import Qubit class

# Ensure single_qubit_experiments.py is in the Python path or same directory
try:
    from single_qubit_experiments import QuantumErrorCorrectionExperiment
except ImportError:
    print("Error: Could not import QuantumErrorCorrectionExperiment.")
    print("Make sure 'single_qubit_experiments.py' is in the same directory or Python path.")
    exit()

# Use slots for memory efficiency
class ExperimentResult:
    __slots__ = ['fidelity', 'qec_method', 'initial_state', 'error_type', 'error_params', 'distance']

    def __init__(self):
        self.fidelity = 0.0
        self.qec_method = ""
        self.initial_state = ""
        self.error_type = ""
        self.error_params = {}
        self.distance = 0.0

def monitor_memory():
    """Monitor current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    return memory_mb

# === Corrected Protocol Definitions (Copied from fixed network_qec_simulation.py) ===

class SenderProtocol(NodeProtocol):
    """Protocol to prepare, encode, and send qubits."""
    def __init__(self, node, qec_experiment, port_name, num_qubits_to_send, name=None):
        super().__init__(node, name=name) # Pass name to superclass
        self.qec_experiment = qec_experiment
        self.port_name = port_name
        self.num_qubits_to_send = num_qubits_to_send
        print(f"SenderProtocol '{self.name}' initialized for node {node.name} on port {port_name}, sending {num_qubits_to_send} qubits.")

    def run(self):
        print(f"{ns.sim_time()}: SenderProtocol '{self.name}' running on {self.node.name}")
        qubit = self.qec_experiment.prepare_qubit()
        initial_dm = ns.qubits.reduced_dm(qubit)

        encoded_qubits = self.qec_experiment.encode_qubit(qubit)

        if len(encoded_qubits) != self.num_qubits_to_send:
             print(f"Warning: Expected to send {self.num_qubits_to_send} qubits, but encoded {len(encoded_qubits)} for QEC method {self.qec_experiment.qec_method}")

        print(f"{ns.sim_time()}: Sending {len(encoded_qubits)} encoded qubits...")
        for i, q in enumerate(encoded_qubits):
            self.node.ports[self.port_name].tx_output(q)
            yield self.await_timer(1) # Wait 1 ns between sends

        print(f"{ns.sim_time()}: Sender '{self.name}' finished sending qubits.")
        return {"initial_dm": initial_dm}

class ReceiverProtocol(NodeProtocol):
    """Protocol to receive, decode qubits, and calculate fidelity."""
    def __init__(self, node, qec_experiment, port_name, num_qubits_to_receive, sender_protocol, name=None):
        super().__init__(node, name=name) # Pass name to superclass
        self.qec_experiment = qec_experiment
        self.received_qubits = []
        self.port_name = port_name
        self.num_qubits_to_receive = num_qubits_to_receive
        self.sender_protocol = sender_protocol # Store the sender protocol instance
        self.initial_dm_sender = None
        # Store the qec_logic object passed to the NetworkQEC instance, which holds the distance.
        # Note: qec_experiment passed here is the *same object* as qec_logic in NetworkQEC
        self.qec_logic = qec_experiment
        print(f"ReceiverProtocol '{self.name}' initialized for node {node.name} on port {port_name}, expecting {num_qubits_to_receive} qubits.")

    def run(self):
        print(f"{ns.sim_time()}: Receiver '{self.name}' running on {self.node.name}, expecting {self.num_qubits_to_receive} qubits.")
        self.received_qubits = []

        # Calculate a deadline by which all qubits should arrive
        # Propagation delay: distance_km * 5000 ns/km (from 200,000 km/s speed)
        # Sending interval: num_qubits * 1 ns (from SenderProtocol's await_timer(1))
        # Add a buffer (e.g., 1 microsecond)
        # Access distance from the qec_logic object stored during __init__
        distance_km = self.qec_logic.distance if hasattr(self.qec_logic, 'distance') else 1.0 # Default 1km if not found
        propagation_delay_ns = distance_km * 5000
        sending_delay_ns = self.num_qubits_to_receive * 1 # Based on SenderProtocol's delay
        buffer_ns = 100000 # 1 us buffer, can be adjusted
        max_total_delay = propagation_delay_ns + (sending_delay_ns*buffer_ns)
        deadline_time = ns.sim_time() + max_total_delay

        print(f"{ns.sim_time()}: Waiting for {self.num_qubits_to_receive} qubits (deadline: {deadline_time:.2f} ns)...")

        # Wait for the expected number of inputs
        for i in range(self.num_qubits_to_receive):
            # Simply wait for the next input on the port
            # If a qubit is lost, this protocol will wait indefinitely here until ns.sim_run() finishes.
            # We will check the time *after* this loop finishes.
            yield self.await_port_input(self.node.ports[self.port_name])
            # Process the received message immediately
            message = self.node.ports[self.port_name].rx_input()
            if message and message.items:
                qubit = message.items[0]
                if isinstance(qubit, ns.qubits.qubit.Qubit):
                    self.received_qubits.append(qubit)
                    print(f"{ns.sim_time()}: Received qubit {len(self.received_qubits)}/{self.num_qubits_to_receive}")
                else:
                    print(f"{ns.sim_time()}: Warning: Received non-qubit item: {type(qubit)}")
            else:
                 print(f"{ns.sim_time()}: Warning: Received empty or invalid message on qubit port at iteration {i}")

        # After the loop, check if the deadline passed or if we didn't get all qubits
        current_time = ns.sim_time()
        received_count = len(self.received_qubits)

        # Check for loss: either time is past deadline OR we didn't get enough qubits
        # (The time check handles the case where we wait indefinitely for a lost qubit)
        if current_time > deadline_time or received_count < self.num_qubits_to_receive:
            print(f"{current_time}: Deadline {deadline_time:.2f} possibly exceeded or incorrect qubit count ({received_count}/{self.num_qubits_to_receive}). Assuming loss.")
            return {"status": "lost"}

        # If deadline not passed and count is correct, proceed
        print(f"{current_time}: Received {received_count} qubits before deadline. Proceeding to decode.")

        # --- Decoding and Fidelity Calculation ---
        print(f"{ns.sim_time()}: Decoding qubits...")
        # Ensure we pass a list copy if decode_qubit modifies the list
        decoded_qubit = self.qec_experiment.decode_qubit(list(self.received_qubits))
        print(f"{ns.sim_time()}: Decoding complete.")

        # Fetch the initial state DM from the sender's result using the stored instance
        if not self.sender_protocol.is_running:
            sender_result = self.sender_protocol.get_signal_result(Signals.FINISHED)
            self.initial_dm_sender = sender_result.get("initial_dm")
        else:
             print(f"{ns.sim_time()}: Warning: Sender protocol '{self.sender_protocol.name}' not finished when receiver needs initial state.")
             # Try to get it anyway, handle error
             try:
                 sender_result = self.sender_protocol.get_signal_result(Signals.FINISHED)
                 self.initial_dm_sender = sender_result.get("initial_dm")
             except Exception as e:
                 print(f"Error getting sender result: {e}")
                 self.initial_dm_sender = None

        if decoded_qubit is None:
             print(f"{ns.sim_time()}: Error: Decoded qubit is None.")
             fidelity = 0.0
             error_msg = "Decoded qubit is None"
        elif self.initial_dm_sender is None:
             print(f"{ns.sim_time()}: Error: Initial state from sender not available for fidelity calculation.")
             fidelity = 0.0 # Cannot calculate fidelity without reference
             error_msg = "Initial state DM not received from sender"
        else:
            print(f"{ns.sim_time()}: Calculating fidelity...")
            # Use the calculate_fidelity method, providing the initial DM as the reference
            fidelity = self.qec_experiment.calculate_fidelity(
                decoded_qubit,
                reference_state=self.initial_dm_sender
            )
            print(f"{ns.sim_time()}: Fidelity calculated: {fidelity:.4f}")
            error_msg = None # No error if fidelity calculated

        result_dict = {
            "fidelity": fidelity,
            "qec_method": self.qec_experiment.qec_method,
            "initial_state": self.qec_experiment.initial_state
        }
        if error_msg:
            result_dict["error"] = error_msg # Include error message if fidelity calculation failed

        return result_dict

# === Updated NetworkQEC Class ===

class NetworkQEC:
    def __init__(self,
                 initial_state="+",
                 error_type="fibre_loss", # Primary error
                 error_params=None,
                 qec_method="shor_nine",
                 distance=1.0,
                 use_combined_errors=False,
                 secondary_error_type=None, # Optional secondary noise
                 secondary_error_params=None,
                 tertiary_error_type=None,    # Optional tertiary noise
                 tertiary_error_params=None):
        self.initial_state = initial_state
        self.error_type = error_type
        self.error_params = error_params if error_params is not None else {}
        self.qec_method = qec_method
        self.distance = distance
        self.use_combined_errors = use_combined_errors
        self.secondary_error_type = secondary_error_type
        self.secondary_error_params = secondary_error_params if secondary_error_params is not None else {}
        self.tertiary_error_type = tertiary_error_type
        self.tertiary_error_params = tertiary_error_params if tertiary_error_params is not None else {}

        # Determine number of physical qubits required by QEC method
        self.num_physical_qubits = self._get_num_qubits_for_qec(qec_method)

        # Create QEC experiment instance - used only for logic (prepare, encode, decode, fidelity)
        # IMPORTANT: Initialize with error_type="none" as channel handles errors.
        self.qec_logic = QuantumErrorCorrectionExperiment(
            initial_state=initial_state,
            error_type="none",
            qec_method=qec_method,
            distance=0 # Distance handled by channel
        )

        # Setup network *after* initializing qec_logic
        self.network = self._setup_network()


    def _get_num_qubits_for_qec(self, qec_method):
        """Determine the number of physical qubits for a given QEC method."""
        if qec_method == "none":
            return 1
        elif qec_method in ["three_qubit_bit_flip", "three_qubit_phase_flip"]:
            return 3
        elif qec_method == "shor_nine":
            return 9
        # Add Steane etc. if implemented in single_qubit_experiments.py
        else:
            raise ValueError(f"Unknown QEC method: {qec_method}")

    def _setup_network(self):
        """Set up a network with sender and receiver nodes and potentially combined noisy channels."""
        ns.sim_reset()
        network = Network("QEC_Network")

        sender = Node("sender")
        receiver = Node("receiver")
        network.add_node(sender)
        network.add_node(receiver)

        qchannel = QuantumChannel(name="qchannel_S_to_R", length=self.distance)
        if not hasattr(qchannel, 'models') or qchannel.models is None:
            qchannel.models = {}

        # 1. Delay Model
        qchannel.models["delay_model"] = FibreDelayModel(c=200000.0)

        # 2. Apply Error Models (Loss and/or Noise)
        loss_model_applied = False
        noise_applied = False
        error_info_parts = []
        noise_model = None
        loss_model = None

        # --- Determine which models to apply ---
        apply_loss = False
        apply_amp_damp = False
        apply_phase_damp = False
        amp_damp_params = None
        phase_damp_params = None

        # Check primary error type
        if self.error_type == "fibre_loss":
            apply_loss = True
            loss_params = self.error_params
        else:
            # Primary is a noise type
            if self.error_type == "amplitude_damping":
                apply_amp_damp = True
                amp_damp_params = self.error_params
            elif self.error_type == "phase_damping":
                apply_phase_damp = True
                phase_damp_params = self.error_params
            else: # Other noise types (depolarizing etc.)
                noise_model = self._get_netsquid_noise_model(self.error_type, self.error_params) # Use fallback/direct model
                if noise_model:
                    noise_applied = True
                    error_info_parts.append(f"{self.error_type}({self.error_params})")

        # Check secondary/tertiary errors if combining
        if self.use_combined_errors and self.secondary_error_type:
            # Check secondary
            if self.secondary_error_type == "fibre_loss": apply_loss = True; loss_params = self.secondary_error_params
            elif self.secondary_error_type == "amplitude_damping": apply_amp_damp = True; amp_damp_params = self.secondary_error_params
            elif self.secondary_error_type == "phase_damping": apply_phase_damp = True; phase_damp_params = self.secondary_error_params
            # Check tertiary (only if secondary was specified)
            if self.tertiary_error_type:
                if self.tertiary_error_type == "fibre_loss": apply_loss = True; loss_params = self.tertiary_error_params
                elif self.tertiary_error_type == "amplitude_damp": apply_amp_damp = True; amp_damp_params = self.tertiary_error_params
                elif self.tertiary_error_type == "phase_damp": apply_phase_damp = True; phase_damp_params = self.tertiary_error_params

        # --- Apply the determined models ---

        # Apply Loss Model if needed
        if apply_loss:
            loss_model = self._get_netsquid_loss_model(loss_params)
            if loss_model:
                qchannel.models["quantum_loss_model"] = loss_model
                loss_model_applied = True
                error_info_parts.append(f"Loss({loss_params})")

        # Apply Noise Model(s)
        if apply_amp_damp and apply_phase_damp:
            # Combine amplitude and phase damping into one T1T2 model
            gamma_amp = amp_damp_params.get("gamma", 0.1)
            gamma_phase = phase_damp_params.get("gamma", 0.1)

            T1_total = 1e9 / gamma_amp if gamma_amp > 0 else float('inf')
            # Dephasing rate from amplitude damping: gamma_amp / 2
            # Dephasing rate from phase damping: gamma_phase
            total_dephasing_rate = (gamma_amp / 2) + gamma_phase
            T2_total = 1e9 / total_dephasing_rate if total_dephasing_rate > 0 else float('inf')

            # Ensure T2 <= 2*T1 constraint (though typically met with physical rates)
            if T1_total != float('inf') and T2_total > 2 * T1_total:
                print(f"Warning: Calculated T2_total {T2_total:.2e} > 2*T1_total {2*T1_total:.2e}. Clamping T2.")
                T2_total = 2 * T1_total

            if T1_total > 0 or T2_total > 0:
                noise_model = T1T2NoiseModel(T1=T1_total, T2=T2_total)
                noise_applied = True
                error_info_parts.append(f"CombinedNoise(T1={T1_total:.2e}, T2={T2_total:.2e})")
                print(f"Applying Combined T1T2NoiseModel: T1={T1_total:.2e} ns, T2={T2_total:.2e} ns")
            else:
                print("Warning: Combined noise calculation resulted in T1=T2=0. No noise applied.")

        elif apply_amp_damp: # Only amplitude damping requested as noise
            noise_model = self._get_netsquid_noise_model("amplitude_damping", amp_damp_params)
            if noise_model: noise_applied = True; error_info_parts.append(f"AmpDamp({amp_damp_params})")
        elif apply_phase_damp: # Only phase damping requested as noise
            noise_model = self._get_netsquid_noise_model("phase_damping", phase_damp_params)
            if noise_model: noise_applied = True; error_info_parts.append(f"PhaseDamp({phase_damp_params})")
        # else: noise_model might have been set if primary was depolarizing etc.

        if noise_model and not qchannel.models.get("quantum_noise_model"): # Assign if not already set by combined logic
            qchannel.models["quantum_noise_model"] = noise_model

        print(f"Quantum Channel Models Applied: {', '.join(error_info_parts) if error_info_parts else 'None'}")

        # Classical Channel (unchanged)
        cchannel = ClassicalChannel(name="cchannel_R_to_S", length=self.distance)
        if not hasattr(cchannel, 'models') or cchannel.models is None:
            cchannel.models = {}
        cchannel.models["delay_model"] = FibreDelayModel(c=200000.0)

        # Connection (unchanged)
        connection = DirectConnection(name="qec_connection", channel_AtoB=qchannel, channel_BtoA=cchannel)

        # Network Connection (unchanged)
        port_name_sender, port_name_receiver = network.add_connection(
            sender.name, receiver.name, connection=connection, label="q_comm"
        )
        self.port_name_sender = port_name_sender
        self.port_name_receiver = port_name_receiver

        print(f"Network setup complete. Sender port: {port_name_sender}, Receiver port: {port_name_receiver}")
        return network

    def _get_netsquid_loss_model(self, loss_params):
        """Creates a FibreLossModel instance."""
        p_loss_init = loss_params.get("p_loss_init", 0.0)
        p_loss_length = loss_params.get("p_loss_length", 0.2)
        print(f"Applying FibreLossModel: init={p_loss_init}, length={p_loss_length} dB/km")
        return FibreLossModel(p_loss_init=p_loss_init, p_loss_length=p_loss_length)

    def _get_netsquid_noise_model(self, error_type, error_params):
        """Maps an error type string and params to a NetSquid QuantumErrorModel (noise only)."""
        # This is essentially the same as the fixed _get_netsquid_error_model,
        # but explicitly excludes FibreLossModel as that's handled separately.
        error_model = None
        p = error_params.get("probability")
        gamma = error_params.get("gamma")

        print(f"Configuring noise model for type: {error_type}, params: {error_params}")

        if error_type == "depolarizing":
            if p is None: p = 0.1
            error_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
            print(f"Using DepolarNoiseModel with probability P={p}")
        elif error_type == "phase_flip":
             if p is None: p = 0.1
             error_model = DephaseNoiseModel(dephase_rate=p, time_independent=True)
             print(f"Using DephaseNoiseModel with probability P={p}")
        elif error_type == "amplitude_damping":
            if gamma is None: gamma = 0.1
            t1 = 1e5 / gamma if gamma > 0 else float('inf')
            t2 = t1 # T2=2*T1 for pure amplitude damping approximation
            error_model = T1T2NoiseModel(T1=t1, T2=t2)
            print(f"Using T1T2NoiseModel for Amplitude Damping: T1={t1:.2e} ns, T2={t2:.2e} ns (gamma={gamma}/s)")
        elif error_type == "phase_damping":
            if gamma is None: gamma = 0.1
            t2 = 1e5 / gamma if gamma > 0 else float('inf')
            t1 = float('inf')
            error_model = T1T2NoiseModel(T1=t1, T2=t2)
            print(f"Using T1T2NoiseModel for Phase Damping: T1=inf, T2={t2:.2e} ns (gamma={gamma}/s)")
        elif error_type == "gen_amplitude_damping":
             if gamma is None: gamma = 0.1
             p_param = error_params.get("probability", 0.5)
             t1 = 1e5 / gamma if gamma > 0 else float('inf')
             t2 = t1 # Approximation
             error_model = T1T2NoiseModel(T1=t1, T2=t2)
             print(f"Warning: Approximating GAD using T1T2NoiseModel: T1={t1:.2e} ns, T2={t2:.2e} ns (gamma={gamma}/s, p={p_param})")
        elif error_type in ["bit_flip", "bit_phase_flip", "coherent_overrotation", "mixed_xz"]:
             print(f"Warning: Error type '{error_type}' falling back to Depolarizing noise.")
             if p is None: p = 0.1
             error_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
             print(f"Using Fallback DepolarNoiseModel with probability P={p}")
        elif error_type == "fibre_loss":
             print(f"Note: '{error_type}' handled by specific loss model method, not noise model.")
             # Return None here, as it's not a 'noise' model in this context
             return None
        else:
             print(f"Warning: Unknown error type '{error_type}' for noise model. No noise applied.")
             return None

        return error_model

    def run_simulation(self):
        """Run the QEC simulation using protocols."""
        error_desc = f"Primary={self.error_type}"
        if self.use_combined_errors:
            if self.secondary_error_type: error_desc += f", Secondary={self.secondary_error_type}"
            if self.tertiary_error_type: error_desc += f", Tertiary={self.tertiary_error_type}"
        print(f"Starting simulation run: QEC={self.qec_method}, Errors=({error_desc}), Dist={self.distance} km")
        sender_node = self.network.get_node("sender")
        receiver_node = self.network.get_node("receiver")

        # Create protocols using corrected definitions and passing sender instance
        sender_protocol = SenderProtocol(
            node=sender_node,
            qec_experiment=self.qec_logic,
            port_name=self.port_name_sender,
            num_qubits_to_send=self.num_physical_qubits,
            name="SenderProtocol"
        )
        receiver_protocol = ReceiverProtocol(
            node=receiver_node,
            qec_experiment=self.qec_logic,
            port_name=self.port_name_receiver,
            num_qubits_to_receive=self.num_physical_qubits,
            sender_protocol=sender_protocol, # Pass sender instance
            name="ReceiverProtocol"
        )

        sender_protocol.start()
        receiver_protocol.start()

        estimated_propagation_delay = self.distance * 5000
        estimated_sending_duration = self.num_physical_qubits * 1
        generous_buffer_ns = 100000 # 100 us buffer
        estimated_duration = estimated_propagation_delay + estimated_sending_duration + generous_buffer_ns

        # Run the simulation for the estimated duration, or until protocols finish (whichever is sooner)
        print(f"Running simulation with estimated max duration: {estimated_duration:.2f} ns")
        ns.sim_run(duration=estimated_duration)

        # Use corrected check for finished protocol
        if not receiver_protocol.is_running:
             result = receiver_protocol.get_signal_result(Signals.FINISHED)
             print(f"Simulation run finished. Result: {result}")
             return result
        else:
             print("Error: Receiver protocol did not finish.")
             # Check sender status too
             if not sender_protocol.is_running:
                  print("Sender protocol finished.")
             else:
                  print("Sender protocol also did not finish.")
             return {
                 "fidelity": 0.0,
                 "qec_method": self.qec_logic.qec_method,
                 "initial_state": self.qec_logic.initial_state,
                 "error": "Protocol did not finish"
             }

# === Experiment Running Functions (with minor formatting fixes) ===

def run_single_experiment_with_streaming(initial_state, error_type, error_params, qec_method,
                                       distance=1.0, use_combined_errors=False,
                                       secondary_error_type=None, secondary_error_params=None,
                                       tertiary_error_type=None, tertiary_error_params=None,
                                       num_iterations=500, gc_every=10, base_dir="distance_qec_fiber_loss"):
    """Run a single experiment with streaming results and optional error re-raising."""
    # Directory and filename setup (unchanged conceptually)
    error_type_str = error_type.replace(" ", "_")
    if use_combined_errors and secondary_error_type:
        combined_err_str = f"{error_type_str}_plus_{secondary_error_type.replace(' ', '_')}"
        if tertiary_error_type:
            combined_err_str += f"_plus_{tertiary_error_type.replace(' ', '_')}"
        error_type_dir = os.path.join(base_dir, combined_err_str)
    else:
         error_type_dir = os.path.join(base_dir, error_type_str)
    os.makedirs(error_type_dir, exist_ok=True)

    param_str = "_".join([f"{k}_{v}" for k, v in sorted(error_params.items())])
    if use_combined_errors and secondary_error_params:
        sec_param_str = "_".join([f"sec_{k}_{v}" for k, v in sorted(secondary_error_params.items())])
        param_str = f"{param_str}_{sec_param_str}"
    if use_combined_errors and tertiary_error_params:
        ter_param_str = "_".join([f"ter_{k}_{v}" for k, v in sorted(tertiary_error_params.items())])
        param_str = f"{param_str}_{ter_param_str}"
    param_str = param_str.replace(" ", "_").replace("=", "_")
    filename = f"{initial_state}_{qec_method}_d{distance}_{param_str}.csv"
    filepath = os.path.join(error_type_dir, filename)

    # Check for existing results and resume logic (unchanged conceptually)
    try:
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            if len(df_existing) >= num_iterations:
                print(f"Skipping (already completed): {filepath} with {len(df_existing)} iterations.")
                return None
            else:
                 print(f"Resuming experiment: {filepath} (found {len(df_existing)}/{num_iterations} iterations).")
                 start_iteration = len(df_existing)
                 iterations_to_run = num_iterations - start_iteration
                 file_mode = 'a'
                 write_header = False
        else:
             iterations_to_run = num_iterations
             start_iteration = 0
             file_mode = 'w'
             write_header = True
    except pd.errors.EmptyDataError:
         print(f"Found empty or corrupted file: {filepath}. Overwriting.")
         iterations_to_run = num_iterations
         start_iteration = 0
         file_mode = 'w'
         write_header = True
    except Exception as e:
         print(f"Error checking existing file {filepath}: {e}. Overwriting.")
         iterations_to_run = num_iterations
         start_iteration = 0
         file_mode = 'w'
         write_header = True

    if iterations_to_run <= 0:
        print(f"No more iterations needed for {filepath}.")
        return None

    loss_count = 0 # Initialize loss counter for this specific experiment config
    with open(filepath, mode=file_mode, newline='') as f:
        if write_header:
            # Corrected: Add newline to header
            f.write("iteration,fidelity\n")

        stats = { "sum_fidelity": 0.0, "sum_squared_fidelity": 0.0, "count": 0 }

        print(f"\nRunning Experiment: State={initial_state}, QEC={qec_method}, Dist={distance} km")
        print(f" Primary Error: {error_type}, Params: {error_params}")
        if use_combined_errors:
            print(f" Secondary Error: {secondary_error_type}, Params: {secondary_error_params}")
            if tertiary_error_type:
                print(f" Tertiary Error: {tertiary_error_type}, Params: {tertiary_error_params}")

        start_memory = monitor_memory()
        print(f"Memory at start: {start_memory:.2f} MB")

        pbar = tqdm(range(iterations_to_run), desc=f"{initial_state},{error_type_str[:5]},{qec_method},d={distance}")
        for i in pbar:
            current_iteration_index = start_iteration + i
            if i % gc_every == 0: gc.collect()

            try:
                ns.sim_reset()
                experiment = NetworkQEC(
                    initial_state=initial_state, error_type=error_type,
                    error_params=error_params, qec_method=qec_method,
                    distance=distance, use_combined_errors=use_combined_errors,
                    secondary_error_type=secondary_error_type,
                    secondary_error_params=secondary_error_params,
                    tertiary_error_type=tertiary_error_type,
                    tertiary_error_params=tertiary_error_params
                )
                result = experiment.run_simulation()

                # Check if the iteration resulted in a loss (timeout)
                # Loss is indicated EITHER by the receiver returning status='lost'
                # OR by NetworkQEC returning error='Protocol did not finish' (meaning receiver got stuck)
                is_lost_internal = result.get("status") == "lost"
                is_lost_external = result.get("error") == "Protocol did not finish"

                if is_lost_internal or is_lost_external:
                    loss_count += 1
                    # Optional: Log loss occurrence if needed beyond the final count
                    if is_lost_internal:
                         print(f"Iteration {current_iteration_index}: Recorded as lost (detected by Receiver).")
                    else:
                         print(f"Iteration {current_iteration_index}: Recorded as lost (detected by NetworkQEC - Receiver stuck).")

                    # Skip fidelity recording and stats update for this iteration
                    del experiment # Clean up instance
                    ns.sim_reset()   # Reset for next iteration
                    continue # Go to the next iteration
                else:
                    # If not lost, proceed with fidelity processing
                    fidelity = result.get("fidelity", 0.0) # Default to 0 if key missing
                    sim_error = result.get("error") # Check for other potential errors returned by receiver
                    # This check is now mainly for unexpected errors, as "Protocol did not finish" is handled above
                    if sim_error:
                        print(f"Warning: Sim error reported in iteration {current_iteration_index}: {sim_error}")
                        # Still record fidelity (likely 0) but be aware of the error

                    stats["sum_fidelity"] += fidelity
                    stats["sum_squared_fidelity"] += fidelity**2
                    stats["count"] += 1

                    # Corrected: Ensure newline after writing data
                    f.write(f"{current_iteration_index},{fidelity}\n")
                    f.flush()

                    del experiment
                    ns.sim_reset()

                    if qec_method in ["shor_nine"] and i % (gc_every // 2 + 1) == 0: gc.collect()

            except Exception as e:
                print(f"\n-----------------------------------------------------")
                print(f"ERROR in iteration {current_iteration_index} ({initial_state}, {error_type}, {qec_method}, d={distance}):")
                import traceback
                traceback.print_exc()
                print(f"-----------------------------------------------------")
                error_log_path = os.path.join(base_dir, "error_log.txt")
                with open(error_log_path, "a") as error_f:
                    error_f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    error_f.write(f"Params: State={initial_state}, Error={error_type}, QEC={qec_method}, Params={error_params}, Dist={distance}, Iteration={current_iteration_index}")
                    if use_combined_errors:
                        error_f.write(f", SecondaryError={secondary_error_type}, SecondaryParams={secondary_error_params}")
                        if tertiary_error_type:
                            error_f.write(f", TertiaryError={tertiary_error_type}, TertiaryParams={tertiary_error_params}")
                    error_f.write(f"\nError: {str(e)}\n")
                    error_f.write(traceback.format_exc())
                    # Corrected: Add newline after separator
                    error_f.write("-" * 20 + "\n")

                # Option to stop on error (uncomment if desired)
                # raise e # Stop script on first error
                # Re-raise the exception to stop the script immediately
                raise e

            if stats["count"] > 0:
                 avg_fid_so_far = stats["sum_fidelity"] / stats["count"]
                 pbar.set_postfix({"avg_fid": f"{avg_fid_so_far:.3f}"})

    # Post-experiment processing (unchanged conceptually, assumes file exists)
    try:
         df_final = pd.read_csv(filepath)
         count = len(df_final)
         if count > 0:
             avg_fidelity = df_final["fidelity"].mean()
             std_fidelity = df_final["fidelity"].std()
         else: avg_fidelity = std_fidelity = 0.0

         metadata = {
             "initial_state": initial_state, "error_type": error_type,
             "error_params": str(error_params), "qec_method": qec_method,
             "distance": distance, "avg_fidelity": avg_fidelity,
             "std_fidelity": std_fidelity, "iterations_completed": count,
             "loss_count": loss_count
         }
         if use_combined_errors and secondary_error_type:
             metadata["secondary_error_type"] = secondary_error_type
             metadata["secondary_error_params"] = str(secondary_error_params)
         if use_combined_errors and tertiary_error_type:
             metadata["tertiary_error_type"] = tertiary_error_type
             metadata["tertiary_error_params"] = str(tertiary_error_params)

         metadata_df = pd.DataFrame([metadata])
         metadata_path = filepath.replace(".csv", "_metadata.csv")
         metadata_df.to_csv(metadata_path, index=False)

         current_memory = monitor_memory()
         print(f"Finished experiment config. Final avg fidelity: {avg_fidelity:.4f} +/- {std_fidelity:.4f}")
         print(f"Results saved to: {filepath}")
         print(f"Metadata saved to: {metadata_path}")
         print(f"Memory usage end: {current_memory:.2f} MB (Delta: {current_memory - start_memory:.2f} MB)")

         gc.collect()
         return metadata

    except Exception as e:
        print(f"Error calculating final stats or saving metadata for {filepath}: {e}")
        gc.collect()
        return None

# Other functions (run_distance_experiments, print_progress, calculate_expected_loss, __main__)
# remain largely the same, relying on the corrected run_single_experiment_with_streaming
# and NetworkQEC. Ensure the parameters passed match the intended logic.

def run_distance_experiments(distances=[10.0, 20.0, 60.0, 100.0, 200.0], num_iterations=500, base_dir="distance_qec_fiber_loss"):
    """Run experiments focused on distance effects with fiber loss and other error models"""
    # Setup parameters (as before)
    fiber_loss_params = [
        {"p_loss_init": 0.05, "p_loss_length": 0.16}, # Premium fiber, low coupling loss
        {"p_loss_init": 0.1, "p_loss_length": 0.20},  # Good fiber, moderate coupling loss
        {"p_loss_init": 0.2, "p_loss_length": 0.25},  # Standard fiber, standard coupling loss
        {"p_loss_init": 0.3, "p_loss_length": 0.35}   # Older/Poor fiber, high coupling loss
    ]
    combined_error_types = ["amplitude_damping", "phase_damping"]
    # Generate 4 gamma values from 0.01 to 0.2
    gamma_values = np.linspace(0.01, 0.2, 4)
    combined_error_params = [{"gamma": g} for g in gamma_values]
    qec_methods = [ "shor_nine"]
    initial_states = ["0","1","+","-"]

    os.makedirs(base_dir, exist_ok=True)

    # Calculate total expected experiments
    num_fiber_configs = len(fiber_loss_params)
    num_secondary_noise_configs = len(combined_error_types) * len(combined_error_params)
    # Triple error: Fiber Loss + Amp Damp + Phase Damp
    # Need to iterate through amp_damp gammas and phase_damp gammas
    num_triple_error_configs = len(combined_error_params) * len(combined_error_params)
    # Total = (Loss Only) + (Loss + Secondary Noise) + (Loss + Amp Damp + Phase Damp)
    total_exps = len(distances) * num_fiber_configs * len(qec_methods) * len(initial_states) * \
                 (1 + num_secondary_noise_configs + num_triple_error_configs)

    print(f"Total Experiment Configurations: {total_exps}")
    completed = 0
    start_time = time.time()

    # Loop through configurations
    for distance in distances:
        for fiber_params in fiber_loss_params:
            for qec_method in qec_methods:
                for initial_state in initial_states:
                    # --- Run with fiber loss ONLY ---
                    print("\n--- Running Fiber Loss Only ---")
                    run_single_experiment_with_streaming(
                        initial_state=initial_state, error_type="fibre_loss",
                        error_params=fiber_params, qec_method=qec_method,
                        distance=distance, num_iterations=num_iterations, base_dir=base_dir
                    )
                    completed += 1
                    print_progress(completed, total_exps, start_time)

                    # --- Run with combined fiber loss + other noise ---
                    for sec_error_type in combined_error_types:
                        for sec_error_params in combined_error_params:
                            print(f"\n--- Running Combined: Fiber Loss + {sec_error_type} ---")
                            run_single_experiment_with_streaming(
                                initial_state=initial_state,
                                error_type="fibre_loss", # Primary is loss
                                error_params=fiber_params,
                                qec_method=qec_method, distance=distance,
                                use_combined_errors=True,
                                secondary_error_type=sec_error_type, # Secondary is noise
                                secondary_error_params=sec_error_params,
                                num_iterations=num_iterations, base_dir=base_dir
                            )
                            completed += 1
                            print_progress(completed, total_exps, start_time)

                    # --- Run with Triple Error: Fiber Loss + Amplitude Damping + Phase Damping ---
                    for amp_damp_params in combined_error_params: # Iterate through gamma for amp damp
                        for phase_damp_params in combined_error_params: # Iterate through gamma for phase damp
                            print(f"\n--- Running Triple: Fiber Loss + Amp Damp + Phase Damp ---")
                            run_single_experiment_with_streaming(
                                initial_state=initial_state,
                                error_type="fibre_loss", # Primary is loss
                                error_params=fiber_params,
                                qec_method=qec_method, distance=distance,
                                use_combined_errors=True, # Signal to combine
                                secondary_error_type="amplitude_damping",
                                secondary_error_params=amp_damp_params,
                                tertiary_error_type="phase_damping", # Specify third error
                                tertiary_error_params=phase_damp_params,
                                num_iterations=num_iterations, base_dir=base_dir
                            )
                            completed += 1
                            print_progress(completed, total_exps, start_time)

    total_time = time.time() - start_time
    print(f"\nAll distance-based experiments completed in {total_time/3600:.2f} hours")


def print_progress(completed, total, start_time):
    """Helper function to print progress information"""
    elapsed = time.time() - start_time
    if completed > 0:
        avg_time = elapsed / completed
        remaining = total - completed
        est_remaining = avg_time * remaining
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        if est_remaining > 0:
            print(f"Estimated time remaining: {est_remaining/60:.1f} minutes")
        else:
            print("Estimated time remaining: < 1 minute")
    gc.collect()

def calculate_expected_loss(distance, p_loss_init, p_loss_length):
    """Calculate expected photon loss probability for a given distance"""
    alpha_db = p_loss_length * distance
    p_transmission = 10**(-alpha_db/10)
    p_total_transmission = (1 - p_loss_init) * p_transmission
    p_total_loss = 1 - p_total_transmission
    return p_total_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run distance-focused quantum error correction network experiments')
    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations per experiment') # Reduced default for testing
    parser.add_argument('--results-dir', type=str, default='distance_qec_fiber_loss', help='Directory to save results')
    parser.add_argument('--distances', type=str, default='10.0,20.0, 60.0, 100.0, 200.0', # Reduced default for testing
                      help='Comma-separated list of distances to test (in km)')
    # Selective run option removed for simplicity in this fix, can be re-added if needed.

    args = parser.parse_args()
    try:
        distances = [float(d.strip()) for d in args.distances.split(',') if d.strip()]
        if not distances: raise ValueError("No valid distances provided.")
    except Exception as e:
        print(f"Error parsing distances argument '{args.distances}': {e}")
        print("Please provide a comma-separated list of numbers (e.g., '1.0,10.0').")
        exit(1)

    print(f"Starting distance-focused QEC experiments with {args.iterations} iterations")
    print(f"Testing distances: {distances} km")
    print(f"Initial memory usage: {monitor_memory():.2f} MB")

    # Expected loss calculation (unchanged)
    print("\nExpected photon loss probabilities:")
    example_params = [
        (0.0, 0.20), # Standard fiber, no coupling loss
        (0.1, 0.17), # Premium fiber, with coupling loss
    ]
    for dist in distances:
        for p_init, p_len in example_params:
            p_loss = calculate_expected_loss(dist, p_init, p_len)
            print(f"  {dist} km (init={p_init}, len={p_len} dB/km): {p_loss*100:.2f}%")

    # Run the experiments
    run_distance_experiments(distances, args.iterations, args.results_dir)

    print(f"\nFinal memory usage: {monitor_memory():.2f} MB")
    print("All done!")
