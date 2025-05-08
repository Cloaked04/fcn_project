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
    DepolarNoiseModel, DephaseNoiseModel, T1T2NoiseModel
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

# === Protocol Definitions (Unchanged from distance_qec_network.py) ===

class SenderProtocol(NodeProtocol):
    """Protocol to prepare, encode, and send qubits."""
    def __init__(self, node, qec_experiment, port_name, num_qubits_to_send, name=None):
        super().__init__(node, name=name) # Pass name to superclass
        self.qec_experiment = qec_experiment
        self.port_name = port_name
        self.num_qubits_to_send = num_qubits_to_send
        # Reduced print statement for less verbose logs
        # print(f"SenderProtocol '{self.name}' initialized for node {node.name} on port {port_name}, sending {num_qubits_to_send} qubits.")

    def run(self):
        # print(f"{ns.sim_time()}: SenderProtocol '{self.name}' running on {self.node.name}") # Reduced verbosity
        qubit = self.qec_experiment.prepare_qubit()
        initial_dm = ns.qubits.reduced_dm(qubit)

        encoded_qubits = self.qec_experiment.encode_qubit(qubit)

        if len(encoded_qubits) != self.num_qubits_to_send:
             print(f"Warning: Expected to send {self.num_qubits_to_send} qubits, but encoded {len(encoded_qubits)} for QEC method {self.qec_experiment.qec_method}")

        # print(f"{ns.sim_time()}: Sending {len(encoded_qubits)} encoded qubits...") # Reduced verbosity
        for i, q in enumerate(encoded_qubits):
            self.node.ports[self.port_name].tx_output(q)
            yield self.await_timer(1) # Wait 1 ns between sends

        # print(f"{ns.sim_time()}: Sender '{self.name}' finished sending qubits.") # Reduced verbosity
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
        self.qec_logic = qec_experiment
        # Reduced print statement
        # print(f"ReceiverProtocol '{self.name}' initialized for node {node.name} on port {port_name}, expecting {num_qubits_to_receive} qubits.")

    def run(self):
        # print(f"{ns.sim_time()}: Receiver '{self.name}' running on {self.node.name}, expecting {self.num_qubits_to_receive} qubits.") # Reduced verbosity
        self.received_qubits = []

        # Correct Deadline Calculation (copied from previous fix attempt)
        distance_km = self.qec_logic.distance if hasattr(self.qec_logic, 'distance') else 1.0
        propagation_delay_ns = distance_km * 5000
        sending_delay_ns = self.num_qubits_to_receive * 1
        buffer_ns = 100000 # 100 us buffer
        max_total_delay = propagation_delay_ns + sending_delay_ns + buffer_ns # Correct addition
        deadline_time = ns.sim_time() + max_total_delay

        # print(f"{ns.sim_time()}: Waiting for {self.num_qubits_to_receive} qubits (deadline: {deadline_time:.2f} ns)...") # Reduced verbosity

        # Wait for the expected number of inputs
        for i in range(self.num_qubits_to_receive):
            yield self.await_port_input(self.node.ports[self.port_name])
            message = self.node.ports[self.port_name].rx_input()
            if message and message.items:
                qubit = message.items[0]
                # Check if it's None or a Qubit before appending
                if qubit is None:
                    print(f"{ns.sim_time()}: Warning: Received None item at iteration {i}. Assuming lost qubit.")
                    # This qubit is lost, break the loop or handle accordingly
                    # For now, just record the issue and continue to the deadline check
                    break
                elif isinstance(qubit, ns.qubits.qubit.Qubit):
                    self.received_qubits.append(qubit)
                    # print(f"{ns.sim_time()}: Received qubit {len(self.received_qubits)}/{self.num_qubits_to_receive}") # Reduced verbosity
                else:
                    print(f"{ns.sim_time()}: Warning: Received non-qubit item: {type(qubit)}")
            else:
                 print(f"{ns.sim_time()}: Warning: Received empty or invalid message on qubit port at iteration {i}")


        # After the loop, check if the deadline passed or if we didn't get all qubits
        current_time = ns.sim_time()
        received_count = len(self.received_qubits)

        # Check for loss (deadline or count mismatch) - This logic remains
        if current_time > deadline_time or received_count < self.num_qubits_to_receive:
            print(f"{current_time}: Deadline ({deadline_time:.2f} ns) possibly exceeded or incorrect qubit count ({received_count}/{self.num_qubits_to_receive}). Assuming loss.")
            return {"status": "lost"}

        # print(f"{current_time}: Received {received_count} qubits before deadline. Proceeding to decode.") # Reduced verbosity

        # --- Decoding and Fidelity Calculation (Unchanged) ---
        # print(f"{ns.sim_time()}: Decoding qubits...") # Reduced verbosity
        decoded_qubit = self.qec_experiment.decode_qubit(list(self.received_qubits))
        # print(f"{ns.sim_time()}: Decoding complete.") # Reduced verbosity

        # Fetch the initial state DM from the sender's result
        if not self.sender_protocol.is_running:
            try:
                 sender_result = self.sender_protocol.get_signal_result(Signals.FINISHED)
                 self.initial_dm_sender = sender_result.get("initial_dm")
            except Exception as e:
                 print(f"Error getting sender result even though finished: {e}")
                 self.initial_dm_sender = None
        else:
             print(f"{ns.sim_time()}: Warning: Sender protocol '{self.sender_protocol.name}' not finished when receiver needs initial state.")
             self.initial_dm_sender = None # Cannot calculate fidelity reliably

        if decoded_qubit is None:
             print(f"{ns.sim_time()}: Error: Decoded qubit is None.")
             fidelity = 0.0
             error_msg = "Decoded qubit is None"
        elif self.initial_dm_sender is None:
             print(f"{ns.sim_time()}: Error: Initial state from sender not available for fidelity calculation.")
             fidelity = 0.0
             error_msg = "Initial state DM not received from sender"
        else:
            # print(f"{ns.sim_time()}: Calculating fidelity...") # Reduced verbosity
            fidelity = self.qec_experiment.calculate_fidelity(
                decoded_qubit,
                reference_state=self.initial_dm_sender
            )
            # print(f"{ns.sim_time()}: Fidelity calculated: {fidelity:.4f}") # Reduced verbosity
            error_msg = None

        result_dict = {
            "fidelity": fidelity,
            "qec_method": self.qec_experiment.qec_method,
            "initial_state": self.qec_experiment.initial_state
        }
        if error_msg:
            result_dict["error"] = error_msg

        return result_dict

# === Modified NetworkQEC Class (Noise Only) ===

class NetworkQEC:
    def __init__(self,
                 initial_state="+",
                 error_type="amplitude_damping", # Changed default to a noise type
                 error_params=None,
                 qec_method="shor_nine",
                 distance=1.0):
                 # Removed combined error parameters
        self.initial_state = initial_state
        self.error_type = error_type
        self.error_params = error_params if error_params is not None else {}
        self.qec_method = qec_method
        self.distance = distance
        # Removed combined error attributes

        # Determine number of physical qubits required by QEC method
        self.num_physical_qubits = self._get_num_qubits_for_qec(qec_method)

        # Create QEC experiment instance - used only for logic (prepare, encode, decode, fidelity)
        self.qec_logic = QuantumErrorCorrectionExperiment(
            initial_state=initial_state,
            error_type="none", # Channel handles errors
            qec_method=qec_method,
            distance=self.distance # Pass distance here for Receiver's deadline calc
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
        # Add Steane etc. if implemented
        else:
            raise ValueError(f"Unknown QEC method: {qec_method}")

    def _setup_network(self):
        """Set up a network with sender/receiver and apply ONLY noise models."""
        ns.sim_reset()
        network = Network("QEC_Noise_Network")

        sender = Node("sender")
        receiver = Node("receiver")
        network.add_node(sender)
        network.add_node(receiver)

        qchannel = QuantumChannel(name="qchannel_S_to_R", length=self.distance)
        if not hasattr(qchannel, 'models') or qchannel.models is None:
            qchannel.models = {}

        # 1. Delay Model (Mandatory)
        qchannel.models["delay_model"] = FibreDelayModel(c=200000.0)

        # 2. Apply Noise Model (No Loss Model)
        noise_model = self._get_netsquid_noise_model(self.error_type, self.error_params)
        if noise_model:
            qchannel.models["quantum_noise_model"] = noise_model
            print(f"Quantum Channel Noise Model Applied: {self.error_type}({self.error_params})")
        else:
            print("No quantum noise model applied to channel.")

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

        # print(f"Network setup complete. Sender port: {port_name_sender}, Receiver port: {port_name_receiver}") # Reduced verbosity
        return network

    def _get_netsquid_noise_model(self, error_type, error_params):
        """Maps an error type string and params to a NetSquid QuantumErrorModel (NO LOSS)."""
        error_model = None
        p = error_params.get("probability")
        gamma = error_params.get("gamma")

        # print(f"Configuring noise model for type: {error_type}, params: {error_params}") # Reduced verbosity

        if error_type == "depolarizing":
            p = p if p is not None else 0.1
            error_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
            # print(f"Using DepolarNoiseModel with probability P={p}")
        elif error_type == "phase_flip":
             p = p if p is not None else 0.1
             error_model = DephaseNoiseModel(dephase_rate=p, time_independent=True)
             # print(f"Using DephaseNoiseModel with probability P={p}")
        elif error_type == "amplitude_damping":
            gamma = gamma if gamma is not None else 0.1
            # Note: T1/T2 are in ns, gamma is often given per second.
            # Conversion: 1/gamma (in seconds) * 1e9 ns/s
            t1 = (1e9 / gamma) if gamma > 0 else float('inf')
            # For pure amplitude damping T_phi = infinity, so 1/T2 = 1/(2*T1) + 1/T_phi = 1/(2*T1)
            # Original: t2 = 2 * t1 if t1 != float('inf') else float('inf')
            # Adjusted: Set T2 slightly less than 2*T1 to avoid potential float issues at the boundary
            t2 = t1
            if t2 < 0: t2 = 0 # Ensure T2 >= 0
            error_model = T1T2NoiseModel(T1=t1, T2=t2)
            # print(f"Using T1T2NoiseModel for Amplitude Damping: T1={t1:.2e} ns, T2={t2:.2e} ns (gamma={gamma}/s)")
        elif error_type == "phase_damping":
            gamma = gamma if gamma is not None else 0.1
            # Note: T1/T2 are in ns, gamma is often given per second.
            # Conversion: 1/gamma (in seconds) * 1e9 ns/s
            # For pure phase damping T1=inf, 1/T2 = 1/(2*T1) + 1/T_phi = 1/T_phi = gamma
            t1 = float('inf')
            t2 = (1 / gamma) if gamma > 0 else float('inf')
            error_model = T1T2NoiseModel(T1=t1, T2=t2)
            # print(f"Using T1T2NoiseModel for Phase Damping: T1=inf, T2={t2:.2e} ns (gamma={gamma}/s)")
        elif error_type == "gen_amplitude_damping":
             gamma = gamma if gamma is not None else 0.1
             p_param = error_params.get("probability", 0.5) # NetSquid GAD param is different
             # Cannot directly map GAD to T1/T2 easily. Use approximation or specific model if available.
             # Using simple T1 approx for now
             t1 = (1 / gamma) if gamma > 0 else float('inf')
             t2 = t1 # Approximation: T2 dominated by T1
             error_model = T1T2NoiseModel(T1=t1, T2=t2)
             print(f"Warning: Approximating GAD using T1T2NoiseModel: T1={t1:.2e} ns, T2={t2:.2e} ns (gamma={gamma}/s, p={p_param})")
        elif error_type in ["bit_flip", "bit_phase_flip", "coherent_overrotation", "mixed_xz"]:
             print(f"Warning: Error type '{error_type}' falling back to Depolarizing noise.")
             p = p if p is not None else 0.1
             error_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
             # print(f"Using Fallback DepolarNoiseModel with probability P={p}")
        elif error_type == "fibre_loss":
             print(f"ERROR: FibreLossModel should not be handled by _get_netsquid_noise_model. No model applied.")
             return None # Explicitly return None for loss type
        elif error_type == "none":
             # print("Explicitly no noise model requested.")
             return None
        else:
             print(f"Warning: Unknown error type '{error_type}' for noise model. No noise applied.")
             return None

        return error_model

    def run_simulation(self):
        """Run the QEC simulation using protocols."""
        # Simplified error description for noise-only
        error_desc = f"{self.error_type}({self.error_params})"
        # print(f"Starting simulation run: QEC={self.qec_method}, Error={error_desc}, Dist={self.distance} km") # Reduced verbosity
        sender_node = self.network.get_node("sender")
        receiver_node = self.network.get_node("receiver")

        # Create protocols (passing sender instance)
        sender_protocol = SenderProtocol(
            node=sender_node, qec_experiment=self.qec_logic,
            port_name=self.port_name_sender, num_qubits_to_send=self.num_physical_qubits,
            name="SenderProtocol"
        )
        receiver_protocol = ReceiverProtocol(
            node=receiver_node, qec_experiment=self.qec_logic,
            port_name=self.port_name_receiver, num_qubits_to_receive=self.num_physical_qubits,
            sender_protocol=sender_protocol, name="ReceiverProtocol"
        )

        sender_protocol.start()
        receiver_protocol.start()

        # Estimated duration calculation (use correct deadline calculation base)
        estimated_propagation_delay = self.distance * 5000
        estimated_sending_duration = self.num_physical_qubits * 1
        buffer_ns = 100000 # 100 us buffer (ensure consistency with ReceiverProtocol)
        estimated_duration = estimated_propagation_delay + estimated_sending_duration + buffer_ns

        # print(f"Running simulation with estimated max duration: {estimated_duration:.2f} ns") # Reduced verbosity
        ns.sim_run(duration=estimated_duration)

        # Check protocol finish status (unchanged)
        if not receiver_protocol.is_running:
             result = receiver_protocol.get_signal_result(Signals.FINISHED)
             # print(f"Simulation run finished. Result: {result}") # Reduced verbosity
             return result
        else:
             print("Error: Receiver protocol did not finish.")
             if not sender_protocol.is_running: print("Sender protocol finished.")
             else: print("Sender protocol also did not finish.")
             return {
                 "fidelity": 0.0, "qec_method": self.qec_logic.qec_method,
                 "initial_state": self.qec_logic.initial_state, "error": "Protocol did not finish"
             }

# === Experiment Running Functions (Modified for Noise Only) ===

def run_single_experiment_with_streaming(initial_state, error_type, error_params, qec_method,
                                       distance=1.0, use_combined_errors=False,
                                       secondary_error_type=None, secondary_error_params=None,
                                       tertiary_error_type=None, tertiary_error_params=None,
                                       num_iterations=200, gc_every=10, base_dir="results_distance_qec"): # Reduced iterations
    """Run a single experiment with streaming results and optional error re-raising."""
    # Simplified Directory and filename setup
    error_type_str = error_type.replace(" ", "_")
    error_type_dir = os.path.join(base_dir, error_type_str)
    os.makedirs(error_type_dir, exist_ok=True)

    param_str = "_".join([f"{k}_{v}" for k, v in sorted(error_params.items())])
    param_str = param_str.replace(" ", "_").replace("=", "_")
    filename = f"{initial_state}_{qec_method}_d{distance}_{param_str}.csv"
    filepath = os.path.join(error_type_dir, filename)

    # Resume logic (unchanged)
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

    loss_count = 0 # Keep loss counter based on deadline check
    with open(filepath, mode=file_mode, newline='') as f:
        if write_header:
            f.write("iteration,fidelity\n")

        stats = { "sum_fidelity": 0.0, "sum_squared_fidelity": 0.0, "count": 0 }

        # Simplified print statement
        print(f"\nRunning Experiment: State={initial_state}, QEC={qec_method}, Dist={distance} km")
        print(f" Error Type: {error_type}, Params: {error_params}")

        start_memory = monitor_memory()
        # print(f"Memory at start: {start_memory:.2f} MB") # Reduced verbosity

        pbar_desc = f"{initial_state},{error_type_str[:5]},{qec_method},d={distance}"
        pbar = tqdm(range(iterations_to_run), desc=pbar_desc)
        for i in pbar:
            current_iteration_index = start_iteration + i
            if i % gc_every == 0: gc.collect()

            try:
                ns.sim_reset()
                # Simplified NetworkQEC call (no combined error args)
                experiment = NetworkQEC(
                    initial_state=initial_state, error_type=error_type,
                    error_params=error_params, qec_method=qec_method,
                    distance=distance
                )
                result = experiment.run_simulation()

                # Check for loss based on deadline (unchanged logic)
                is_lost = result.get("status") == "lost" or result.get("error") == "Protocol did not finish"

                if is_lost:
                    loss_count += 1
                    # Optional: Log loss occurrence
                    # print(f"Iteration {current_iteration_index}: Recorded as lost (deadline/protocol finish issue).")
                    del experiment
                    ns.sim_reset()
                    continue # Skip fidelity processing for this iteration
                else:
                    fidelity = result.get("fidelity", 0.0)
                    sim_error = result.get("error")
                    if sim_error:
                        print(f"Warning: Sim error reported in iteration {current_iteration_index}: {sim_error}")

                    stats["sum_fidelity"] += fidelity
                    stats["sum_squared_fidelity"] += fidelity**2
                    stats["count"] += 1

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
                    # Simplified error logging parameters
                    error_f.write(f"Params: State={initial_state}, Error={error_type}, QEC={qec_method}, Params={error_params}, Dist={distance}, Iteration={current_iteration_index}\n")
                    error_f.write(f"Error: {str(e)}\n")
                    error_f.write(traceback.format_exc())
                    error_f.write("-" * 20 + "\n")
                raise e # Re-raise to stop

            if stats["count"] > 0:
                 avg_fid_so_far = stats["sum_fidelity"] / stats["count"]
                 pbar.set_postfix({"avg_fid": f"{avg_fid_so_far:.3f}"})

    # Post-experiment processing (simplified metadata)
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
             "loss_count": loss_count # Keep loss count from deadline check
         }
         # Removed combined error metadata fields

         metadata_df = pd.DataFrame([metadata])
         metadata_path = filepath.replace(".csv", "_metadata.csv")
         metadata_df.to_csv(metadata_path, index=False)

         current_memory = monitor_memory()
         print(f"Finished experiment config. Final avg fidelity: {avg_fidelity:.4f} +/- {std_fidelity:.4f}")
         print(f"Results saved to: {filepath}")
         print(f"Metadata saved to: {metadata_path}")
         # print(f"Memory usage end: {current_memory:.2f} MB (Delta: {current_memory - start_memory:.2f} MB)") # Reduced verbosity

         gc.collect()
         return metadata

    except Exception as e:
        print(f"Error calculating final stats or saving metadata for {filepath}: {e}")
        gc.collect()
        return None


def run_distance_experiments(distances=[10.0, 20.0, 60.0, 100.0, 200.0], num_iterations=200, base_dir="results_distance_qec"): # Reduced iterations
    """Run experiments focused on distance effects with fiber loss and other error models"""
    # Setup parameters (as before)
    fiber_loss_params = [
        {"p_loss_init": 0.05, "p_loss_length": 0.16}, # Premium fiber, low coupling loss
        {"p_loss_init": 0.1, "p_loss_length": 0.20},  # Good fiber, moderate coupling loss
        {"p_loss_init": 0.2, "p_loss_length": 0.25},  # Standard fiber, standard coupling loss
        {"p_loss_init": 0.3, "p_loss_length": 0.35}   # Older/Poor fiber, high coupling loss
    ]
    combined_error_types = ["amplitude_damping", "phase_damping"]
    # Generate 4 gamma values for faster decoherence (kHz to MHz range)
    # gamma_values = np.linspace(0.01, 0.2, 4) # Old slow values
    gamma_values = [1e4, 1e5, 1e6, 1e7] # Faster decoherence values (10kHz to 10MHz)
    combined_error_params = [{"gamma": g} for g in gamma_values]
    qec_methods = ["none", "three_qubit_phase_flip", "shor_nine"]
    initial_states = ["0","1","+","-"]

    os.makedirs(base_dir, exist_ok=True)

    # Calculate total expected experiments based ONLY on noise parameters
    total_exps = len(distances) * len(qec_methods) * len(initial_states) * \
                 len(fiber_loss_params) * len(combined_error_types) * len(combined_error_params)

    print(f"Total Experiment Configurations (Noise Only): {total_exps}")
    completed = 0
    start_time = time.time()

    # --- Restructured Loops for Noise Only ---
    for distance in distances:
        for qec_method in qec_methods:
            for initial_state in initial_states:
                for fiber_loss_param in fiber_loss_params:
                    for error_type in combined_error_types:
                        for error_params in combined_error_params:
                            # print(f"\n--- Running {error_type} ---") # Reduced verbosity
                            # Simplified call to run_single_experiment_with_streaming
                            run_single_experiment_with_streaming(
                                initial_state=initial_state,
                                error_type=error_type,
                                error_params=error_params,
                                qec_method=qec_method,
                                distance=distance,
                                use_combined_errors=True,
                                secondary_error_type=fiber_loss_param.get("p_loss_init"),
                                secondary_error_params=fiber_loss_param.get("p_loss_length"),
                                num_iterations=num_iterations,
                                base_dir=base_dir
                            )
                            completed += 1
                            print_progress(completed, total_exps, start_time)

    total_time = time.time() - start_time
    print(f"\nAll distance-based noise experiments completed in {total_time/3600:.2f} hours")


def print_progress(completed, total, start_time):
    """Helper function to print progress information"""
    elapsed = time.time() - start_time
    if completed > 0:
        avg_time = elapsed / completed
        remaining = total - completed
        est_remaining = avg_time * remaining
        # print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)") # Reduced verbosity
        # if est_remaining > 0:
        #     print(f"Estimated time remaining: {est_remaining/60:.1f} minutes")
        # else:
        #     print("Estimated time remaining: < 1 minute")
    gc.collect()

# Removed calculate_expected_loss function as it's not relevant here

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run distance-focused quantum error correction network experiments')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations per experiment') # Reduced default iterations
    parser.add_argument('--results-dir', type=str, default='results_distance_qec', help='Directory to save results')
    parser.add_argument('--distances', type=str, default='10.0,20.0, 60.0, 100.0, 200.0',
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

    print(f"Starting distance-focused QEC experiments (NOISE ONLY) with {args.iterations} iterations")
    print(f"Testing distances: {distances} km")
    print(f"Results will be saved in: {args.results_dir}")
    print(f"Initial memory usage: {monitor_memory():.2f} MB")

    # Removed expected loss calculation printout

    # Run the experiments
    run_distance_experiments(distances, args.iterations, args.results_dir)

    print(f"\nFinal memory usage: {monitor_memory():.2f} MB")
    print("All done!")
