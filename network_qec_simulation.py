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
from netsquid.qubits.qubit import Qubit

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

class SenderProtocol(NodeProtocol):
    """Protocol to prepare, encode, and send qubits."""
    def __init__(self, node, qec_experiment, port_name, num_qubits_to_send, name=None):
        super().__init__(node, name=name)
        self.qec_experiment = qec_experiment
        self.port_name = port_name
        self.num_qubits_to_send = num_qubits_to_send # Added to ensure correct sending logic
        print(f"SenderProtocol '{self.name}' initialized for node {node.name} on port {port_name}, sending {num_qubits_to_send} qubits.")


    def run(self):
        print(f"{ns.sim_time()}: SenderProtocol running on {self.node.name}")
        # Prepare qubit
        qubit = self.qec_experiment.prepare_qubit()
        initial_dm = ns.qubits.reduced_dm(qubit) # Store initial state for reference

        # Encode qubit
        # Pass the original qubit to encode
        encoded_qubits = self.qec_experiment.encode_qubit(qubit)

        # Verify number of encoded qubits matches expected
        if len(encoded_qubits) != self.num_qubits_to_send:
             print(f"Warning: Expected to send {self.num_qubits_to_send} qubits, but encoded {len(encoded_qubits)} for QEC method {self.qec_experiment.qec_method}")
             # Adjust if necessary, though ideally encode_qubit should be correct
             # self.num_qubits_to_send = len(encoded_qubits)

        print(f"{ns.sim_time()}: Sending {len(encoded_qubits)} encoded qubits...")
        # Send encoded qubits one by one
        for i, q in enumerate(encoded_qubits):
            # Send qubit via quantum channel port
            self.node.ports[self.port_name].tx_output(q)
            # Small delay between sending qubits to avoid potential simulation issues
            # This might not be physically realistic but helps in some simulation contexts.
            # Adjust or remove if needed. A proper physical model would involve timing
            # based on qubit generation/transmission rates.
            yield self.await_timer(1) # Wait 1 ns between sends

        print(f"{ns.sim_time()}: Sender finished sending qubits.")
        # Return the initial density matrix for fidelity calculation at the receiver
        return {"initial_dm": initial_dm}


class ReceiverProtocol(NodeProtocol):
    """Protocol to receive, decode qubits, and calculate fidelity."""
    def __init__(self, node, qec_experiment, port_name, num_qubits_to_receive, sender_protocol, name=None):
        super().__init__(node, name=name)
        self.qec_experiment = qec_experiment
        self.received_qubits = []
        self.port_name = port_name
        self.num_qubits_to_receive = num_qubits_to_receive
        self.sender_protocol = sender_protocol
        self.initial_dm_sender = None # To store the initial DM sent by sender
        print(f"ReceiverProtocol '{self.name}' initialized for node {node.name} on port {port_name}, expecting {num_qubits_to_receive} qubits.")

    def run(self):
        print(f"{ns.sim_time()}: ReceiverProtocol running on {self.node.name}")
        self.received_qubits = [] # Reset for each run

        # Wait for all expected qubits to arrive
        print(f"{ns.sim_time()}: Waiting for {self.num_qubits_to_receive} qubits...")
        for i in range(self.num_qubits_to_receive):
            yield self.await_port_input(self.node.ports[self.port_name])
            message = self.node.ports[self.port_name].rx_input()
            if message and message.items:
                qubit = message.items[0]
                if isinstance(qubit, ns.qubits.qubit.Qubit):
                    self.received_qubits.append(qubit)
                    # print(f"{ns.sim_time()}: Received qubit {i+1}/{self.num_qubits_to_receive}")
                else:
                     print(f"{ns.sim_time()}: Warning: Received non-qubit item: {type(qubit)}")
            else:
                 print(f"{ns.sim_time()}: Warning: Received empty or invalid message on qubit port at iteration {i}")


        print(f"{ns.sim_time()}: Received {len(self.received_qubits)} qubits.")

        if len(self.received_qubits) < self.num_qubits_to_receive:
            print(f"Error: Expected {self.num_qubits_to_receive} qubits but received {len(self.received_qubits)}. Cannot decode.")
            # Decide how to handle this - returning 0 fidelity might be appropriate
            return {
                "fidelity": 0.0, # Indicate failure
                "qec_method": self.qec_experiment.qec_method,
                "initial_state": self.qec_experiment.initial_state,
                "error": "Incorrect number of qubits received"
            }

        # Check if qubits were lost (this requires FibreLossModel to be active)
        # NetSquid's FibreLossModel might implicitly handle loss by not delivering the qubit,
        # leading to the length check above. Explicit checks might be needed for more complex loss scenarios.

        # Decode and correct errors
        print(f"{ns.sim_time()}: Decoding qubits...")
        # Ensure we pass a list copy if decode_qubit modifies the list
        decoded_qubit = self.qec_experiment.decode_qubit(list(self.received_qubits))
        print(f"{ns.sim_time()}: Decoding complete.")

        # Fetch the initial state DM from the sender's result using the stored instance
        # Check if the sender protocol is NOT running (i.e., it has finished)
        if not self.sender_protocol.is_running:
             sender_result = self.sender_protocol.get_signal_result(Signals.FINISHED)
             self.initial_dm_sender = sender_result.get("initial_dm")
        else:
             print(f"{ns.sim_time()}: Warning: Sender protocol '{self.sender_protocol.name}' not finished when receiver needs initial state.")
             # Attempt to get it anyway if possible, or handle error
             try:
                 sender_result = self.sender_protocol.get_signal_result(Signals.FINISHED) # Might raise error if not finished
                 self.initial_dm_sender = sender_result.get("initial_dm")
             except Exception as e:
                 print(f"Error getting sender result: {e}")
                 self.initial_dm_sender = None


        # Calculate fidelity
        if decoded_qubit is None:
             print(f"{ns.sim_time()}: Error: Decoded qubit is None.")
             fidelity = 0.0
        elif self.initial_dm_sender is None:
             print(f"{ns.sim_time()}: Error: Initial state from sender not available for fidelity calculation.")
             fidelity = 0.0 # Cannot calculate fidelity without reference
        else:
            print(f"{ns.sim_time()}: Calculating fidelity...")
            # Use the calculate_fidelity method, providing the initial DM as the reference
            fidelity = self.qec_experiment.calculate_fidelity(
                decoded_qubit,
                reference_state=self.initial_dm_sender
            )
            print(f"{ns.sim_time()}: Fidelity calculated: {fidelity:.4f}")


        return {
            "fidelity": fidelity,
            "qec_method": self.qec_experiment.qec_method,
            "initial_state": self.qec_experiment.initial_state
        }

class NetworkQEC:
    """
    Manages the NetSquid network setup and simulation run for a single QEC experiment.
    Uses channel-based noise models for errors during transmission.
    """
    def __init__(self,
                 initial_state="+",
                 error_type="depolarizing", # Default to depolarizing as bit-flip not native
                 error_params=None,
                 qec_method="shor_nine",
                 distance=1.0):
        self.initial_state = initial_state
        self.error_type = error_type
        self.error_params = error_params if error_params is not None else {}
        self.qec_method = qec_method
        self.distance = distance # in km

        # Determine number of physical qubits required by QEC method
        self.num_physical_qubits = self._get_num_qubits_for_qec(qec_method)

        # Setup network
        self.network = self._setup_network()

        # Create QEC experiment instance - used only for logic (prepare, encode, decode, fidelity)
        # Error type is set to "none" here as channel model handles errors.
        self.qec_logic = QuantumErrorCorrectionExperiment(
            initial_state=initial_state,
            error_type="none", # IMPORTANT: Channel applies error, not this instance
            qec_method=qec_method,
            distance=0 # Distance handled by channel
        )


    def _get_num_qubits_for_qec(self, qec_method):
        """Determine the number of physical qubits for a given QEC method."""
        if qec_method == "none":
            return 1
        elif qec_method in ["three_qubit_bit_flip", "three_qubit_phase_flip"]:
            return 3
        elif qec_method == "shor_nine":
            return 9
        # Add other codes like Steane if implemented
        # elif qec_method == "steane_seven":
        #     return 7
        else:
            raise ValueError(f"Unknown QEC method: {qec_method}")

    def _setup_network(self):
        """Set up a network with sender and receiver nodes and noisy channels."""
        ns.sim_reset() # Ensure clean simulation environment for each setup
        network = Network("QEC_Network")

        # Add nodes
        sender = Node("sender")
        receiver = Node("receiver")
        network.add_node(sender)
        network.add_node(receiver)

        # --- Configure Quantum Channel (Sender to Receiver) ---
        qchannel = QuantumChannel(
            name="qchannel_S_to_R",
            length=self.distance # Distance in km
            # models = {} # Initialize models dict - THIS IS KEY
        )
        # Ensure the models dictionary exists
        if not hasattr(qchannel, 'models') or qchannel.models is None:
            qchannel.models = {}


        # 1. Delay Model: Simulates travel time based on distance
        qchannel.models["delay_model"] = FibreDelayModel(c=200000.0) # c in km/s

        # 2. Quantum Error Model: Applies noise during transmission
        error_model = self._get_netsquid_error_model()
        if error_model:
            qchannel.models["quantum_noise_model"] = error_model
            print(f"Attached noise model {type(error_model).__name__} to quantum channel.")
        else:
            print("No noise model attached to quantum channel.")


        # 3. Loss Model (Optional but recommended for realism)
        # Check if fibre_loss is the selected error type or add it alongside others
        if self.error_type == "fibre_loss":
             p_loss_init = self.error_params.get("p_loss_init", 0.0) # Loss at input coupling
             p_loss_length = self.error_params.get("p_loss_length", 0.2) # Loss per km (e.g., 0.2 dB/km)
             loss_model = FibreLossModel(p_loss_init=p_loss_init, p_loss_length=p_loss_length)
             # If a noise model already exists, combine them (if supported/makes sense)
             # For simplicity here, we might overwrite or only use loss if specified.
             # A cleaner way might be to have separate params for noise and loss.
             # Let's attach it:
             qchannel.models["quantum_loss_model"] = loss_model # Use a different key if noise model also present
             print(f"Attached loss model {type(loss_model).__name__} to quantum channel.")
        # Note: If you want both noise AND loss, NetSquid allows multiple models.
        # You might need to adjust keys, e.g., qchannel.models["noise"] = ..., qchannel.models["loss"] = ...
        # and ensure the channel processes them. Default QuantumChannel processes "quantum_noise_model".
        # Check NetSquid docs for combining models if needed.


        # --- Configure Classical Channel (Receiver to Sender - for potential acknowledgements etc.) ---
        # Although not used by the current simple protocols, it's good practice for network setup.
        cchannel = ClassicalChannel(
            name="cchannel_R_to_S",
            length=self.distance # km
            # models = {}
        )
        if not hasattr(cchannel, 'models') or cchannel.models is None:
            cchannel.models = {}
        cchannel.models["delay_model"] = FibreDelayModel(c=200000.0) # km/s


        # Create a direct connection using these channels
        # Connection from A (sender) to B (receiver) uses qchannel
        # Connection from B (receiver) to A (sender) uses cchannel
        connection = DirectConnection(
            name="qec_connection",
            channel_AtoB=qchannel,
            channel_BtoA=cchannel # Classical channel for reverse communication
        )

        # Connect nodes using NetSquid's network connection method
        # This automatically creates and connects ports on the nodes.
        port_name_sender, port_name_receiver = network.add_connection(
            sender.name, receiver.name, connection=connection, label="q_comm" # Use a label
        )

        # Store port names for protocol initialization
        self.port_name_sender = port_name_sender
        self.port_name_receiver = port_name_receiver

        print(f"Network setup complete. Sender port: {port_name_sender}, Receiver port: {port_name_receiver}")
        return network

    def _get_netsquid_error_model(self):
        """Maps the experiment's error type string to a NetSquid QuantumErrorModel."""
        error_model = None
        p = self.error_params.get("probability") # Used by several models
        gamma = self.error_params.get("gamma") # Used by damping models

        # Convert probability/gamma to rates if models expect rates (Hz) and are time-dependent.
        # However, the original code used time_independent=True for prob-based models,
        # implying the 'rate' parameter is directly treated as probability per qubit.
        # T1/T2 models inherently use rates (1/T1, 1/T2).

        print(f"Configuring error model for type: {self.error_type}, params: {self.error_params}")

        if self.error_type == "depolarizing":
            if p is None: p = 0.1 # Default
            # DepolarNoiseModel interprets 'depolar_rate' as probability if time_independent=True
            error_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
            print(f"Using DepolarNoiseModel with probability P={p}")

        elif self.error_type == "phase_flip":
             if p is None: p = 0.1
             # DephaseNoiseModel interprets 'dephase_rate' as probability if time_independent=True
             error_model = DephaseNoiseModel(dephase_rate=p, time_independent=True)
             print(f"Using DephaseNoiseModel with probability P={p}")

        elif self.error_type == "amplitude_damping":
            if gamma is None: gamma = 0.1
            t1 = 1e9 / gamma if gamma > 0 else float('inf') # Convert rate (Hz assumed) to T1 (ns)
            # To approximate pure amplitude damping with T1/T2 model, use T2 = 2*T1.
            t2 = t1
            error_model = T1T2NoiseModel(T1=t1, T2=t2)
            print(f"Using T1T2NoiseModel for Amplitude Damping: T1={t1:.2e} ns, T2={t2:.2e} ns (gamma={gamma}/s)")


        elif self.error_type == "phase_damping":
            if gamma is None: gamma = 0.1
            t2 = 1e9 / gamma if gamma > 0 else float('inf') # Convert rate (Hz assumed) to T2 (ns)
            # Pure phase damping corresponds to T1=inf
            t1 = float('inf')
            error_model = T1T2NoiseModel(T1=t1, T2=t2)
            print(f"Using T1T2NoiseModel for Phase Damping: T1=inf, T2={t2:.2e} ns (gamma={gamma}/s)")


        elif self.error_type == "gen_amplitude_damping":
             # GAD is complex (2 params: gamma, p_thermal). T1/T2 is an approximation.
             # The original paper defines GAD with specific Kraus operators.
             # We might approximate it with T1/T2, but it's not exact.
             # Let's use the same T1/T2 as amplitude damping for now, acknowledging it's an approximation.
             if gamma is None: gamma = 0.1
             p_param = self.error_params.get("probability", 0.5) # This 'p' is part of GAD, not error prob.
             t1 = 1e9 / gamma if gamma > 0 else float('inf') # T1 relates to gamma
             # How p_param relates to T2 is complex. Let's stick to T2=2*T1 as a basic approximation.
             t2 = 2 * t1
             error_model = T1T2NoiseModel(T1=t1, T2=t1)
             print(f"Warning: Approximating Generalized Amplitude Damping using T1T2NoiseModel: T1={t1:.2e} ns, T2={t2:.2e} ns (gamma={gamma}/s, p={p_param}) - This is an approximation.")


        # --- Fallback for unsupported types ---
        # Bit-flip, Bit-phase-flip, Coherent Overrotation, Mixed XZ don't have direct time_independent prob models in qerrormodels.
        # They could be implemented with custom models or approximated.
        # We fall back to Depolarizing noise as per the original script's logic.
        elif self.error_type in ["bit_flip", "bit_phase_flip", "coherent_overrotation", "mixed_xz", "fibre_loss"]:
             # Fibre loss is handled separately, so no noise model here if that's the primary type.
             if self.error_type == "fibre_loss":
                  print(f"Note: Using FibreLossModel (handled separately), no additional noise model for '{self.error_type}'.")
             else:
                 print(f"Warning: Error type '{self.error_type}' not directly mapped to a standard NetSquid channel noise model. Falling back to Depolarizing noise.")
                 if p is None: p = 0.1 # Default probability
                 error_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
                 print(f"Using Fallback DepolarNoiseModel with probability P={p}")

        else:
             print(f"Warning: Unknown error type '{self.error_type}'. No noise model applied.")

        return error_model


    def run_simulation(self):
        """Run the QEC simulation using protocols."""
        print(f"\nStarting simulation run: QEC={self.qec_method}, Error={self.error_type}, Params={self.error_params}, Dist={self.distance} km")
        # Retrieve nodes from the network
        sender_node = self.network.get_node("sender")
        receiver_node = self.network.get_node("receiver")

        # Create protocols for this run
        sender_protocol = SenderProtocol(
            node=sender_node,
            qec_experiment=self.qec_logic,
            port_name=self.port_name_sender,
            num_qubits_to_send=self.num_physical_qubits,
            name="SenderProtocol" # Give specific name
        )
        receiver_protocol = ReceiverProtocol(
            node=receiver_node,
            qec_experiment=self.qec_logic,
            port_name=self.port_name_receiver,
            num_qubits_to_receive=self.num_physical_qubits,
            sender_protocol=sender_protocol, # Pass the sender instance here
            name="ReceiverProtocol" # Give specific name
        )

        # Start protocols
        sender_protocol.start()
        receiver_protocol.start()

        # Run the simulation engine
        # ns.sim_run(duration=self.distance * 5000 * 2) # Estimate runtime based on distance (ns)
        ns.sim_run() # Run until protocols complete

        # Get results from receiver protocol's finished signal
        # Check if the receiver protocol is NOT running (i.e., finished)
        if not receiver_protocol.is_running:
             result = receiver_protocol.get_signal_result(Signals.FINISHED)
             print(f"Simulation run finished. Result: {result}")
             return result
        else:
             print("Error: Receiver protocol did not finish.")
             # Check sender status too
             if sender_protocol.is_finished:
                  print("Sender protocol finished.")
             else:
                  print("Sender protocol also did not finish.")

             return {
                 "fidelity": 0.0, # Indicate failure
                 "qec_method": self.qec_logic.qec_method,
                 "initial_state": self.qec_logic.initial_state,
                 "error": "Protocol did not finish"
             }


# --- Functions for Running Batch Experiments (Mostly unchanged from original) ---

def run_single_experiment_with_streaming(initial_state, error_type, error_params, qec_method,
                                         distance=1.0, num_iterations=10, gc_every=5, base_dir="results_network_qec_v2"):
    """Run a single experiment configuration for multiple iterations, streaming results."""
    # Create directory structure
    error_type_dir = os.path.join(base_dir, error_type.replace(" ", "_")) # Sanitize name
    os.makedirs(error_type_dir, exist_ok=True)

    # Create parameter string for filename
    param_str = "_".join([f"{k}_{v}" for k, v in sorted(error_params.items())]) # Sort for consistency
    param_str = param_str.replace(" ", "_").replace("=", "_") # Sanitize

    # Create filename
    filename = f"{initial_state}_{qec_method}_d{distance}_{param_str}.csv"
    filepath = os.path.join(error_type_dir, filename)

    # Check if file exists and has enough entries (simple check)
    try:
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            if len(df_existing) >= num_iterations:
                print(f"Skipping (already completed): {filepath} with {len(df_existing)} iterations.")
                return None # Skip
            else:
                 print(f"Resuming experiment: {filepath} (found {len(df_existing)}/{num_iterations} iterations).")
                 # We will append, but need to track remaining iterations
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


    # Open file for writing/appending
    with open(filepath, mode=file_mode, newline='') as f:
        # Write header only if creating a new file
        if write_header:
            f.write("iteration,fidelity")

        # Statistics accumulators (only for this run's iterations)
        stats = { "sum_fidelity": 0.0, "sum_squared_fidelity": 0.0, "count": 0 }

        print(f"Running Experiment: State={initial_state}, Error={error_type}, QEC={qec_method}, Params={error_params}, Dist={distance} km")
        start_memory = monitor_memory()
        print(f"Memory at start: {start_memory:.2f} MB")

        # Run iterations with progress bar
        pbar = tqdm(range(iterations_to_run), desc=f"{initial_state},{error_type},{qec_method},d={distance}")
        for i in pbar:
            current_iteration_index = start_iteration + i

            # Force garbage collection regularly
            if i % gc_every == 0:
                gc.collect()

            try:
                # --- Simulation Core ---
                # 1. Reset NetSquid simulation environment FOR EACH ITERATION
                ns.sim_reset()

                # 2. Create NetworkQEC instance for this iteration's parameters
                experiment = NetworkQEC(
                    initial_state=initial_state,
                    error_type=error_type,
                    error_params=error_params,
                    qec_method=qec_method,
                    distance=distance
                )

                # 3. Run the simulation protocols
                result = experiment.run_simulation()

                # 4. Extract fidelity
                fidelity = result.get("fidelity", 0.0) # Default to 0 if key missing or error occurred

                # Check for simulation errors reported by the protocol
                sim_error = result.get("error")
                if sim_error:
                    print(f"Warning: Simulation error reported in iteration {current_iteration_index}: {sim_error}")
                    # Optionally write errors to a log or handle differently
                    # For now, we record the fidelity (likely 0)

                # Update running statistics for this batch
                stats["sum_fidelity"] += fidelity
                stats["sum_squared_fidelity"] += fidelity**2
                stats["count"] += 1

                # Write result directly to file
                f.write(f"{current_iteration_index},{fidelity}\n")
                f.flush() # Ensure data is written immediately


                # Explicitly delete the experiment object and clear protocols
                # ns.sim_reset() handles protocol stopping and removal, so explicit stopping is removed.
                #if 'experiment' in locals() and experiment.network:
                #     # Need to properly remove protocols before deleting the network instance
                #     sender_node = experiment.network.get_node("sender")
                #     receiver_node = experiment.network.get_node("receiver")
                #     if "SenderProtocol" in sender_node.protocols:
                #         sender_node.protocols["SenderProtocol"].stop()
                #     if "ReceiverProtocol" in receiver_node.protocols:
                #         receiver_node.protocols["ReceiverProtocol"].stop()

                del experiment # Delete the NetworkQEC instance
                ns.sim_reset() # Reset again to clear any leftover state


                # More aggressive GC for complex QEC codes might help
                if qec_method in ["shor_nine"] and i % (gc_every // 2 + 1) == 0:
                    gc.collect()

            except Exception as e:
                print(f"-----------------------------------------------------")
                print(f"ERROR in iteration {current_iteration_index} ({initial_state}, {error_type}, {qec_method}, d={distance}):")
                import traceback
                traceback.print_exc()
                print(f"-----------------------------------------------------")
                # Log error to a separate file
                error_log_path = os.path.join(base_dir, "error_log.txt")
                with open(error_log_path, "a") as error_f:
                    error_f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    error_f.write(f"Params: State={initial_state}, Error={error_type}, QEC={qec_method}, Params={error_params}, Dist={distance}, Iteration={current_iteration_index}\n")
                    error_f.write(f"Error: {str(e)}\n")
                    error_f.write(traceback.format_exc())
                    error_f.write("-" * 20 + "\n")

                # Re-raise the exception to stop the script
                raise e

            # Update progress bar description with current avg fidelity
            if stats["count"] > 0:
                 avg_fid_so_far = stats["sum_fidelity"] / stats["count"]
                 pbar.set_postfix({"avg_fid": f"{avg_fid_so_far:.3f}"})


    # --- Post-Experiment Processing for this Configuration ---
    # Read the full CSV to calculate final stats
    try:
         df_final = pd.read_csv(filepath)
         count = len(df_final)
         if count > 0:
             avg_fidelity = df_final["fidelity"].mean()
             std_fidelity = df_final["fidelity"].std()
         else:
             avg_fidelity = std_fidelity = 0.0

         # Save metadata (consider saving only once after all iterations)
         metadata = {
             "initial_state": initial_state, "error_type": error_type,
             "error_params": str(error_params), "qec_method": qec_method,
             "distance": distance, "avg_fidelity": avg_fidelity,
             "std_fidelity": std_fidelity, "iterations_completed": count
         }
         metadata_df = pd.DataFrame([metadata])
         metadata_path = filepath.replace(".csv", "_metadata.csv")
         metadata_df.to_csv(metadata_path, index=False)

         current_memory = monitor_memory()
         print(f"Finished experiment config. Final avg fidelity: {avg_fidelity:.4f} +/- {std_fidelity:.4f}")
         print(f"Results saved to: {filepath}")
         print(f"Metadata saved to: {metadata_path}")
         print(f"Memory usage end: {current_memory:.2f} MB (Delta: {current_memory - start_memory:.2f} MB)")

         # Force cleanup
         gc.collect()
         return metadata

    except Exception as e:
        print(f"Error calculating final stats or saving metadata for {filepath}: {e}")
        gc.collect()
        return None # Indicate failure

def process_error_type(error_type, distances, initial_states, qec_methods, error_params_list,
                       num_iterations=10, base_dir="results_network_qec_v2"):
    """Process all experiments for a single error type."""
    print(f"{'='*20} Processing Error Type: {error_type} {'='*20}")

    # Count total experiments for this error type
    total_exps = len(initial_states) * len(qec_methods) * len(error_params_list) * len(distances)
    print(f"Total experiment configurations for {error_type}: {total_exps}")

    completed_configs = 0
    start_time_error_type = time.time()

    # Iterate through all configurations for this error type
    for initial_state in initial_states:
        # Prioritize less computationally expensive methods if desired
        # methods_order = ["none", "three_qubit_bit_flip", "three_qubit_phase_flip", "shor_nine"] # Example order
        for qec_method in qec_methods:
            for error_params in error_params_list:
                for distance in distances:
                    # --- Run the single experiment configuration ---
                    run_single_experiment_with_streaming(
                        initial_state=initial_state,
                        error_type=error_type,
                        error_params=error_params,
                        qec_method=qec_method,
                        distance=distance,
                        num_iterations=num_iterations,
                        base_dir=base_dir
                        # gc_every can be adjusted here if needed
                    )
                    # ---------------------------------------------

                    completed_configs += 1
                    elapsed_error_type = time.time() - start_time_error_type
                    if completed_configs > 0:
                        avg_time_per_config = elapsed_error_type / completed_configs
                        remaining_configs = total_exps - completed_configs
                        est_remaining_time = avg_time_per_config * remaining_configs
                        print(f"--- Config {completed_configs}/{total_exps} ({completed_configs/total_exps*100:.1f}%) complete for {error_type} ---")
                        if est_remaining_time > 0:
                             print(f"--- Estimated time remaining for {error_type}: {est_remaining_time/60:.1f} minutes ---")

                    # Explicit GC after each configuration finishes
                    gc.collect()
                    time.sleep(0.1) # Small pause

    print(f"{'='*20} Finished Processing Error Type: {error_type} {'='*20}")
    return True


def define_experiment_parameters(error_type):
     """Returns a list of error parameter dictionaries for a given error type."""
     params_list = []
     # Define default ranges
     default_probabilities = [0.01, 0.05, 0.1, 0.2]
     # Define gamma values for faster decoherence (kHz to MHz range)
     fast_gammas = [1e4, 1e5, 1e6, 1e7] # 10kHz, 100kHz, 1MHz, 10MHz
 
     if error_type in ["bit_flip", "phase_flip", "bit_phase_flip", "depolarizing", "mixed_xz"]:
         # These are often modeled with a single probability parameter 'p'
         # Note: Our implementation maps bit_flip, bit_phase_flip, mixed_xz to depolarizing
         params_list = [{"probability": p} for p in default_probabilities]
     elif error_type in ["amplitude_damping", "phase_damping"]:
         # These use a 'gamma' parameter (related to T1 or T2)
         # Use the faster gamma values specifically for these types
         print(f"Using fast gamma values for {error_type}: {fast_gammas}")
         params_list = [{"gamma": g} for g in fast_gammas]
     elif error_type == "gen_amplitude_damping":
         # Uses gamma and a thermal probability 'p' (often fixed, e.g., 0.5)
         # Decide if you want fast gammas here too, or default
         # Using fast gammas for GAD as well for this example:
         print(f"Using fast gamma values for {error_type}: {fast_gammas}")
         params_list = [{"gamma": g, "probability": 0.5} for g in fast_gammas]
         # Could also vary the 'probability' parameter if desired:
         # params_list = [{"gamma": g, "probability": p_therm} for g in gammas for p_therm in [0.25, 0.5, 0.75]]
     elif error_type == "coherent_overrotation":
          # Uses rotation angle 'theta' and 'axis'
          # Note: Our implementation maps this to depolarizing
          thetas = [0.05, 0.1, 0.2] # Radians
          axes = ["X", "Y", "Z"]
          # Use default probabilities if mapped to depolarizing
          params_list = [{"probability": p} for p in default_probabilities]
     elif error_type == "fibre_loss":
          # Uses initial loss probability and loss per length (dB/km)
          p_loss_inits = [0.0, 0.1] # E.g., coupling loss
          p_loss_lengths = [0.16, 0.20, 0.25] # dB/km for different fiber types/wavelengths
          params_list = [{"p_loss_init": pi, "p_loss_length": pl} for pi in p_loss_inits for pl in p_loss_lengths]

     else:
          print(f"Warning: No specific parameters defined for error type '{error_type}'. Using default empty dict.")
          params_list = [{}] # Default if no specific params needed or type unknown

     return params_list


def run_all_network_qec_experiments(distances=[1.0, 10.0], num_iterations=10, base_dir="results_network_qec_v2"):
    """Runs the full suite of QEC experiments across different parameters."""
    print("=============================================")
    print(" Starting Full Network QEC Experiment Suite")
    print("=============================================")
    print(f"Parameters: Distances={distances} km, Iterations/config={num_iterations}")
    print(f"Results directory: {base_dir}")

    # --- Define Experiment Space --- #
    initial_states = ["+", "-"]
    print(f"\n>>> Running ONLY for Initial States: {initial_states} <<<")
    qec_methods = ["none", "three_qubit_bit_flip", "three_qubit_phase_flip", "shor_nine"]
    # Add Steane code here if/when implemented and tested in single_qubit_experiments.py

    # --- Modify Error Types to Test --- #
    # Original list:
    # error_types_to_test = [
    #     "amplitude_damping",
    #     "phase_damping",
    # ]
    # New list for only bit_flip:
    error_types_to_test = [
        "bit_flip"
    ]
    print(f"\n>>> Running ONLY for Error Types: {error_types_to_test} <<<")
    # --- End Modification --- #

    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    overall_start_time = time.time()

    # Process one error type at a time to manage memory and organization
    for error_type in error_types_to_test:
        # Get the list of parameter combinations for this error type
        error_params_list = define_experiment_parameters(error_type)
        if not error_params_list:
             print(f"Skipping error type {error_type} due to empty parameter list.")
             continue

        # Clear memory before starting a new error type batch
        gc.collect()
        time.sleep(1) # Pause briefly

        # Run all experiments for this error type
        process_error_type(
            error_type=error_type,
            distances=distances,
            initial_states=initial_states,
            qec_methods=qec_methods,
            error_params_list=error_params_list,
            num_iterations=num_iterations,
            base_dir=base_dir
        )

        # More aggressive GC between major error type batches
        for _ in range(3):
            gc.collect()
            time.sleep(0.5)

    overall_end_time = time.time()
    total_time_secs = overall_end_time - overall_start_time
    print("=============================================")
    print(f" All Experiment Suites Completed!")
    print(f" Total execution time: {total_time_secs:.2f} seconds ({total_time_secs/3600:.2f} hours)")
    print("=============================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Memory-Optimized Network QEC Simulations')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations per experiment configuration')
    parser.add_argument('--results-dir', type=str, default='results_network_qec_v2', help='Directory to save results')
    parser.add_argument('--distances', type=str, default='1.0,10.0', help='Comma-separated list of distances (km), e.g., "1.0,10.0,50.0"')
    # Add options for selective runs if needed (e.g., specific error type, QEC method)
    # parser.add_argument('--error-type', type=str, help='Run only a specific error type')
    # parser.add_argument('--qec-method', type=str, help='Run only a specific QEC method')

    args = parser.parse_args()

    # Parse distances
    try:
        distances = [float(d.strip()) for d in args.distances.split(',') if d.strip()]
        if not distances: raise ValueError("No valid distances provided.")
    except Exception as e:
        print(f"Error parsing distances argument '{args.distances}': {e}")
        print("Please provide a comma-separated list of numbers (e.g., '1.0,10.0').")
        exit(1)

    print(f"Starting Network QEC Simulations...")
    print(f"Iterations per config: {args.iterations}")
    print(f"Distances (km): {distances}")
    print(f"Results directory: {args.results_dir}")
    print(f"Initial memory usage: {monitor_memory():.2f} MB")

    # Run the full suite
    run_all_network_qec_experiments(
        distances=distances,
        num_iterations=args.iterations,
        base_dir=args.results_dir
    )

    print(f"Final memory usage: {monitor_memory():.2f} MB")
    print("Simulation suite finished.") 