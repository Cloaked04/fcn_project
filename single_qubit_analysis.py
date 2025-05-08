import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib as mpl
import warnings

# --- Configuration ---
# Make sure this points to the correct directory containing the error_type subfolders
RESULTS_DIR = "results_network_qec_v2"
PLOTS_DIR = "plots_network_qec_v2"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Plotting Configuration (from original notebook) ---
def configure_plots():
    """Configure matplotlib rcParams for publication-quality plots"""
    plt.style.use('seaborn-v0_8-whitegrid') # Use an available style
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'figure.figsize': (11, 7), # Adjusted figure size
        'figure.dpi': 300,
        'text.usetex': False,  # Keep False for broader compatibility
        'mathtext.fontset': 'stix', # Or 'cm' if text.usetex is True
        'axes.prop_cycle': plt.cycler('color', plt.cm.viridis(np.linspace(0, 0.85, 8))), # Sample 8 colors from viridis
        'axes.axisbelow': True,
        'grid.alpha': 0.6,
        'grid.linestyle': ':'
    })

configure_plots() # Apply the configuration
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib') # Ignore minor warnings

# --- Data Loading Function ---
def parse_param_string(param_str):
    """Parses parameter strings like 'gamma_0.1' or 'p_loss_init_0.0_p_loss_length_0.16'"""
    params = {}
    parts = param_str.split('_')
    if not parts:
        return params
    i = 0
    while i < len(parts):
        # Combine parts until the next part looks like a number (or is the last part)
        key = parts[i]
        j = i + 1
        # Look ahead: check if the next part starts a new key=value pair
        # This is tricky, assume value is the last part or the part before the next non-numeric key
        while j < len(parts) - 1:
            # If the part after the potential value (parts[j+1]) looks like a key (alphabetic start)
            # and parts[j] looks like a value (numeric start, or contains '.', or is '-'), stop combining.
            is_numeric_val = parts[j][0].isdigit() or '.' in parts[j] or parts[j].startswith('-')
            next_is_key = j+1 < len(parts) and parts[j+1] and parts[j+1][0].isalpha()

            if is_numeric_val and next_is_key:
                 break # parts[j] is likely the value for the current key

            # Otherwise, combine this part into the key
            key += "_" + parts[j]
            j += 1

        value_str = parts[j]
        try:
             value = float(value_str)
        except ValueError:
            value = value_str # Keep as string if not float

        params[key] = value
        i = j + 1 # Move to the start of the next potential key
    return params

def load_network_qec_data(results_dir=RESULTS_DIR, load_full_data=False):
    """
    Loads experiment data from the network_qec_simulation results structure.
    Handles potentially malformed headers and assumes data rows have iteration, fidelity.

    Args:
        results_dir (str): The path to the main results directory.
        load_full_data (bool): If True, loads the full DataFrame for each file.
                               Otherwise, only loads metrics. Defaults to False.

    Returns:
        list: A list of dictionaries, where each dictionary represents one
              experiment run and contains its parameters and loaded data/metrics.
    """
    print(f"Loading experiment data from: {results_dir} (Load full data: {load_full_data})")
    all_experiments_data = []
    skipped_files = []
    loaded_count = 0
    file_parse_errors = []

    filename_pattern = re.compile(r"^([+\-01])_(.+?)_d([\d\.]+?)_(.+)\.csv$")
    error_type_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for error_type in error_type_dirs:
        error_dir_path = os.path.join(results_dir, error_type)
        # print(f" Processing directory: {error_dir_path}") # Reduce verbosity

        for filename in os.listdir(error_dir_path):
            if not filename.endswith(".csv") or filename.endswith("_metadata.csv"):
                continue

            file_path = os.path.join(error_dir_path, filename)

            # --- Check 1: File Size --- #
            try:
                if os.path.getsize(file_path) == 0:
                    # print(f"  - Skipping 0-byte file: {filename}")
                    skipped_files.append(filename)
                    continue
            except OSError as size_e:
                # print(f"  - Error checking size for {filename}: {size_e}")
                file_parse_errors.append((filename, f"Size check error: {size_e}"))
                skipped_files.append(filename)
                continue
            # --- End Check 1 --- #

            match = filename_pattern.match(filename)
            if match:
                initial_state = match.group(1)
                qec_method = match.group(2)
                distance = float(match.group(3))
                param_string = match.group(4)

                try:
                    error_params = parse_param_string(param_string)
                    if not error_params:
                        # print(f"  - Warning: Could not parse parameters from '{param_string}' in file {filename}")
                        skipped_files.append(filename)
                        continue

                    try:
                        # --- Modified pd.read_csv (Attempt 2) --- #
                        data_df = pd.read_csv(
                            file_path,
                            header=None,         # Still ignore the actual header row
                            skiprows=1,          # Still skip the first row
                            names=['iteration', 'fidelity'], # Expect exactly two columns
                            # usecols removed as we expect exactly the named columns
                            on_bad_lines='warn'
                        )
                        # --- End of modification --- #

                        # --- Check 2: Empty DataFrame after load? --- #
                        if data_df.empty:
                            # print(f"  - Warning: DataFrame empty after loading (check content/header): {filename}")
                            skipped_files.append(filename)
                            continue
                        # --- End Check 2 --- #

                        # --- Check 3: Drop NaNs and check again --- #
                        # Ensure fidelity column exists before trying to coerce/dropna
                        if 'fidelity' not in data_df.columns:
                            # print(f"  - Warning: 'fidelity' column not found after loading: {filename}")
                            skipped_files.append(filename)
                            continue

                        # Convert fidelity to numeric, coercing errors
                        data_df['fidelity'] = pd.to_numeric(data_df['fidelity'], errors='coerce')
                        data_df.dropna(subset=['fidelity'], inplace=True) # Drop rows where fidelity is NaN

                        if data_df.empty:
                            # print(f"  - Warning: No valid numeric fidelity data after cleaning: {filename}")
                            skipped_files.append(filename)
                            continue
                        # --- End Check 3 --- #

                        # --- Metadata Loading (keep as is) --- #
                        metadata_df = None
                        metadata_path = file_path.replace(".csv", "_metadata.csv")
                        if os.path.exists(metadata_path):
                             try:
                                 metadata_df = pd.read_csv(metadata_path).iloc[0].to_dict()
                             except Exception as meta_e:
                                 print(f"  - Warning: Could not load metadata file {metadata_path}: {meta_e}")
                        # --- End Metadata Loading --- #

                        # --- Calculate metrics --- #
                        avg_fidelity = data_df['fidelity'].mean()
                        std_fidelity = data_df['fidelity'].std()

                        experiment_entry = {
                             "initial_state": initial_state,
                             "qec_method": qec_method,
                             "distance": distance,
                             "error_type": error_type,
                             "error_params": error_params,
                             "avg_fidelity": avg_fidelity,
                             "std_fidelity": std_fidelity,
                             "num_iterations": len(data_df),
                             "metadata": metadata_df,
                             "filename": filename
                        }
                        if load_full_data:
                            experiment_entry["data"] = data_df

                        all_experiments_data.append(experiment_entry)
                        loaded_count += 1

                    except Exception as e:
                        # Catch errors during pd.read_csv or subsequent checks
                        # print(f"  - Error processing file content for {filename}: {e}")
                        file_parse_errors.append((filename, str(e)))
                        skipped_files.append(filename)

                except Exception as parse_e:
                    # print(f"  - Error parsing parameters for {filename}: {parse_e}")
                    file_parse_errors.append((filename, f"Param parsing error: {parse_e}"))
                    skipped_files.append(filename)
            else:
                skipped_files.append(filename)

    print(f"Finished loading. Loaded {loaded_count} experiment configurations.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files (check format/content/errors).")
    if file_parse_errors:
        print(f"Encountered {len(file_parse_errors)} errors during file processing:")
        # --- Print specific errors --- #
        for fname, err in file_parse_errors[:10]: # Print first 10 errors
             print(f"  - {fname}: {err}")
        if len(file_parse_errors) > 10:
             print("  ... (additional errors truncated)")
        # --- End error printing --- #
    return all_experiments_data

# Load the data
all_data = load_network_qec_data()
print(f"\nTotal number of loaded experiment configurations: {len(all_data)}")
# Example: Print details of the first loaded experiment
if all_data:
    print("\nExample loaded data point:")
    print(f"  State: {all_data[0]['initial_state']}")
    print(f"  QEC: {all_data[0]['qec_method']}")
    print(f"  Dist: {all_data[0]['distance']}")
    print(f"  Error Type: {all_data[0]['error_type']}")
    print(f"  Error Params: {all_data[0]['error_params']}")
    # Safely check if 'data' key exists (depends on load_full_data flag)
    if 'data' in all_data[0]:
        print(f"  Data points: {len(all_data[0]['data'])}")
    else:
        print(f"  Data points: (Full data not loaded)")
    print(f"  Avg Fidelity: {all_data[0]['avg_fidelity']:.4f}")
    # print(f"  Metadata: {all_data[0]['metadata']}") # Can be long

# --- Analysis Functions ---

def aggregate_and_plot_error_types(df, plots_subdir="error_type_analysis"):
    """
    Analyzes and plots results grouped by error type.
    - Overall fidelity per error type.
    - Fidelity per error type broken down by initial state.
    - Fidelity vs. error parameters (scatter plot) for baseline (no QEC).
    """
    print("\n--- Analyzing by Error Type ---")
    error_plots_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(error_plots_dir, exist_ok=True)

    # 1. Overall Fidelity by Error Type
    print("  Plotting overall fidelity...")
    overall_metrics = df.groupby('error_type').agg(
        avg_fidelity=('avg_fidelity', 'mean'),
        std_fidelity=('avg_fidelity', 'std') # Std dev of the *average* fidelities per config
    ).sort_values('avg_fidelity', ascending=False)

    plt.figure()
    ax = plt.gca()
    # Use explicit bar plot for better color control from cycle
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(overall_metrics)))
    bars = ax.bar(overall_metrics.index, overall_metrics['avg_fidelity'],
            yerr=overall_metrics['std_fidelity'], capsize=4, alpha=0.85,
            color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(r'Overall Average Fidelity by Error Type')
    ax.set_ylabel(r'Average Fidelity $F$')
    ax.set_xlabel(r'Error Type')

    # Add std dev text annotations
    for bar, std in zip(bars, overall_metrics['std_fidelity']):
        height = bar.get_height()
        if pd.notna(std):
            ax.text(bar.get_x() + bar.get_width() / 2., height + (overall_metrics['std_fidelity'].max() * 0.05 if pd.notna(overall_metrics['std_fidelity'].max()) else 0.01), # Adjust vertical position slightly based on max std dev
                    rf'$\pm{std:.2f}$',
                    ha='center', va='bottom', fontsize=8, color='dimgray')

    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_plots_dir, "overall_fidelity_by_error_type.png"))
    plt.close()

    # 2. Fidelity by Error Type and Initial State
    print("  Plotting fidelity by error type and state...")
    # Use Mathtext-compatible ket notation
    state_display_map = {'0': r'$|0\rangle$', '1': r'$|1\rangle$', '+': r'$|+\rangle$', '-': r'$|-\rangle$'}
    state_error_metrics = df.groupby(['error_type', 'initial_state']).agg(
        avg_fidelity=('avg_fidelity', 'mean'),
        std_fidelity=('avg_fidelity', 'std')
    ).unstack() # Pivot initial_state to columns

    if not state_error_metrics.empty:
        fig, ax = plt.subplots(figsize=(15, 8)) # Slightly wider figure
        num_states = len(state_error_metrics['avg_fidelity'].columns)
        colors = plt.cm.viridis(np.linspace(0, 0.85, num_states))

        # Plot bars manually for better control over labels and annotations
        bar_width = 0.8 / num_states
        x = np.arange(len(state_error_metrics.index))

        for i, state in enumerate(state_error_metrics['avg_fidelity'].columns):
            means = state_error_metrics[('avg_fidelity', state)]
            stds = state_error_metrics[('std_fidelity', state)]
            positions = x - (num_states / 2 - i - 0.5) * bar_width

            bars = ax.bar(positions, means, bar_width, yerr=stds,
                          label=state_display_map.get(state, state), # Use LaTeX labels
                          color=colors[i], alpha=0.85, capsize=3,
                          edgecolor='black', linewidth=0.5)

            # Add std dev text annotations for each bar group
            for bar, std_val in zip(bars, stds):
                height = bar.get_height()
                y_err_val = std_val if pd.notna(std_val) else 0
                if pd.notna(height):
                    # Position text slightly above the bar top, aligned below the specified point
                    y_pos = min(height + 0.01, 1.02)
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                            rf'$\pm{std_val:.2f}$' if pd.notna(std_val) else '', # Use raw f-string
                            ha='center', va='bottom', fontsize=7, color='dimgray', clip_on=True)

        ax.set_title(r'Average Fidelity by Error Type and Initial State')
        ax.set_ylabel(r'Average Fidelity $F$')
        ax.set_xlabel(r'Error Type')
        ax.set_xticks(x)
        ax.set_xticklabels(state_error_metrics.index, rotation=45, ha='right')

        # Adjust legend position
        ax.legend(title=r'Initial State', loc='best')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(error_plots_dir, "fidelity_by_error_and_state.png"))
        plt.close()

    # 3. Fidelity vs. Error Parameter (Scatter Plot for No QEC)
    print("  Plotting fidelity vs. error parameters (Scatter - No QEC)...")
    unique_error_types = df['error_type'].unique()
    df_no_qec = df[df['qec_method'] == 'none'].copy()

    if df_no_qec.empty:
         print("    No data found for 'qec_method = none'. Skipping parameter plots.")
         return # Skip parameter analysis if no baseline data

    for error_type in unique_error_types:
        error_df_no_qec = df_no_qec[df_no_qec['error_type'] == error_type]
        if error_df_no_qec.empty:
            continue

        # Identify varying parameters within this error type subset
        # Handle cases where error_params might be None or empty
        valid_params = [p for p in error_df_no_qec['error_params'] if p and isinstance(p, dict)]
        if not valid_params: continue
        param_keys = list(valid_params[0].keys())

        varying_params = defaultdict(list)
        param_agg = []

        for idx, row in error_df_no_qec.iterrows():
            if not row['error_params'] or not isinstance(row['error_params'], dict): continue # Skip if no params
            param_values = tuple(row['error_params'].get(k, None) for k in param_keys)
            param_agg.append(param_values + (row['avg_fidelity'],))
            for i, k in enumerate(param_keys):
                 # Ensure the value is appended, even if None
                 varying_params[k].append(param_values[i])


        param_df = pd.DataFrame(param_agg, columns=param_keys + ['avg_fidelity'])

        # Plot against each parameter if it varies
        for param_key in param_keys:
            # Filter out None values for this parameter before checking uniqueness/plotting
            valid_param_values = [v for v in varying_params[param_key] if v is not None]
            if not valid_param_values: continue # Skip if no valid values for this key

            unique_values = sorted(list(set(valid_param_values)))

            if len(unique_values) > 1: # Only plot if the parameter actually varies
                # Average over other parameters if they exist
                plot_data = param_df.groupby(param_key)['avg_fidelity'].agg(['mean', 'std']).reset_index()
                # Filter out rows where the parameter might be NaN after grouping if necessary
                plot_data = plot_data.dropna(subset=[param_key])
                # Sort by parameter value to ensure lines connect points in the right order
                plot_data = plot_data.sort_values(by=param_key)
                
                if plot_data.empty: continue

                plt.figure()
                # Create scatter plot for means with error bars
                plt.errorbar(plot_data[param_key], plot_data['mean'], yerr=plot_data['std'],
                             fmt='o-', # Add connecting lines with '-'
                             capsize=5, alpha=0.6, color='grey', label='Std Dev') # Error bars
                plt.scatter(plot_data[param_key], plot_data['mean'],
                             marker='o', s=50, alpha=0.85, zorder=3, label='Mean Fidelity') # Scatter points for mean

                # Use LaTeX for gamma if applicable
                param_label = r'$\gamma$' if 'gamma' in param_key else param_key.replace('_', ' ')
                plt.title(f"Fidelity (No QEC) vs. {param_label} for {error_type.replace('_',' ').title()}")
                plt.xlabel(f"Parameter: {param_label}")
                plt.ylabel(r'Average Fidelity $F$')
                plt.ylim(0, 1.05)
                plt.grid(True, linestyle=':', alpha=0.7)
                # Consider log scale for x-axis if parameters span orders of magnitude
                numeric_unique_values = [v for v in unique_values if isinstance(v, (int, float))]
                if numeric_unique_values and len(numeric_unique_values) > 1 and \
                   min(numeric_unique_values) > 0 and \
                   max(numeric_unique_values) / min(numeric_unique_values) > 100:
                       plt.xscale('log')
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(error_plots_dir, f"{error_type}_vs_{param_key}_no_qec_scatter.png"))
                plt.close()

def analyze_qec_methods(df, plots_subdir="qec_method_analysis"):
    """
    Analyzes and plots results grouped by QEC method.
    - Compares QEC methods against different error types (with std dev annotations).
    - Compares QEC methods for different initial states (with std dev annotations).
    """
    print("\n--- Analyzing by QEC Method ---")
    qec_plots_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(qec_plots_dir, exist_ok=True)

    qec_methods = sorted([m for m in df['qec_method'].unique() if m != 'none'])
    error_types = sorted(df['error_type'].unique())
    initial_states = sorted(df['initial_state'].unique())

    # Define readable names and consistent colors
    method_display = {
        'three_qubit_bit_flip': '3QB Bit', 'three_qubit_phase_flip': '3QB Phase',
        'shor_nine': 'Shor-9', 'none': 'No QEC'
    }
    method_colors = plt.cm.tab10(np.linspace(0, 1, len(qec_methods) + 1))
    method_color_map = {m: method_colors[i] for i, m in enumerate(qec_methods + ['none'])}


    # 1. QEC Performance vs. Error Type
    print("  Plotting QEC performance vs. error type...")
    qec_error_metrics = df.groupby(['qec_method', 'error_type']).agg(
        avg_fidelity=('avg_fidelity', 'mean'),
        std_fidelity=('avg_fidelity', 'std')
    ).unstack(level=0) # Pivot qec_method to columns

    if not qec_error_metrics.empty:
        fig, ax = plt.subplots(figsize=(16, 8)) # Wider figure
        num_methods = len(qec_error_metrics['avg_fidelity'].columns)
        bar_width = 0.8 / num_methods
        x = np.arange(len(qec_error_metrics.index))

        for i, method in enumerate(qec_error_metrics['avg_fidelity'].columns):
            means = qec_error_metrics[('avg_fidelity', method)]
            stds = qec_error_metrics[('std_fidelity', method)]
            positions = x - (num_methods / 2 - i - 0.5) * bar_width

            bars = ax.bar(positions, means, bar_width, yerr=stds,
                          label=method_display.get(method, method),
                          color=method_color_map.get(method, 'grey'),
                          alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)

            # Add std dev text annotations
            for bar, std_val in zip(bars, stds):
                height = bar.get_height()
                y_err_val = std_val if pd.notna(std_val) else 0
                if pd.notna(height):
                    # Position text slightly above the bar top, aligned below the specified point
                    y_pos = min(height + 0.01, 1.02)
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                            rf'$\pm{std_val:.2f}$' if pd.notna(std_val) else '', # Use raw f-string
                            ha='center', va='bottom', fontsize=7, color='dimgray', clip_on=True)

        ax.set_title(r'QEC Method Performance by Error Type')
        ax.set_ylabel(r'Average Fidelity $F$')
        ax.set_xlabel(r'Error Type')
        ax.set_xticks(x)
        ax.set_xticklabels(qec_error_metrics.index, rotation=45, ha='right')
        ax.legend(title="QEC Method", loc='best')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(qec_plots_dir, "qec_vs_error_type.png"))
        plt.close()

    # 2. QEC Performance vs. Initial State
    print("  Plotting QEC performance vs. initial state...")
    qec_state_metrics = df.groupby(['qec_method', 'initial_state']).agg(
        avg_fidelity=('avg_fidelity', 'mean'),
        std_fidelity=('avg_fidelity', 'std')
    ).unstack(level=0) # Pivot qec_method to columns

    if not qec_state_metrics.empty:
        fig, ax = plt.subplots(figsize=(13, 7))
        num_methods = len(qec_state_metrics['avg_fidelity'].columns)
        bar_width = 0.8 / num_methods
        x = np.arange(len(qec_state_metrics.index))
        # Use Mathtext-compatible ket notation (ensure consistency)
        state_display_map = {'0': r'$|0\rangle$', '1': r'$|1\rangle$', '+': r'$|+\rangle$', '-': r'$|-\rangle$'}

        for i, method in enumerate(qec_state_metrics['avg_fidelity'].columns):
            means = qec_state_metrics[('avg_fidelity', method)]
            stds = qec_state_metrics[('std_fidelity', method)]
            positions = x - (num_methods / 2 - i - 0.5) * bar_width

            bars = ax.bar(positions, means, bar_width, yerr=stds,
                          label=method_display.get(method, method),
                          color=method_color_map.get(method, 'grey'),
                          alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)

            # Add std dev text annotations
            for bar, std_val in zip(bars, stds):
                height = bar.get_height()
                y_err_val = std_val if pd.notna(std_val) else 0
                if pd.notna(height):
                    # Position text slightly above the bar top, aligned below the specified point
                    y_pos = min(height + 0.01, 1.02)
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                            rf'$\pm{std_val:.2f}$' if pd.notna(std_val) else '', # Use raw f-string
                            ha='center', va='bottom', fontsize=7, color='dimgray', clip_on=True)

        ax.set_title(r'QEC Method Performance by Initial State')
        ax.set_ylabel(r'Average Fidelity $F$')
        ax.set_xlabel(r'Initial State')
        ax.set_xticks(x)
        ax.set_xticklabels([state_display_map.get(s, s) for s in qec_state_metrics.index])
        ax.legend(title="QEC Method", loc='best')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(qec_plots_dir, "qec_vs_initial_state.png"))
        plt.close()


def analyze_distance(df, plots_subdir="distance_analysis"):
    """
    Analyzes and plots fidelity as a function of distance.
    """
    print("\n--- Analyzing by Distance ---")
    dist_plots_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(dist_plots_dir, exist_ok=True)

    # Group by distance, error_type, and qec_method
    dist_metrics = df.groupby(['distance', 'error_type', 'qec_method']).agg(
        avg_fidelity=('avg_fidelity', 'mean'),
        std_fidelity=('avg_fidelity', 'std') # Std dev of average fidelities across param variations
    ).reset_index()

    error_types = sorted(dist_metrics['error_type'].unique())
    qec_methods = sorted(dist_metrics['qec_method'].unique())
    method_display = {
        'three_qubit_bit_flip': '3QB Bit', 'three_qubit_phase_flip': '3QB Phase',
        'shor_nine': 'Shor-9', 'none': 'No QEC'
    }
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'X'] # Different markers for QEC methods
    colors = plt.cm.tab10 # Use a colormap

    # Create one plot per error type, showing different QEC methods
    for i, error_type in enumerate(error_types):
        plt.figure()
        ax = plt.gca()
        subset_error = dist_metrics[dist_metrics['error_type'] == error_type]

        method_colors_dist = plt.cm.viridis(np.linspace(0, 0.85, len(qec_methods)))
        for j, qec_method in enumerate(qec_methods):
            subset_qec = subset_error[subset_error['qec_method'] == qec_method].sort_values('distance')
            if not subset_qec.empty:
                ax.errorbar(subset_qec['distance'], subset_qec['avg_fidelity'], yerr=subset_qec['std_fidelity'],
                            label=method_display.get(qec_method, qec_method),
                            marker=markers[j % len(markers)], capsize=3, linestyle='-', linewidth=1.5, markersize=5,
                            color=method_colors_dist[j] )

        ax.set_title(f"Fidelity vs. Distance for {error_type.replace('_', ' ').title()} Error")
        ax.set_xlabel(r'Distance (km)')
        ax.set_ylabel(r'Average Fidelity $F$')
        ax.legend(title="QEC Method")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
        ax.grid(True, linestyle=':', alpha=0.6)
        # Use minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tight_layout()
        plt.savefig(os.path.join(dist_plots_dir, f"fidelity_vs_distance_{error_type}.png"))
        plt.close()

# --- Add these new plotting functions below the existing ones ---

def create_distribution_fidelity_plots(all_data, plots_subdir="fidelity_distributions"):
    """Create fidelity distribution visualizations (violin plots)."""
    print("\n--- Creating Fidelity Distribution Plots ---")
    if not all_data:
        print("  No data available for plotting.")
        return
    # Check if full data was loaded
    if "data" not in all_data[0]:
         print("  Full iteration data not loaded. Run load_network_qec_data with load_full_data=True.")
         print("  Skipping distribution plots.")
         return

    violin_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(violin_dir, exist_ok=True)

    states = sorted({exp['initial_state'] for exp in all_data})
    state_display_map = {'0': r'$|0\rangle$', '1': r'$|1\rangle$', '+': r'$|+\rangle$', '-': r'$|-\rangle$'} # Use compatible kets

    # 1. Combined violin plot of all states
    print("  Creating combined violin plot of all states...")
    fig, ax = plt.subplots(figsize=(14, 8)) # Adjusted size

    all_violins = []
    plot_labels = []
    positions = []
    mean_values = {}

    for i, state in enumerate(states):
        # Concatenate fidelity data from *all* relevant experiments for this state
        state_data = [exp['data']['fidelity'].values for exp in all_data
                     if exp['initial_state'] == state and "data" in exp] # Ensure 'data' exists

        if state_data:
            concatenated_data = np.concatenate(state_data)
            if len(concatenated_data) > 0: # Ensure we have data points
                 all_violins.append(concatenated_data)
                 plot_labels.append(state_display_map.get(state, state)) # Use display name
                 positions.append(i + 1)
                 mean_values[state] = np.mean(concatenated_data)

    if all_violins:
        parts = ax.violinplot(all_violins, positions=positions, showmeans=False, showextrema=True, widths=0.8)

        # Style violins
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(plot_labels)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        # Style lines
        for partname in ('cbars','cmins','cmaxes'):
             vp = parts[partname]
             vp.set_edgecolor('black')
             vp.set_linewidth(1)

        ax.set_title("Overall Fidelity Distribution Across Initial States", pad=20, fontsize=18)
        ax.set_ylabel("Fidelity", fontsize=14)
        ax.set_xlabel("Initial State", fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels)
        ax.set_ylim(-0.05, 1.05) # Adjusted ylim for visibility

        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which='major', axis='y', alpha=0.6, linestyle='-')
        ax.grid(True, which='minor', axis='y', alpha=0.3, linestyle=':')

        # Create legend with mean values
        legend_elements = [
            mpl.lines.Line2D([0], [0], marker='o', color='w', label=f'{plot_labels[i]}: Mean={mean_values[state]:.3f}',
                             markerfacecolor=colors[i], markersize=8, alpha=0.8)
            for i, state in enumerate(states) if state in mean_values
        ]
        ax.legend(handles=legend_elements, title="State Means", loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(violin_dir, "all_states_fidelity_violin.png"), dpi=300)
        plt.close()
    else:
        print("  No data found to create combined state violin plot.")


def create_distribution_state_error_plots(all_data, plots_subdir="state_error_distributions"):
    """Create improved violin plots for each state showing distributions per error type"""
    print("\n--- Creating State vs Error Type Distribution Plots ---")
    if not all_data or "data" not in all_data[0]:
         print("  Full iteration data not loaded or no data. Skipping distribution plots.")
         return

    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    states = sorted({exp['initial_state'] for exp in all_data})
    error_types = sorted({exp['error_type'] for exp in all_data})

    for state in states:
        # Determine grid size
        n_errors = len(error_types)
        n_cols = 3
        n_rows = (n_errors + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle(f"Fidelity Distribution by Error Type: State {state}", fontsize=18, y=1.0) # Adjusted y

        axs_flat = axs.flatten()
        plot_idx = 0
        for error_type in error_types:
            ax = axs_flat[plot_idx]

            # Aggregate all fidelity data for this state and error type across all params/QEC
            state_error_data = [exp['data']['fidelity'].values for exp in all_data
                                if exp['initial_state'] == state and exp['error_type'] == error_type and "data" in exp]

            if state_error_data:
                concatenated_data = np.concatenate(state_error_data)
                if len(concatenated_data) > 0:
                    parts = ax.violinplot(concatenated_data, showmeans=False, showextrema=True, widths=0.8)
                    # Style violin
                    for pc in parts['bodies']:
                        pc.set_facecolor(plt.cm.tab10(plot_idx % 10)) # Use consistent color cycle
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)
                    for partname in ('cbars','cmins','cmaxes'):
                        vp = parts[partname]
                        vp.set_edgecolor('black')
                        vp.set_linewidth(1)

                    # Add mean value text annotation in the top right of the subplot
                    mean_val = np.mean(concatenated_data)
                    ax.text(0.95, 0.95, f'Mean: {mean_val:.3f}',
                            transform=ax.transAxes, # Use axes coordinates
                            ha='right', va='top', fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='grey'))

                    ax.set_ylim(-0.05, 1.05)
                    ax.set_xticks([]) # No x-ticks needed for single violin
                    ax.set_ylabel("Fidelity")
                    ax.set_title(error_type.replace('_', ' ').title(), fontsize=14)
                    ax.yaxis.set_major_locator(MultipleLocator(0.2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.grid(True, which='both', axis='y', alpha=0.5, linestyle=':')
                    plot_idx += 1
                else:
                     ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                     ax.set_title(error_type.replace('_', ' ').title(), fontsize=14)
                     ax.set_xticks([])
                     ax.set_yticks([])
                     plot_idx += 1
            else:
                 ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                 ax.set_title(error_type.replace('_', ' ').title(), fontsize=14)
                 ax.set_xticks([])
                 ax.set_yticks([])
                 plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axs_flat)):
            axs_flat[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout rect for suptitle
        plt.savefig(os.path.join(plot_dir, f"{state}_error_type_distribution.png"), dpi=300)
        plt.close()


def create_distribution_qec_plots(all_data, plots_subdir="qec_method_distributions"):
    """Create violin/box plots comparing QEC method fidelity distributions."""
    print("\n--- Creating QEC Method Distribution Plots ---")
    if not all_data or "data" not in all_data[0]:
         print("  Full iteration data not loaded or no data. Skipping distribution plots.")
         return

    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    states = sorted({exp['initial_state'] for exp in all_data})
    qec_methods = sorted({exp['qec_method'] for exp in all_data}) # Include 'none'
    state_display_map = {'0': r'$|0\rangle$', '1': r'$|1\rangle$', '+': r'$|+\rangle$', '-': r'$|-\rangle$'} # For title

    method_display = { # Define readable names
        'three_qubit_bit_flip': '3QB Bit', 'three_qubit_phase_flip': '3QB Phase',
        'shor_nine': 'Shor-9', 'none': 'No QEC'
    }
    method_colors = plt.cm.viridis(np.linspace(0, 0.9, len(qec_methods)))
    method_color_map = {m: method_colors[i] for i, m in enumerate(qec_methods)}

    for state in states:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        plot_data = []
        plot_labels = []
        plot_positions = []
        mean_texts = []
        current_pos = 1

        for method in qec_methods: # Iterate through all methods including 'none'
            # Aggregate all fidelity data for this state and QEC method
            method_state_data = [exp['data']['fidelity'].values for exp in all_data
                                 if exp['initial_state'] == state and exp['qec_method'] == method and "data" in exp]

            if method_state_data:
                concatenated_data = np.concatenate(method_state_data)
                if len(concatenated_data) > 0:
                    plot_data.append(concatenated_data)
                    plot_labels.append(method_display.get(method, method))
                    plot_positions.append(current_pos)
                    mean_texts.append(f'{method_display.get(method, method)}: {np.mean(concatenated_data):.3f}')
                    current_pos += 1

        if plot_data:
            # Violin Plot
            vparts = ax.violinplot(plot_data, positions=plot_positions, showmeans=False, showextrema=False, widths=0.7)
            for i, pc in enumerate(vparts['bodies']):
                pc.set_facecolor(method_color_map.get(qec_methods[i], 'grey')) # Use original method name for color lookup
                pc.set_edgecolor('black')
                pc.set_alpha(0.4)

            # Box Plot overlay
            bparts = ax.boxplot(plot_data, positions=plot_positions, showfliers=False, patch_artist=True, widths=0.3,
                                medianprops=dict(color='black', linewidth=1.5))
            for i, box in enumerate(bparts['boxes']):
                box.set_facecolor(method_color_map.get(qec_methods[i], 'grey'))
                box.set_alpha(0.8)
                box.set_edgecolor('black')

            ax.set_title(f"Fidelity Distribution by QEC Method for State {state_display_map.get(state, state)}", pad=20)
            ax.set_ylabel("Fidelity")
            ax.set_xlabel("QEC Method")
            ax.set_xticks(plot_positions)
            ax.set_xticklabels(plot_labels, rotation=30, ha='right')
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which='both', axis='y', alpha=0.5, linestyle=':')

            # Add text box with all means in the top right
            mean_summary_text = 'Means:\n' + '\n'.join(mean_texts)
            ax.text(0.97, 0.97, mean_summary_text,
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{state}_qec_method_distribution.png"), dpi=300)
            plt.close()
        else:
            print(f"  No data found to create QEC distribution plot for state {state}.")



# --- Main Execution ---
if __name__ == "__main__":
    configure_plots()
    # Load data *with* full iteration details for distribution plots
    all_data = load_network_qec_data(load_full_data=True)

    if all_data:
        # Create a DataFrame for easier analysis of *averages*
        df_all_metrics = pd.DataFrame([{k: v for k, v in exp.items() if k != 'data'} for exp in all_data])
        print(f"Original number of configurations loaded: {len(df_all_metrics)}")

        # --- Filter out 'fibre_loss' error type --- #
        df_filtered_metrics = df_all_metrics[df_all_metrics['error_type'] != 'fibre_loss'].copy()
        all_data_filtered = [exp for exp in all_data if exp['error_type'] != 'fibre_loss']
        print(f"Number of configurations after filtering 'fibre_loss': {len(df_filtered_metrics)}")
        # ------------------------------------------ #

        # Run the analyses based on average metrics
        aggregate_and_plot_error_types(df_filtered_metrics)
        analyze_qec_methods(df_filtered_metrics)
        analyze_distance(df_filtered_metrics)

        # Run the analyses based on full distributions
        create_distribution_fidelity_plots(all_data_filtered)
        create_distribution_state_error_plots(all_data_filtered)
        create_distribution_qec_plots(all_data_filtered)


        print(f"\nAnalysis complete. Plots saved to: {PLOTS_DIR}")
    else:
        print("No data loaded, cannot perform analysis.")