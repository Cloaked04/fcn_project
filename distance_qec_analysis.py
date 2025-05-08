import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib as mpl
# Use non-interactive backend to avoid Qt conflicts
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import warnings
import math

# --- Configuration ---
# Make sure this points to the correct directory containing the distance experiment results
RESULTS_DIR = "distance_qec_fiber_loss"
PLOTS_DIR = "plots_distance_qec_fiber_loss"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Plotting Configuration (Copied from single_qubit_analysis.py) ---
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
        'legend.fontsize': 10, # Adjusted legend size
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'figure.figsize': (12, 7), # Slightly wider default
        'figure.dpi': 300,
        'text.usetex': False,  # Keep False for broader compatibility
        'mathtext.fontset': 'stix',
        'axes.prop_cycle': plt.cycler('color', plt.cm.viridis(np.linspace(0, 0.85, 8))),
        'axes.axisbelow': True,
        'grid.alpha': 0.6,
        'grid.linestyle': ':'
    })

configure_plots() # Apply the configuration
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore potential runtime warnings from mean of empty slice

# --- Helper Functions ---
def format_gamma(value, precision=4):
    """Formats gamma values, truncating long decimals."""
    if isinstance(value, (int, float)):
        if abs(value - round(value)) < 1e-9: # Check if it's effectively an integer
             return str(int(value))
        else:
             # Truncate instead of round
             factor = 10.0 ** precision
             return str(math.trunc(value * factor) / factor)
             # Alternative: standard formatting
             # return f"{value:.{precision}g}"
    return str(value)

def parse_param_string_distance(param_str):
    """Parses complex parameter strings from distance experiments.
       Handles p_loss_init, p_loss_length, sec_gamma, ter_gamma etc.
    """
    params = {}
    parts = param_str.split('_')
    i = 0
    current_key = ""
    while i < len(parts):
        part = parts[i]
        # Check if the part looks like a value (numeric, possibly with decimal or E notation)
        is_value = False
        try:
            float(part)
            is_value = True
        except ValueError:
            is_value = False

        if not is_value and current_key:
            # If we have a key and the current part isn't a value, it's part of the key
            current_key += "_" + part
            i += 1
        elif not is_value and not current_key:
            # Starting a new key
            current_key = part
            i += 1
        elif is_value and current_key:
            # Found the value for the current key
            try:
                 params[current_key] = float(part)
            except ValueError:
                 params[current_key] = part # Keep as string if not float
            current_key = "" # Reset key
            i += 1
        else:
            # Edge case: value without a preceding key? Or starting with a value?
            # Treat as part of the next key or ignore if error
            # print(f"Warning: Unexpected part '{part}' in param string: {param_str}")
            i += 1 # Move on

    # Rename keys for clarity if needed (optional)
    final_params = {}
    for k, v in params.items():
        if k.startswith("sec_"):
            final_params[k] = v # Keep prefix for now
        elif k.startswith("ter_"):
            final_params[k] = v # Keep prefix for now
        else:
            final_params[k] = v

    return final_params

def format_params_dict(params_dict):
    """Creates a readable string representation of the params dict, formatting gamma."""
    if not params_dict:
        return "No Params"
    items = []
    for k, v in sorted(params_dict.items()):
        if 'gamma' in k:
            items.append(f"{k}={format_gamma(v)}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)

# --- Data Loading Function --- 
def load_distance_qec_data(results_dir=RESULTS_DIR):
    """
    Loads experiment data from the distance_qec_network results structure.
    Extracts metrics from metadata files and calculates loss-adjusted fidelity.
    """
    print(f"Loading distance experiment data from: {results_dir}")
    all_experiments_data = []
    skipped_files = []
    loaded_count = 0
    file_parse_errors = []

    # Regex to capture: state, qec_method, distance, params_string
    # Example: +_none_d10.0_p_loss_init_0.05_p_loss_length_0.16_sec_gamma_0.01_ter_gamma_0.2.csv
    filename_pattern = re.compile(r"^([+\-01])_(.+?)_d([\d\.]+?)_(.+)\.csv$")

    # Iterate through error combination directories
    error_combo_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for error_combo in error_combo_dirs:
        dir_path = os.path.join(results_dir, error_combo)
        # print(f" Processing directory: {dir_path}") # Reduced verbosity

        for filename in os.listdir(dir_path):
            # Focus ONLY on metadata files for primary data extraction
            if not filename.endswith("_metadata.csv"):
                continue

            base_filename = filename.replace("_metadata.csv", ".csv")
            match = filename_pattern.match(base_filename)

            if match:
                initial_state = match.group(1)
                qec_method = match.group(2)
                distance = float(match.group(3))
                param_string = match.group(4)
                metadata_path = os.path.join(dir_path, filename)

                try:
                    # Parse the parameter string first
                    error_params = parse_param_string_distance(param_string)
                    if not error_params:
                        # print(f"  - Warning: Could not parse parameters from '{param_string}' in {base_filename}")
                        skipped_files.append(base_filename)
                        continue

                    # Load metadata
                    metadata_df = pd.read_csv(metadata_path)
                    if metadata_df.empty:
                        # print(f"  - Warning: Empty metadata file: {filename}")
                        skipped_files.append(base_filename)
                        continue
                    metadata = metadata_df.iloc[0].to_dict()

                    # Extract required fields from metadata
                    raw_avg_fidelity = metadata.get('avg_fidelity', 0.0)
                    raw_std_fidelity = metadata.get('std_fidelity', 0.0)
                    total_runs = metadata.get('iterations_completed', 0)
                    loss_count = metadata.get('loss_count', 0)

                    # Validate data - ensure loss_count doesn't exceed total_runs
                    if loss_count > total_runs and total_runs > 0:
                        print(f"  - Warning: Loss count ({loss_count}) exceeds total runs ({total_runs}) in {filename}")
                        # Cap loss_count to total_runs
                        loss_count = total_runs

                    # Calculate loss-adjusted metrics
                    loss_adjusted_avg_fidelity = 0.0
                    loss_ratio = 0.0
                    successful_runs = total_runs - loss_count

                    if total_runs > 0:
                        loss_ratio = min(loss_count / total_runs, 1.0)  # Ensure loss_ratio is between 0 and 1
                        # Adjust fidelity: assume lost runs have fidelity 0
                        # Also handle case where raw_avg_fidelity is NaN (if successful_runs is 0)
                        if successful_runs > 0 and pd.notna(raw_avg_fidelity):
                             loss_adjusted_avg_fidelity = (raw_avg_fidelity * successful_runs) / total_runs
                        else:
                             loss_adjusted_avg_fidelity = 0.0 # If no successful runs or raw fidelity is NaN

                    else:
                         # Handle case with 0 iterations completed? Set fidelities to 0.
                         raw_avg_fidelity = 0.0
                         raw_std_fidelity = 0.0
                         loss_adjusted_avg_fidelity = 0.0

                    experiment_entry = {
                         "initial_state": initial_state,
                         "qec_method": qec_method,
                         "distance": distance,
                         "error_combo": error_combo, # Store the directory name as the error combo identifier
                         "error_params": error_params,
                         "param_string": format_params_dict(error_params), # Store readable param string
                         "avg_fidelity_raw": raw_avg_fidelity,
                         "std_fidelity_raw": raw_std_fidelity, # Std dev of individual trial fidelities (from metadata)
                         "avg_fidelity_adj": loss_adjusted_avg_fidelity,
                         "loss_ratio": loss_ratio,
                         "loss_count": loss_count,
                         "total_runs": total_runs,
                         "filename_base": base_filename
                         # We are not loading the full data CSV by default here
                    }

                    all_experiments_data.append(experiment_entry)
                    loaded_count += 1

                except Exception as e:
                    print(f"  - Error processing metadata file {filename}: {e}")
                    import traceback
                    traceback.print_exc() # Print traceback for debugging loading errors
                    file_parse_errors.append((filename, str(e)))
                    skipped_files.append(base_filename)
            else:
                 # Metadata filename doesn't match expected base pattern
                 skipped_files.append(filename)

    print(f"Finished loading. Loaded {loaded_count} experiment configurations from metadata.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files/metadata (check format/content/errors).")
    if file_parse_errors:
        print(f"Encountered {len(file_parse_errors)} errors during file processing:")
        for fname, err in file_parse_errors[:10]:
             print(f"  - {fname}: {err}")
        if len(file_parse_errors) > 10:
             print("  ... (additional errors truncated)")
    return all_experiments_data

# ===========================================
# --- NEW Plotting Functions Implementation ---
# ===========================================

# --- Mappings for Plotting ---
ERROR_COMBO_MAP = {
    'fibre_loss': 'Fiber',
    'fibre_loss_plus_amplitude_damping': 'Fiber+Amp',
    'fibre_loss_plus_phase_damping': 'Fiber+Phase',
    'fibre_loss_plus_amplitude_damping_plus_phase_damping': 'Fiber+Amp+Phase'
}

METHOD_DISPLAY_MAP = {
    'none': 'No QEC',
    'three_qubit_phase_flip': '3QB Phase',
    'shor_nine': 'Shor-9' # Add other methods if present
}

# --- Plot 1: Fidelity (Raw & Adjusted) vs Error Combination ---
def plot_fidelity_by_error_combo(df, plots_subdir="1_fidelity_vs_error_combo"):
    """
    Plots avg fidelity (raw & adjusted) vs. error combination.
    Averages over distances, QEC methods, initial states, and specific parameters.
    Includes std dev of the *mean* fidelities across aggregated groups.
    """
    print("\n--- Plot 1: Fidelity vs Error Combination ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    # Aggregate fidelities, calculating mean and std dev of the means across runs
    agg_data = df.groupby('error_combo').agg(
        mean_raw=('avg_fidelity_raw', 'mean'),
        std_raw=('avg_fidelity_raw', 'std'), # Std dev of the mean fidelities from different configs
        mean_adj=('avg_fidelity_adj', 'mean'),
        std_adj=('avg_fidelity_adj', 'std')  # Std dev of the mean adjusted fidelities
    ).reset_index()

    # Sort for consistent plotting order (optional)
    agg_data = agg_data.sort_values(by='mean_adj', ascending=False)

    if agg_data.empty:
        print("  No data to plot for Fidelity vs Error Combination.")
        return

    n_combos = len(agg_data['error_combo'])
    index = np.arange(n_combos)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n_combos * 1.5), 7)) # Adjust width based on number of combos

    # Plot bars
    bars_raw = ax.bar(index - bar_width/2, agg_data['mean_raw'], bar_width,
                      yerr=agg_data['std_raw'], label=r'Raw Fidelity',
                      capsize=4, alpha=0.85, edgecolor='black', linewidth=0.5)
    bars_adj = ax.bar(index + bar_width/2, agg_data['mean_adj'], bar_width,
                      yerr=agg_data['std_adj'], label=r'Loss-Adjusted Fidelity',
                      capsize=4, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Add labels and title with LaTeX
    ax.set_xlabel(r'Error Combination')
    ax.set_ylabel(r'Average Fidelity $F$')
    ax.set_title(r'Overall Average Fidelity by Error Combination')
    ax.set_xticks(index)
    # Use the mapped short names for labels
    ax.set_xticklabels(agg_data['error_combo'], rotation=30, ha='right')
    ax.legend(loc='upper right') # Changed from 'best'
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Add value labels on top of bars (optional, can be cluttered)
    # ax.bar_label(bars_raw, fmt=r'%.2f', padding=3, fontsize=8)
    # ax.bar_label(bars_adj, fmt=r'%.2f', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "avg_fidelity_by_error_combo.png"))
    plt.close()

# --- Plot 2: Fidelity/Loss vs Fiber Parameters --- (REMOVED)
# def plot_fidelity_loss_vs_fiber_params(df, plots_subdir="2_vs_fiber_params"):
#     ...

# --- Plot 3: Fidelity (Raw & Adjusted) vs Distance (Combined) ---
def plot_fidelity_vs_distance_combined(df, plots_subdir="2_fidelity_vs_distance_combined"):
    """Creates a single plot showing fidelity vs distance for all error combinations."""
    print("\n--- Plotting Fidelity vs. Distance (Combined) ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Filter out too-few points
    distances = sorted(df['distance'].unique())
    if len(distances) <= 1:
        print("  Not enough distance points to create plot.")
        return
    
    # Create a single plot instead of 4 subplots
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Set up color scheme by error combination
    error_combos = sorted(df['error_combo'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(error_combos)))
    error_colors = {combo: colors[i] for i, combo in enumerate(error_combos)}
    
    # Set up marker scheme by QEC method
    qec_methods = sorted(df['qec_method'].unique())
    markers = ['o', 's', '^', 'd', '*']
    marker_map = {qec: markers[i % len(markers)] for i, qec in enumerate(qec_methods)}
    
    # Track lines for legend
    error_lines = {}
    method_markers = {}
    
    # Plot each error combination and QEC method
    for error_combo in error_combos:
        for qec_method in qec_methods:
            subset = df[(df['error_combo'] == error_combo) & 
                        (df['qec_method'] == qec_method)]
            
            if subset.empty:
                continue
                
            # Average across other parameters for each distance
            avg_by_dist = subset.groupby('distance').agg(
                avg_fid_raw=('avg_fidelity_raw', 'mean'),
                std_fid_raw=('avg_fidelity_raw', 'std'),
                avg_fid_adj=('avg_fidelity_adj', 'mean'),
                std_fid_adj=('avg_fidelity_adj', 'std'),
                avg_loss=('loss_ratio', 'mean')  # Track loss ratio for reference
            ).reset_index()
            
            # Include single data points as well - don't require at least 2 points
            if avg_by_dist.empty:
                continue
                
            # Plot adjusted fidelity FIRST and with greater prominence (solid line)
            line_adj, = ax.plot(avg_by_dist['distance'], avg_by_dist['avg_fid_adj'],
                             marker=marker_map[qec_method], linestyle='-', linewidth=2.5, 
                             color=error_colors[error_combo], markersize=8,
                             label=f"{ERROR_COMBO_MAP.get(error_combo, error_combo)} - {METHOD_DISPLAY_MAP.get(qec_method, qec_method)} (Adj.)")
            
            # Plot raw fidelity with less prominence (dotted line)
            line_raw, = ax.plot(avg_by_dist['distance'], avg_by_dist['avg_fid_raw'],
                             marker=marker_map[qec_method], linestyle=':', linewidth=1.5,
                             color=error_colors[error_combo], markersize=6,
                             label=f"{ERROR_COMBO_MAP.get(error_combo, error_combo)} - {METHOD_DISPLAY_MAP.get(qec_method, qec_method)} (Raw)")
            
            # Add error bands
            ax.fill_between(avg_by_dist['distance'], 
                          avg_by_dist['avg_fid_adj'] - avg_by_dist['std_fid_adj'],
                          avg_by_dist['avg_fid_adj'] + avg_by_dist['std_fid_adj'], 
                          color=error_colors[error_combo], alpha=0.1)
            
            # Track for legend
            error_combo_name = ERROR_COMBO_MAP.get(error_combo, error_combo)
            if error_combo_name not in error_lines:
                error_lines[error_combo_name] = line_adj  # Use adjusted line for legend
            
            method_name = METHOD_DISPLAY_MAP.get(qec_method, qec_method)
            if method_name not in method_markers:
                method_markers[method_name] = marker_map[qec_method]
    
    # Set up plot appearance
    ax.set_title("Fidelity vs. Distance by Error Combination", fontsize=16)
    ax.set_xlabel("Distance (km)", fontsize=14)
    ax.set_ylabel("Average Fidelity", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Create a better legend
    from matplotlib.lines import Line2D
    
    # Create custom legend elements
    legend_elements = []
    
    # Add error types
    for name, line in error_lines.items():
        legend_elements.append(Line2D([0], [0], color=line.get_color(), lw=2, label=name))
    
    # Add line styles
    legend_elements.append(Line2D([0], [0], color='gray', lw=2.5, linestyle='-', label='Loss-Adjusted Fidelity'))
    legend_elements.append(Line2D([0], [0], color='gray', lw=1.5, linestyle=':', label='Raw Fidelity'))
    
    # Add QEC methods
    for name, marker in method_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='gray', linestyle='None',
                                     markersize=8, label=name))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9,
             bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fidelity_vs_distance_all.png"), dpi=300)
    
    # Create a second plot with individual panels for each QEC method
    qec_methods_to_plot = list(method_markers.keys())
    n_methods = len(qec_methods_to_plot)
    if n_methods > 0:
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
        axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
        
        for i, method_display in enumerate(qec_methods_to_plot):
            ax = axs[i]
            # Reverse lookup the actual QEC method from display name
            qec_method = next((m for m, d in METHOD_DISPLAY_MAP.items() if d == method_display), method_display)
            
            # Plot each error combination for this method
            for error_combo in error_combos:
                subset = df[(df['error_combo'] == error_combo) & 
                            (df['qec_method'] == qec_method)]
                
                if subset.empty:
                    continue
                    
                avg_by_dist = subset.groupby('distance').agg(
                    avg_fid_adj=('avg_fidelity_adj', 'mean'),
                    std_fid_adj=('avg_fidelity_adj', 'std')
                ).reset_index()
                
                if not avg_by_dist.empty:
                    # Sort by distance for proper line connection
                    avg_by_dist = avg_by_dist.sort_values('distance')
                    
                    # Plot adjusted fidelity only
                    line, = ax.plot(avg_by_dist['distance'], avg_by_dist['avg_fid_adj'],
                                   marker='o', linestyle='-', linewidth=2, 
                                   color=error_colors[error_combo], markersize=6,
                                   label=f"{ERROR_COMBO_MAP.get(error_combo, error_combo)}")
                    
                    # Add error bands
                    ax.fill_between(avg_by_dist['distance'], 
                                  avg_by_dist['avg_fid_adj'] - avg_by_dist['std_fid_adj'],
                                  avg_by_dist['avg_fid_adj'] + avg_by_dist['std_fid_adj'], 
                                  color=error_colors[error_combo], alpha=0.1)
            
            ax.set_title(method_display, fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            if i % n_cols == 0:  # First column
                ax.set_ylabel("Loss-Adjusted Fidelity", fontsize=12)
            if i >= n_methods - n_cols:  # Last row
                ax.set_xlabel("Distance (km)", fontsize=12)
            
            # Add legend to first plot only
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)
        
        # Hide unused axes
        for j in range(n_methods, len(axs)):
            axs[j].set_visible(False)
            
        fig.suptitle("Loss-Adjusted Fidelity vs. Distance by QEC Method", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fidelity_vs_distance_by_qec_method.png"), dpi=300)
    
    plt.close('all')

# --- Plot 4: QEC Performance (Raw & Adjusted) vs Error Combination (Faceted by Error Combo) ---
def plot_qec_performance_faceted_by_error(df, plots_subdir="3_qec_performance_faceted_by_error"):
    """
    Plots raw and adjusted fidelity vs. QEC method, faceted by error combination.
    Includes std dev error bars representing variation across parameters/distances.
    """
    print("\n--- Plot 3: QEC Performance (Faceted by Error Combo) ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    qec_methods = sorted(df['qec_method'].unique())
    error_combos = sorted(df['error_combo'].unique())
    # Use METHOD_DISPLAY_MAP defined globally

    # Determine grid size
    n_combos = len(error_combos)
    n_cols = 2
    n_rows = (n_combos + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=True, sharey=True, squeeze=False)
    axs_flat = axs.flatten()

    for i, error_combo in enumerate(error_combos):
        ax = axs_flat[i]
        combo_subset = df[df['error_combo'] == error_combo]

        # Aggregate fidelities for this error combo
        agg_data = combo_subset.groupby('qec_method').agg(
            mean_raw=('avg_fidelity_raw', 'mean'),
            std_raw=('avg_fidelity_raw', 'std'),
            mean_adj=('avg_fidelity_adj', 'mean'),
            std_adj=('avg_fidelity_adj', 'std')
        ).reindex(qec_methods) # Ensure all methods are present and ordered

        if agg_data.empty:
             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
             ax.set_title(error_combo)
             continue

        n_methods_local = len(agg_data.index)
        index = np.arange(n_methods_local)
        bar_width = 0.35

        # Plot bars
        bars_raw = ax.bar(index - bar_width/2, agg_data['mean_raw'], bar_width,
                          yerr=agg_data['std_raw'], label=r'Raw Fidelity',
                          capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)
        bars_adj = ax.bar(index + bar_width/2, agg_data['mean_adj'], bar_width,
                          yerr=agg_data['std_adj'], label=r'Loss-Adjusted Fidelity',
                          capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_title(error_combo) # Use mapped short name
        ax.set_xticks(index)
        # Use METHOD_DISPLAY_MAP for labels
        ax.set_xticklabels([METHOD_DISPLAY_MAP.get(m, m) for m in agg_data.index], rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        if i % n_cols == 0: # Only first column
             ax.set_ylabel(r'Average Fidelity $F$')
        if i == 0: # Add legend to the first plot
            ax.legend(loc='best', fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axs_flat[j].set_visible(False)

    fig.suptitle(r'QEC Method Performance Across Error Combinations', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust rect
    plt.savefig(os.path.join(plot_dir, "qec_performance_faceted_by_error.png"))
    plt.close()

# --- Plot 5: Loss Ratio vs Distance (Combined) ---
def plot_loss_vs_distance_combined(df, plots_subdir="4_loss_vs_distance_combined"):
    """Creates a single plot showing loss ratio vs distance for all error combinations."""
    print("\n--- Plotting Loss Ratio vs. Distance (Combined) ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Filter out too-few points
    distances = sorted(df['distance'].unique())
    if len(distances) <= 1:
        print("  Not enough distance points to create plot.")
        return
    
    # Create a single plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Set up color scheme by error combination
    error_combos = sorted(df['error_combo'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(error_combos)))
    error_colors = {combo: colors[i] for i, combo in enumerate(error_combos)}
    
    # Set up marker scheme by QEC method
    qec_methods = sorted(df['qec_method'].unique())
    markers = ['o', 's', '^', 'd', '*']
    marker_map = {qec: markers[i % len(markers)] for i, qec in enumerate(qec_methods)}
    
    # Track lines for legend
    error_lines = {}
    method_markers = {}
    
    # Plot each error combination and QEC method
    for error_combo in error_combos:
        for qec_method in qec_methods:
            subset = df[(df['error_combo'] == error_combo) & 
                        (df['qec_method'] == qec_method)]
            
            if subset.empty:
                continue
                
            # Average across other parameters for each distance
            avg_by_dist = subset.groupby('distance').agg(
                avg_loss=('loss_ratio', 'mean'),
                std_loss=('loss_ratio', 'std')
            ).reset_index()
            
            if avg_by_dist.empty or len(avg_by_dist) < 2:
                continue
                
            # Plot loss ratio
            line, = ax.plot(avg_by_dist['distance'], avg_by_dist['avg_loss'],
                          marker=marker_map[qec_method], linestyle='-', linewidth=2,
                          color=error_colors[error_combo], markersize=8,
                          label=f"{ERROR_COMBO_MAP.get(error_combo, error_combo)} - {METHOD_DISPLAY_MAP.get(qec_method, qec_method)}")
            
            # Add error bars
            ax.fill_between(avg_by_dist['distance'], 
                          avg_by_dist['avg_loss'] - avg_by_dist['std_loss'],
                          avg_by_dist['avg_loss'] + avg_by_dist['std_loss'], 
                          color=error_colors[error_combo], alpha=0.1)
            
            # Track for legend
            error_combo_name = ERROR_COMBO_MAP.get(error_combo, error_combo)
            if error_combo_name not in error_lines:
                error_lines[error_combo_name] = line
            
            method_name = METHOD_DISPLAY_MAP.get(qec_method, qec_method)
            if method_name not in method_markers:
                method_markers[method_name] = marker_map[qec_method]
    
    # Set up plot appearance
    ax.set_title("Loss Ratio vs. Distance by Error Combination", fontsize=16)
    ax.set_xlabel("Distance (km)", fontsize=14)
    ax.set_ylabel("Average Loss Ratio", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Create a better legend
    from matplotlib.lines import Line2D
    
    # Create custom legend elements
    legend_elements = []
    
    # Add error types
    for name, line in error_lines.items():
        legend_elements.append(Line2D([0], [0], color=line.get_color(), lw=2, label=name))
    
    # Add QEC methods
    for name, marker in method_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='gray', linestyle='None',
                                     markersize=8, label=name))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9,
             bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_vs_distance_all.png"), dpi=300)
    plt.close()

# --- Add new function for fiber parameter analysis ---
def plot_fiber_params_analysis(df, plots_subdir="5_fiber_params_analysis"):
    """
    Creates plots analyzing the effect of fiber loss parameters:
    1. Fidelity vs p_loss_init
    2. Fidelity vs p_loss_length
    """
    print("\n--- Plotting Fidelity vs. Fiber Loss Parameters ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Only include fiber loss experiments
    fiber_df = df[df['error_combo'] == 'fibre_loss'].copy()
    if fiber_df.empty:
        print("  No fiber loss data found for parameter analysis.")
        return
        
    # Extract parameters
    fiber_df['p_loss_init'] = fiber_df['error_params'].apply(
        lambda p: p.get('p_loss_init', None) if isinstance(p, dict) else None)
    fiber_df['p_loss_length'] = fiber_df['error_params'].apply(
        lambda p: p.get('p_loss_length', None) if isinstance(p, dict) else None)
    
    # Drop rows with missing parameters
    fiber_df = fiber_df.dropna(subset=['p_loss_init', 'p_loss_length'])
    
    if fiber_df.empty:
        print("  Unable to extract fiber parameters from data.")
        return
    
    # --- 1. Plot Fidelity vs p_loss_init ---
    print("  Plotting fidelity vs p_loss_init...")
    
    # Get unique values for faceting
    p_lens = sorted(fiber_df['p_loss_length'].unique())
    distances = sorted(fiber_df['distance'].unique())
    
    if len(p_lens) > 1 and len(distances) > 0:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # Set up color and marker schemes
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(distances)))
        dist_colors = {d: colors[i] for i, d in enumerate(distances)}
        
        qec_methods = sorted(fiber_df['qec_method'].unique())
        markers = ['o', 's', '^', 'd', '*']
        qec_markers = {qec: markers[i % len(markers)] for i, qec in enumerate(qec_methods)}
        
        linestyles = ['-', '--', '-.', ':']
        p_len_lines = {p_len: linestyles[i % len(linestyles)] for i, p_len in enumerate(p_lens)}
        
        # Track for legend
        dist_handles = {}
        method_handles = {}
        p_len_handles = {}
        
        # For each p_loss_length value
        for p_len in p_lens:
            # For each distance
            for dist in distances:
                # For each QEC method
                for qec_method in qec_methods:
                    subset = fiber_df[(fiber_df['p_loss_length'] == p_len) & 
                                      (fiber_df['distance'] == dist) & 
                                      (fiber_df['qec_method'] == qec_method)]
                    
                    if len(subset) < 2:
                        continue
                    
                    # Group by p_loss_init
                    grouped = subset.groupby('p_loss_init').agg(
                        avg_fid_raw=('avg_fidelity_raw', 'mean'),
                        std_fid_raw=('avg_fidelity_raw', 'std'),
                        avg_fid_adj=('avg_fidelity_adj', 'mean'),
                        std_fid_adj=('avg_fidelity_adj', 'std')
                    ).reset_index()
                    
                    # Sort by p_loss_init
                    grouped = grouped.sort_values('p_loss_init')
                    
                    if len(grouped) < 2:
                        continue
                    
                    # Plot raw fidelity
                    line_raw, = ax.plot(grouped['p_loss_init'], grouped['avg_fid_raw'],
                                      marker=qec_markers[qec_method], 
                                      linestyle=p_len_lines[p_len],
                                      color=dist_colors[dist], 
                                      markersize=8, linewidth=2,
                                      label=f"Dist={dist}, p_len={p_len}, {METHOD_DISPLAY_MAP.get(qec_method, qec_method)} (Raw)")
                    
                    # Plot adjusted fidelity (slightly transparent)
                    line_adj, = ax.plot(grouped['p_loss_init'], grouped['avg_fid_adj'],
                                      marker=qec_markers[qec_method], 
                                      linestyle=p_len_lines[p_len],
                                      color=dist_colors[dist], 
                                      markersize=8, linewidth=2, alpha=0.5,
                                      label=f"Dist={dist}, p_len={p_len}, {METHOD_DISPLAY_MAP.get(qec_method, qec_method)} (Adj.)")
                    
                    # Track for legend
                    dist_handles[dist] = dist_colors[dist]
                    method_handles[METHOD_DISPLAY_MAP.get(qec_method, qec_method)] = qec_markers[qec_method]
                    p_len_handles[p_len] = p_len_lines[p_len]
        
        # Set up plot appearance
        ax.set_title(r"Fidelity vs. Initial Loss Probability ($p_{loss\_init}$)", fontsize=16)
        ax.set_xlabel(r"Initial Loss Probability ($p_{loss\_init}$)", fontsize=14)
        ax.set_ylabel("Average Fidelity", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Create multi-part legend
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Distance legend elements
        for dist, color in dist_handles.items():
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"Distance: {dist} km"))
        
        # Line style legend elements (p_loss_length)
        for p_len, ls in p_len_handles.items():
            legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls, lw=2,
                                        label=f"Loss/km: {p_len} dB/km"))
        
        # QEC method legend elements
        for method, marker in method_handles.items():
            legend_elements.append(Line2D([0], [0], color='gray', marker=marker, linestyle='None',
                                        markersize=8, label=f"QEC: {method}"))
        
        # Fidelity type legend elements
        legend_elements.append(Line2D([0], [0], color='gray', lw=2, alpha=1.0,
                                    label='Raw Fidelity'))
        legend_elements.append(Line2D([0], [0], color='gray', lw=2, alpha=0.5,
                                    label='Adjusted Fidelity'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                 framealpha=0.9, ncol=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fidelity_vs_p_loss_init.png"), dpi=300)
        plt.close()
    
    # --- 2. Plot Fidelity vs p_loss_length ---
    print("  Plotting fidelity vs p_loss_length...")
    
    # Get unique values for faceting
    p_inits = sorted(fiber_df['p_loss_init'].unique())
    
    if len(p_inits) > 1 and len(distances) > 0:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # Set up color and marker schemes
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(distances)))
        dist_colors = {d: colors[i] for i, d in enumerate(distances)}
        
        qec_methods = sorted(fiber_df['qec_method'].unique())
        markers = ['o', 's', '^', 'd', '*']
        qec_markers = {qec: markers[i % len(markers)] for i, qec in enumerate(qec_methods)}
        
        linestyles = ['-', '--', '-.', ':']
        p_init_lines = {p_init: linestyles[i % len(linestyles)] for i, p_init in enumerate(p_inits)}
        
        # Track for legend
        dist_handles = {}
        method_handles = {}
        p_init_handles = {}
        
        # For each p_loss_init value
        for p_init in p_inits:
            # For each distance
            for dist in distances:
                # For each QEC method
                for qec_method in qec_methods:
                    subset = fiber_df[(fiber_df['p_loss_init'] == p_init) & 
                                      (fiber_df['distance'] == dist) & 
                                      (fiber_df['qec_method'] == qec_method)]
                    
                    if len(subset) < 2:
                        continue
                    
                    # Group by p_loss_length
                    grouped = subset.groupby('p_loss_length').agg(
                        avg_fid_raw=('avg_fidelity_raw', 'mean'),
                        std_fid_raw=('avg_fidelity_raw', 'std'),
                        avg_fid_adj=('avg_fidelity_adj', 'mean'),
                        std_fid_adj=('avg_fidelity_adj', 'std')
                    ).reset_index()
                    
                    # Sort by p_loss_length
                    grouped = grouped.sort_values('p_loss_length')
                    
                    if len(grouped) < 2:
                        continue
                    
                    # Plot raw fidelity
                    line_raw, = ax.plot(grouped['p_loss_length'], grouped['avg_fid_raw'],
                                      marker=qec_markers[qec_method], 
                                      linestyle=p_init_lines[p_init],
                                      color=dist_colors[dist], 
                                      markersize=8, linewidth=2,
                                      label=f"Dist={dist}, p_init={p_init}, {METHOD_DISPLAY_MAP.get(qec_method, qec_method)} (Raw)")
                    
                    # Plot adjusted fidelity (slightly transparent)
                    line_adj, = ax.plot(grouped['p_loss_length'], grouped['avg_fid_adj'],
                                      marker=qec_markers[qec_method], 
                                      linestyle=p_init_lines[p_init],
                                      color=dist_colors[dist], 
                                      markersize=8, linewidth=2, alpha=0.5,
                                      label=f"Dist={dist}, p_init={p_init}, {METHOD_DISPLAY_MAP.get(qec_method, qec_method)} (Adj.)")
                    
                    # Track for legend
                    dist_handles[dist] = dist_colors[dist]
                    method_handles[METHOD_DISPLAY_MAP.get(qec_method, qec_method)] = qec_markers[qec_method]
                    p_init_handles[p_init] = p_init_lines[p_init]
        
        # Set up plot appearance
        ax.set_title(r"Fidelity vs. Loss per Length ($p_{loss\_length}$)", fontsize=16)
        ax.set_xlabel(r"Loss per Length (dB/km)", fontsize=14)
        ax.set_ylabel("Average Fidelity", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Create multi-part legend
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Distance legend elements
        for dist, color in dist_handles.items():
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"Distance: {dist} km"))
        
        # Line style legend elements (p_loss_init)
        for p_init, ls in p_init_handles.items():
            legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls, lw=2,
                                        label=f"Init Loss: {p_init}"))
        
        # QEC method legend elements
        for method, marker in method_handles.items():
            legend_elements.append(Line2D([0], [0], color='gray', marker=marker, linestyle='None',
                                        markersize=8, label=f"QEC: {method}"))
        
        # Fidelity type legend elements
        legend_elements.append(Line2D([0], [0], color='gray', lw=2, alpha=1.0,
                                    label='Raw Fidelity'))
        legend_elements.append(Line2D([0], [0], color='gray', lw=2, alpha=0.5,
                                    label='Adjusted Fidelity'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                 framealpha=0.9, ncol=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fidelity_vs_p_loss_length.png"), dpi=300)
        plt.close()

# --- New Function: QEC Loss Comparison ---
def plot_qec_loss_comparison(df, plots_subdir="6_qec_loss_comparison"):
    """Creates plots specifically focused on comparing loss rates between QEC methods."""
    print("\n--- Plotting QEC Loss Comparison ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Overall loss ratio by QEC method (all error combinations)
    plt.figure(figsize=(10, 6))
    qec_methods = sorted(df['qec_method'].unique())
    
    # Aggregate loss ratio by QEC method
    agg_loss = df.groupby('qec_method').agg(
        mean_loss=('loss_ratio', 'mean'),
        std_loss=('loss_ratio', 'std'),
        count=('loss_ratio', 'count')
    ).reindex(qec_methods).reset_index()
    
    if not agg_loss.empty:
        # Sort by mean loss (descending)
        agg_loss = agg_loss.sort_values('mean_loss', ascending=False)
        
        # Plot bars
        bars = plt.bar(
            [METHOD_DISPLAY_MAP.get(m, m) for m in agg_loss['qec_method']], 
            agg_loss['mean_loss'],
            yerr=agg_loss['std_loss'],
            capsize=5,
            color=plt.cm.viridis(np.linspace(0, 0.85, len(agg_loss)))
        )
        
        # Add count labels
        for bar, count in zip(bars, agg_loss['count']):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                0.02,
                f"n={count}",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90,
                color='gray'
            )
        
        plt.xlabel("QEC Method")
        plt.ylabel("Average Loss Ratio")
        plt.title("Overall Loss Ratio by QEC Method")
        plt.ylim(0, 1)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "overall_loss_ratio_by_qec.png"), dpi=300)
    
    # 2. Loss ratio vs Distance for each QEC method
    plt.figure(figsize=(10, 6))
    
    # Set up color scheme for QEC methods
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(qec_methods)))
    qec_colors = {qec: colors[i] for i, qec in enumerate(qec_methods)}
    
    # Set up line style for error combinations
    error_combos = sorted(df['error_combo'].unique())
    linestyles = ['-', '--', '-.', ':']
    error_lines = {combo: linestyles[i % len(linestyles)] for i, combo in enumerate(error_combos)}
    
    # Track lines for legend
    qec_method_lines = {}
    error_combo_lines = {}
    
    # Plot for each QEC method and error combination
    for qec_method in qec_methods:
        for error_combo in error_combos:
            subset = df[(df['qec_method'] == qec_method) & 
                        (df['error_combo'] == error_combo)]
            
            if subset.empty:
                continue
                
            # Group by distance
            agg_dist = subset.groupby('distance').agg(
                mean_loss=('loss_ratio', 'mean'),
                std_loss=('loss_ratio', 'std')
            ).reset_index()
            
            if agg_dist.empty:
                continue
                
            # Sort by distance for proper line connection
            agg_dist = agg_dist.sort_values('distance')
            
            # Plot line
            line, = plt.plot(
                agg_dist['distance'], 
                agg_dist['mean_loss'],
                marker='o',
                linestyle=error_lines[error_combo],
                color=qec_colors[qec_method],
                linewidth=2,
                label=f"{METHOD_DISPLAY_MAP.get(qec_method, qec_method)} - {ERROR_COMBO_MAP.get(error_combo, error_combo)}"
            )
            
            # Add error band
            plt.fill_between(
                agg_dist['distance'],
                agg_dist['mean_loss'] - agg_dist['std_loss'],
                agg_dist['mean_loss'] + agg_dist['std_loss'],
                color=qec_colors[qec_method],
                alpha=0.1
            )
            
            # Track for legend
            qec_method_display = METHOD_DISPLAY_MAP.get(qec_method, qec_method)
            if qec_method_display not in qec_method_lines:
                qec_method_lines[qec_method_display] = qec_colors[qec_method]
            
            error_combo_display = ERROR_COMBO_MAP.get(error_combo, error_combo)
            if error_combo_display not in error_combo_lines:
                error_combo_lines[error_combo_display] = error_lines[error_combo]
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # QEC methods
    for name, color in qec_method_lines.items():
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=name))
    
    # Error combinations
    for name, ls in error_combo_lines.items():
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls, lw=2, label=name))
    
    plt.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.9)
    plt.xlabel("Distance (km)")
    plt.ylabel("Loss Ratio")
    plt.title("Loss Ratio vs Distance by QEC Method and Error Combination")
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_ratio_vs_distance.png"), dpi=300)
    
    # 3. Loss rate comparison: Faceted by distance
    distances = sorted(df['distance'].unique())
    if len(distances) > 1:
        n_cols = min(3, len(distances))
        n_rows = (len(distances) + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
        axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
        
        for i, distance in enumerate(distances):
            if i >= len(axs):
                break
                
            ax = axs[i]
            
            # Get data for this distance
            dist_df = df[df['distance'] == distance]
            
            # Skip if no data
            if dist_df.empty:
                ax.text(0.5, 0.5, f"No data for d={distance}km", 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                continue
            
            # Group by QEC method and error combination
            agg_data = dist_df.groupby(['qec_method', 'error_combo']).agg(
                mean_loss=('loss_ratio', 'mean'),
                std_loss=('loss_ratio', 'std'),
                count=('loss_ratio', 'count')
            ).reset_index()
            
            # Skip if no aggregated data
            if agg_data.empty:
                continue
                
            # Get unique combinations for this distance
            combos = agg_data[['qec_method', 'error_combo']].drop_duplicates()
            combo_labels = [f"{METHOD_DISPLAY_MAP.get(row['qec_method'], row['qec_method'])}-{ERROR_COMBO_MAP.get(row['error_combo'], row['error_combo'])}" 
                          for _, row in combos.iterrows()]
            
            # Sort by mean loss
            loss_values = [agg_data[(agg_data['qec_method'] == qec) & 
                                  (agg_data['error_combo'] == err)]['mean_loss'].values[0]
                         for qec, err in zip(combos['qec_method'], combos['error_combo'])]
            
            std_values = [agg_data[(agg_data['qec_method'] == qec) & 
                                 (agg_data['error_combo'] == err)]['std_loss'].values[0]
                        for qec, err in zip(combos['qec_method'], combos['error_combo'])]
            
            count_values = [agg_data[(agg_data['qec_method'] == qec) & 
                                   (agg_data['error_combo'] == err)]['count'].values[0]
                          for qec, err in zip(combos['qec_method'], combos['error_combo'])]
            
            # Sort all by loss value (descending)
            sorted_indices = np.argsort(loss_values)[::-1]
            combo_labels = [combo_labels[idx] for idx in sorted_indices]
            loss_values = [loss_values[idx] for idx in sorted_indices]
            std_values = [std_values[idx] for idx in sorted_indices]
            count_values = [count_values[idx] for idx in sorted_indices]
            
            # Plot bars
            bars = ax.bar(
                combo_labels,
                loss_values,
                yerr=std_values,
                capsize=4,
                color=plt.cm.viridis(np.linspace(0, 0.85, len(combo_labels)))
            )
            
            # Add count labels
            for bar, count in zip(bars, count_values):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    0.02,
                    f"n={count}",
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    rotation=90,
                    color='gray'
                )
            
            ax.set_title(f"Distance = {distance} km", fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=8)
            
            if i % n_cols == 0:  # First column
                ax.set_ylabel("Loss Ratio", fontsize=12)
        
        # Hide unused axes
        for j in range(len(distances), len(axs)):
            axs[j].set_visible(False)
        
        fig.suptitle("Loss Ratio by QEC Method and Error Type at Different Distances", 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "loss_ratio_by_distance_facets.png"), dpi=300)
    
    plt.close('all')

# Add a new plot function specifically for comparing QEC methods
def plot_qec_method_comparison(df, plots_subdir="7_qec_method_comparison"):
    """Create plots directly comparing QEC methods side by side."""
    print("\n--- Plotting QEC Method Comparison ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Overall comparison across all error types
    plt.figure(figsize=(12, 8))
    
    # Aggregate data by QEC method
    qec_agg = df.groupby('qec_method').agg(
        raw_fid_mean=('avg_fidelity_raw', 'mean'),
        raw_fid_std=('avg_fidelity_raw', 'std'),
        adj_fid_mean=('avg_fidelity_adj', 'mean'),
        adj_fid_std=('avg_fidelity_adj', 'std'),
        loss_ratio_mean=('loss_ratio', 'mean'),
        loss_ratio_std=('loss_ratio', 'std'),
        count=('loss_ratio', 'count')
    ).reset_index()
    
    if not qec_agg.empty:
        # Sort by loss-adjusted fidelity
        qec_agg = qec_agg.sort_values('adj_fid_mean', ascending=False)
        
        # Set up x positions
        qec_labels = [METHOD_DISPLAY_MAP.get(m, m) for m in qec_agg['qec_method']]
        x = np.arange(len(qec_labels))
        width = 0.3
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PLOT 1: Fidelity comparison
        # Plot raw fidelity
        bars1 = ax1.bar(x - width/2, qec_agg['raw_fid_mean'], width, 
                      yerr=qec_agg['raw_fid_std'], 
                      label='Raw Fidelity', alpha=0.8, color='blue')
        
        # Plot adjusted fidelity
        bars2 = ax1.bar(x + width/2, qec_agg['adj_fid_mean'], width, 
                      yerr=qec_agg['adj_fid_std'], 
                      label='Loss-Adjusted Fidelity', alpha=0.8, color='green')
        
        # Add count labels
        for bar, count in zip(bars1, qec_agg['count']):
            ax1.text(bar.get_x() + bar.get_width()/2, 0.05, 
                    f"n={count}", ha='center', va='bottom', 
                    fontsize=9, rotation=0)
        
        # Configure first subplot
        ax1.set_xlabel('QEC Method')
        ax1.set_ylabel('Average Fidelity')
        ax1.set_title('Raw vs. Loss-Adjusted Fidelity by QEC Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels(qec_labels)
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(axis='y', linestyle=':', alpha=0.7)
        
        # PLOT 2: Loss ratio
        bars3 = ax2.bar(x, qec_agg['loss_ratio_mean'], width*1.5, 
                       yerr=qec_agg['loss_ratio_std'], alpha=0.8, color='red')
        
        # Configure second subplot
        ax2.set_xlabel('QEC Method')
        ax2.set_ylabel('Average Loss Ratio')
        ax2.set_title('Loss Ratio by QEC Method')
        ax2.set_xticks(x)
        ax2.set_xticklabels(qec_labels)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "qec_method_comparison_overall.png"), dpi=300)
        plt.close()
    
    # 2. QEC Method comparison by error combination
    error_combos = sorted(df['error_combo'].unique())
    
    if error_combos:
        n_cols = min(2, len(error_combos))
        n_rows = (len(error_combos) + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), sharex=True, sharey=True)
        if n_rows * n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1 or n_cols == 1:
            axs = axs.reshape(-1, 1) if n_cols == 1 else axs.reshape(1, -1)
            
        for i, error_combo in enumerate(error_combos):
            row, col = i // n_cols, i % n_cols
            ax = axs[row, col]
            
            # Get data for this error combination
            combo_df = df[df['error_combo'] == error_combo]
            
            if combo_df.empty:
                ax.text(0.5, 0.5, f"No data for {error_combo}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Aggregate by QEC method for this error combo
            combo_agg = combo_df.groupby('qec_method').agg(
                raw_fid_mean=('avg_fidelity_raw', 'mean'),
                adj_fid_mean=('avg_fidelity_adj', 'mean'),
                loss_ratio_mean=('loss_ratio', 'mean'),
                count=('loss_ratio', 'count')
            ).reset_index()
            
            if combo_agg.empty:
                continue
                
            # Sort by adjusted fidelity
            combo_agg = combo_agg.sort_values('adj_fid_mean', ascending=False)
            
            # Set up x positions
            qec_labels = [METHOD_DISPLAY_MAP.get(m, m) for m in combo_agg['qec_method']]
            x = np.arange(len(qec_labels))
            width = 0.3
            
            # Plot data
            bars1 = ax.bar(x - width/2, combo_agg['raw_fid_mean'], width, 
                         label='Raw Fidelity', alpha=0.8, color='blue')
            
            bars2 = ax.bar(x + width/2, combo_agg['adj_fid_mean'], width, 
                         label='Loss-Adj. Fidelity', alpha=0.8, color='green')
            
            # Add loss ratio as text
            for i, (x_pos, loss) in enumerate(zip(x, combo_agg['loss_ratio_mean'])):
                ax.text(x_pos, 0.02, f"Loss: {loss:.2f}", 
                      ha='center', va='bottom', fontsize=8, rotation=90)
            
            # Configure subplot
            ax.set_title(f"{error_combo}")
            ax.set_xticks(x)
            ax.set_xticklabels(qec_labels, rotation=30, ha='right')
            ax.set_ylim(0, 1.05)
            
            if col == 0:  # First column
                ax.set_ylabel("Fidelity")
                
            # Add legend to the first subplot only
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=9)
            
            ax.grid(axis='y', linestyle=':', alpha=0.7)
        
        # Hide unused subplots
        for i in range(len(error_combos), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axs[row, col].set_visible(False)
        
        plt.suptitle("QEC Method Performance by Error Combination", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "qec_method_by_error_combo.png"), dpi=300)
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    configure_plots()
    
    # Load the data (no need for full data for these plots)
    all_distance_data = load_distance_qec_data()
    
    if not all_distance_data:
        print("No distance data loaded, cannot perform analysis.")
    else:
        # Convert list of dictionaries to DataFrame
        df_distance_metrics = pd.DataFrame(all_distance_data)
        print(f"Loaded {len(df_distance_metrics)} experiment configurations.")
        
        # Map error combo names to shorter versions for better display
        if 'error_combo' in df_distance_metrics.columns:
            df_distance_metrics['error_combo'] = df_distance_metrics['error_combo'].map(
                ERROR_COMBO_MAP).fillna(df_distance_metrics['error_combo'])
            
        if len(df_distance_metrics) > 0:
            print("\nExample loaded distance data point:")
            print(df_distance_metrics.iloc[0])
            
            # Diagnostic information
            print("\n=== Diagnostic Information ===")
            print(f"Unique distances: {sorted(df_distance_metrics['distance'].unique())}")
            print(f"Unique error combos: {sorted(df_distance_metrics['error_combo'].unique())}")
            print(f"Unique QEC methods: {sorted(df_distance_metrics['qec_method'].unique())}")
            
            if 'Fiber' in df_distance_metrics['error_combo'].unique():
                fiber_df = df_distance_metrics[df_distance_metrics['error_combo'] == 'Fiber']
                print(f"Found {len(fiber_df)} fiber loss entries")
            
            # Call all the analysis functions
            plot_fidelity_by_error_combo(df_distance_metrics)
            plot_fidelity_vs_distance_combined(df_distance_metrics)
            plot_loss_vs_distance_combined(df_distance_metrics)
            plot_qec_performance_faceted_by_error(df_distance_metrics)
            plot_fiber_params_analysis(df_distance_metrics)
            plot_qec_loss_comparison(df_distance_metrics)
            plot_qec_method_comparison(df_distance_metrics)
            
            print(f"\nAnalysis complete. Plots saved to: {PLOTS_DIR}") 