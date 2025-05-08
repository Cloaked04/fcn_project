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
RESULTS_DIR = "results_distance_qec"
PLOTS_DIR = "plots_distance_qec_no_fiber_loss"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Plotting Configuration ---
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
        'figure.figsize': (12, 7),
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
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    return str(value)

def gamma_to_time(gamma):
    """Converts gamma (decoherence rate in Hz) to coherence time with appropriate units."""
    if gamma <= 0:
        return "∞"
    
    coherence_time = 1.0 / gamma  # Time in seconds
    
    # Always display in nanoseconds for phase damping plots to avoid unit conversion issues
    return f"{coherence_time * 1e9:.2f} ns"
    
    # Original conversion logic - commented out
    # if coherence_time < 1e-6:
    #     return f"{coherence_time * 1e9:.2f} ns"
    # elif coherence_time < 1e-3:
    #     return f"{coherence_time * 1e6:.2f} μs"
    # elif coherence_time < 1:
    #     return f"{coherence_time * 1e3:.2f} ms"
    # else:
    #     return f"{coherence_time:.2f} s"

def parse_param_string(param_str):
    """Parses parameter strings from distance experiments.
       Handles gamma values and other parameters.
    """
    params = {}
    parts = param_str.split('_')
    i = 0
    current_key = ""
    
    while i < len(parts):
        part = parts[i]
        # Check if part looks like a value
        is_value = False
        try:
            float(part)
            is_value = True
        except ValueError:
            is_value = False

        if not is_value and current_key:
            # If we have a key and current part isn't a value, it's part of the key
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
            # Edge case
            i += 1

    return params

def format_params_dict(params_dict):
    """Creates a readable string representation of the params dict, formatting gamma."""
    if not params_dict:
        return "No Params"
    items = []
    for k, v in sorted(params_dict.items()):
        if 'gamma' in k:
            items.append(f"{k}={format_gamma(v, precision=4)}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)

# Add helper function after the other helper functions but before the load_distance_qec_data function
def extract_gamma_from_params(error_params):
    """Extract and validate gamma value from error parameters dictionary."""
    if not error_params or not isinstance(error_params, dict):
        return None
    
    gamma = error_params.get('gamma')
    if gamma is not None and gamma > 0:
        return gamma
    
    # Check for sec_gamma or ter_gamma if primary gamma not found
    for key in ['sec_gamma', 'ter_gamma']:
        if key in error_params and error_params[key] > 0:
            return error_params[key]
    
    return None

# --- Data Loading Function --- 
def load_distance_qec_data(results_dir=RESULTS_DIR):
    """
    Loads experiment data from the distance_qec_network_no_fiber_loss results structure.
    """
    print(f"Loading distance experiment data from: {results_dir}")
    all_experiments_data = []
    skipped_files = 0
    loaded_count = 0

    # Regex to capture: state, qec_method, distance, params_string
    # Example: +_none_d10.0_gamma_10000.0.csv
    filename_pattern = re.compile(r"^([+\-01])_(.+?)_d([\d\.]+?)_(.+)\.csv$")

    # Iterate through error combination directories
    error_combo_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for error_combo in error_combo_dirs:
        dir_path = os.path.join(results_dir, error_combo)

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
                    error_params = parse_param_string(param_string)
                    if not error_params:
                        skipped_files += 1
                        continue

                    # Load metadata
                    metadata_df = pd.read_csv(metadata_path)
                    if metadata_df.empty:
                        skipped_files += 1
                        continue
                    metadata = metadata_df.iloc[0].to_dict()

                    # Extract required fields from metadata
                    avg_fidelity = metadata.get('avg_fidelity', 0.0)
                    std_fidelity = metadata.get('std_fidelity', 0.0)
                    iterations_completed = metadata.get('iterations_completed', 0)

                    experiment_entry = {
                        "initial_state": initial_state,
                        "qec_method": qec_method,
                        "distance": distance,
                        "error_type": error_combo,  # Store directory name as error type
                        "error_params": error_params,
                        "param_string": format_params_dict(error_params),
                        "avg_fidelity": avg_fidelity,
                        "std_fidelity": std_fidelity,
                        "iterations": iterations_completed,
                        "filename_base": base_filename
                    }

                    # Extract gamma directly during data loading
                    gamma_value = extract_gamma_from_params(error_params)
                    if gamma_value is not None:
                        # Calculate coherence times
                        t1_time = float('inf') if error_combo != 'amplitude_damping' else 1e9 / gamma_value
                        t2_time = float('inf') if error_combo != 'phase_damping' else 1e9 / gamma_value
                    else:
                        # Default values if gamma not found or invalid
                        gamma_value = None
                        t1_time = float('inf')
                        t2_time = float('inf')

                    # Add gamma and coherence times to experiment entry
                    experiment_entry["gamma"] = gamma_value
                    experiment_entry["t1_time"] = t1_time
                    experiment_entry["t2_time"] = t2_time

                    all_experiments_data.append(experiment_entry)
                    loaded_count += 1

                except Exception as e:
                    print(f"  - Error processing metadata file {filename}: {e}")
                    skipped_files += 1
            else:
                skipped_files += 1

    print(f"Finished loading. Loaded {loaded_count} experiment configurations from metadata.")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files/metadata (check format/content/errors).")
    
    return all_experiments_data

# --- Mapping for better display ---
ERROR_TYPE_MAP = {
    'amplitude_damping': 'Amplitude Damping ($T_1$)',
    'phase_damping': 'Phase Damping ($T_2$)'
}

METHOD_DISPLAY_MAP = {
    'none': 'No QEC',
    'three_qubit_phase_flip': '3QB Phase',
    'shor_nine': 'Shor-9'
}

# --- Plot 1: Fidelity vs Error Type ---
def plot_fidelity_by_error_type(df, plots_subdir="1_fidelity_vs_error_type"):
    """
    Plot average fidelity for each error type, aggregating across all other parameters.
    """
    print("\n--- Plot 1: Fidelity vs Error Type ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    # Aggregate fidelities by error type
    agg_data = df.groupby('error_type').agg(
        mean_fid=('avg_fidelity', 'mean'),
        std_fid=('avg_fidelity', 'std')
    ).reset_index()
    
    if agg_data.empty:
        print("  No data to plot for Fidelity vs Error Type.")
        return

    # Sort by mean fidelity
    agg_data = agg_data.sort_values(by='mean_fid', ascending=False)
    
    # Map error types to display names
    agg_data['error_type_display'] = agg_data['error_type'].map(
        ERROR_TYPE_MAP).fillna(agg_data['error_type'])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with error bars
    bars = ax.bar(agg_data['error_type_display'], agg_data['mean_fid'], 
                 yerr=agg_data['std_fid'], capsize=5, 
                 color=plt.cm.viridis(np.linspace(0.2, 0.8, len(agg_data))),
                 edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # Customize plot
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'Error Type', fontsize=14)
    ax.set_ylabel(r'Average Fidelity $F$', fontsize=14)
    ax.set_title(r'Average Fidelity by Error Type', fontsize=16)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "avg_fidelity_by_error_type.png"))
    plt.close()

# --- Plot 2: Fidelity vs Gamma ---
def plot_fidelity_vs_gamma(df, plots_subdir="2_fidelity_vs_gamma"):
    """Plot fidelity vs gamma for different QEC methods and initial states."""
    print("\n--- Plotting Fidelity vs. Gamma ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Only include amplitude_damping and phase_damping error types
    damping_df = df[df['error_type'].isin(['amplitude_damping', 'phase_damping'])].copy()
    if damping_df.empty:
        print("  No damping data found for plotting.")
        return
    
    # Extract gamma values for plotting
    damping_df['gamma'] = damping_df['error_params'].apply(extract_gamma_from_params)
    # Remove rows with invalid gamma values
    damping_df = damping_df.dropna(subset=['gamma'])
    
    # Create plots for each error type
    for error_type in ['amplitude_damping', 'phase_damping']:
        error_df = damping_df[damping_df['error_type'] == error_type].copy()
        if error_df.empty:
            continue
            
        # Process data for secondary axis (coherence time)
        gamma_ticks = [1e4, 1e5, 1e6, 1e7]  # Powers of 10 for gamma
        time_labels = [gamma_to_time(g) for g in gamma_ticks]
        
        # Create a plot for each QEC method
        for qec_method in error_df['qec_method'].unique():
            qec_df = error_df[error_df['qec_method'] == qec_method].copy()
            
            # Create figure with larger height to accommodate labels
            fig = plt.figure(figsize=(11, 8.5))
            # Use gridspec to control the layout more precisely
            gs = fig.add_gridspec(1, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Create a secondary x-axis with more space at top
            ax2 = ax1.twiny()
            
            # Set up colors for states
            states = qec_df['initial_state'].unique()
            colors = plt.cm.viridis(np.linspace(0, 0.85, len(states)))
            state_colors = {state: colors[i] for i, state in enumerate(states)}
            
            # Plot fidelity vs gamma for each initial state
            for state in states:
                state_df = qec_df[qec_df['initial_state'] == state]
                if state_df.empty:
                    continue
                    
                # Group by gamma value and average across distances
                grouped = state_df.groupby('gamma').agg(
                    avg_fid=('avg_fidelity', 'mean'),
                    std_fid=('avg_fidelity', 'std')
                ).reset_index()
                
                # Plot data points with error bars
                ax1.errorbar(grouped['gamma'], grouped['avg_fid'], 
                           yerr=grouped['std_fid'], 
                           label=f"State: |{state}⟩",
                           marker='o', linestyle='-', linewidth=2, 
                           color=state_colors[state], markersize=8, capsize=5)
                
                # Add shaded error regions
                ax1.fill_between(grouped['gamma'], 
                               grouped['avg_fid'] - grouped['std_fid'],
                               grouped['avg_fid'] + grouped['std_fid'],
                               color=state_colors[state], alpha=0.2)
            
            # Configure primary x-axis (gamma)
            ax1.set_xscale('log')
            ax1.set_xlim(1e4, 1e7)  # Fixed x-axis limits
            ax1.set_xticks(gamma_ticks)
            ax1.set_xticklabels([f"$10^{int(np.log10(g))}$" for g in gamma_ticks])
            ax1.set_xlabel(f"Decoherence Rate γ (Hz)")
            
            # Configure secondary x-axis (coherence time)
            ax2.set_xscale('log')
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(gamma_ticks)
            ax2.set_xticklabels(time_labels)
            
            # Configure y-axis
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Average Fidelity $F$")
            
            # Set up plot title and labels
            error_name = error_type.replace('_', ' ').title()
            qec_name = qec_method.replace('_', ' ').title() if qec_method != 'none' else 'No QEC'
            
            # Add title with enough padding
            ax1.set_title(f"{error_name} - {qec_name} - Fidelity by Initial State", pad=45)
            
            # Add the coherence time label with proper positioning 
            if error_type == 'amplitude_damping':
                coherence_time_label = "T1 Coherence Time"
            else:  # phase_damping
                coherence_time_label = "T2 Coherence Time"
            
            ax2.set_xlabel(coherence_time_label, labelpad=15)
            
            # Adjust grid
            ax1.grid(True, alpha=0.3, linestyle=':')
            
            # Move legend outside the plot
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            # Adjust spacing - very important for fixing overlap
            plt.tight_layout()
            # Add more space at top explicitly
            plt.subplots_adjust(top=0.80)
            
            # Save without bbox_inches to prevent it from overriding our manual adjustment
            plt.savefig(os.path.join(plot_dir, f"{error_type}_{qec_method}_fidelity_vs_gamma.png"))
            plt.close()

# --- Plot 3: Fidelity vs Distance ---
def plot_fidelity_vs_distance(df, plots_subdir="3_fidelity_vs_distance"):
    """
    Plot fidelity as a function of distance for each error type and gamma value.
    """
    print("\n--- Plot 3: Fidelity vs Distance ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Check if we have multiple distance values
    distances = sorted(df['distance'].unique())
    if len(distances) <= 1:
        print("  Not enough distance points to create distance plot.")
        return
    
    # Set up color scheme by gamma value
    gamma_values = sorted(df['gamma'].unique())
    # Use only specific gamma values to avoid interpolation issues
    valid_gamma_values = [g for g in gamma_values if g >= 1e4]
    if not valid_gamma_values:
        print("  No valid gamma values (≥ 10^4) for distance plot.")
        return
    
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(valid_gamma_values)))
    gamma_colors = {gamma: colors[i] for i, gamma in enumerate(valid_gamma_values)}
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set up marker scheme by error type
    error_types = sorted(df['error_type'].unique())
    markers = ['o', 's', '^', 'd', '*']
    error_markers = {error: markers[i % len(markers)] for i, error in enumerate(error_types)}
    
    # Set up linestyle by QEC method
    qec_methods = sorted(df['qec_method'].unique())
    linestyles = ['-', '--', '-.', ':']
    qec_lines = {qec: linestyles[i % len(linestyles)] for i, qec in enumerate(qec_methods)}
    
    # Track for legend
    gamma_handles = {}
    error_handles = {}
    qec_handles = {}
    
    # Plot for each error type, gamma value, and QEC method
    for error_type in error_types:
        for gamma in valid_gamma_values:
            for qec_method in qec_methods:
                # Filter data
                subset = df[(df['error_type'] == error_type) & 
                           (df['gamma'] == gamma) &
                           (df['qec_method'] == qec_method)]
                
                if len(subset) < 2:
                    continue
                
                # Average across initial states for each distance
                avg_by_dist = subset.groupby('distance').agg(
                    avg_fid=('avg_fidelity', 'mean'),
                    std_fid=('avg_fidelity', 'std')
                ).reset_index()
                
                # Sort by distance for proper line drawing
                avg_by_dist = avg_by_dist.sort_values('distance')
                
                if len(avg_by_dist) < 2:
                    continue
                
                # Plot line
                line, = ax.plot(avg_by_dist['distance'], avg_by_dist['avg_fid'],
                              marker=error_markers[error_type],
                              linestyle=qec_lines[qec_method],
                              color=gamma_colors[gamma],
                              markersize=8, linewidth=2,
                              label=f"{ERROR_TYPE_MAP.get(error_type, error_type)} - γ={format_gamma(gamma, precision=4)} ({gamma_to_time(gamma)}) - {METHOD_DISPLAY_MAP.get(qec_method, qec_method)}")
                
                # Add error bars
                ax.fill_between(avg_by_dist['distance'],
                              avg_by_dist['avg_fid'] - avg_by_dist['std_fid'],
                              avg_by_dist['avg_fid'] + avg_by_dist['std_fid'],
                              color=gamma_colors[gamma], alpha=0.1)
                
                # Track for legend
                time_type = "T1" if error_type == "amplitude_damping" else "T2"
                gamma_label = f"γ={format_gamma(gamma, precision=4)} ({time_type}={gamma_to_time(gamma)})"
                if gamma_label not in gamma_handles:
                    gamma_handles[gamma_label] = line
                
                error_display = ERROR_TYPE_MAP.get(error_type, error_type)
                if error_display not in error_handles:
                    error_handles[error_display] = error_markers[error_type]
                
                qec_display = METHOD_DISPLAY_MAP.get(qec_method, qec_method)
                if qec_display not in qec_handles:
                    qec_handles[qec_display] = qec_lines[qec_method]
    
    # Set up plot appearance
    ax.set_title(r"Fidelity vs. Distance", fontsize=16)
    ax.set_xlabel(r"Distance (km)", fontsize=14)
    ax.set_ylabel(r"Average Fidelity $F$", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Create multi-part legend
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Gamma value legend elements
    for gamma_label, line in gamma_handles.items():
        legend_elements.append(Line2D([0], [0], color=line.get_color(), lw=2, 
                                    label=gamma_label))
    
    # Error type legend elements
    for error_name, marker in error_handles.items():
        legend_elements.append(Line2D([0], [0], color='gray', marker=marker, 
                                     linestyle='None', markersize=8, 
                                     label=error_name))
    
    # QEC method legend elements
    for qec_name, ls in qec_handles.items():
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls,
                                     lw=2, label=f"QEC: {qec_name}"))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=10, 
             framealpha=0.9, ncol=1, bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fidelity_vs_distance.png"))
    plt.close()

# --- Plot 4: QEC Method Performance by Error Type ---
def plot_qec_performance(df, plots_subdir="4_qec_performance"):
    """
    Plot QEC method performance for each error type and gamma value.
    """
    print("\n--- Plot 4: QEC Method Performance ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Filter data to include only valid gamma values
    df_filtered = df[df['gamma'] >= 1e4].copy()
    
    if df_filtered.empty:
        print("  No valid gamma values for QEC performance plots.")
        return
    
    # Set up for separate plots by error type
    error_types = sorted(df_filtered['error_type'].unique())
    
    for error_type in error_types:
        error_df = df_filtered[df_filtered['error_type'] == error_type]
        error_display = ERROR_TYPE_MAP.get(error_type, error_type)
        
        # Get unique gamma values for this error type
        gamma_values = sorted(error_df['gamma'].unique())
        
        # Set up colors for gamma values
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(gamma_values)))
        gamma_colors = {gamma: colors[i] for i, gamma in enumerate(gamma_values)}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width of bars
        num_gammas = len(gamma_values)
        num_methods = len(df['qec_method'].unique())
        bar_width = 0.8 / num_gammas
        
        # Plot bars for each QEC method and gamma value
        for i, gamma in enumerate(gamma_values):
            gamma_df = error_df[error_df['gamma'] == gamma]
            
            # Aggregate by QEC method
            agg_data = gamma_df.groupby('qec_method').agg(
                avg_fid=('avg_fidelity', 'mean'),
                std_fid=('avg_fidelity', 'std')
            ).reset_index()
            
            # Map QEC methods to display names
            agg_data['qec_display'] = agg_data['qec_method'].map(
                METHOD_DISPLAY_MAP).fillna(agg_data['qec_method'])
            
            # Sort by QEC method for consistent order
            agg_data = agg_data.sort_values('qec_method')
            
            # Calculate positions for bars
            x = np.arange(len(agg_data))
            offset = (i - num_gammas/2 + 0.5) * bar_width
            
            # Create bars
            bars = ax.bar(x + offset, agg_data['avg_fid'], bar_width,
                         yerr=agg_data['std_fid'], capsize=3,
                         color=gamma_colors[gamma], edgecolor='black',
                         linewidth=0.8, label=f"γ={format_gamma(gamma, precision=4)} ({gamma_to_time(gamma)})")
            
            # Add values on top of bars for first and last gamma (to avoid clutter)
            if i == 0 or i == len(gamma_values) - 1:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.1:  # Only add text if bar is tall enough
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{height:.2f}', ha='center', va='bottom',
                               fontsize=8, rotation=90)
        
        # Customize plot
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r'QEC Method', fontsize=14)
        ax.set_ylabel(r'Average Fidelity $F$', fontsize=14)
        ax.set_title(f"{error_display} - QEC Method Performance", fontsize=16)
        ax.set_xticks(np.arange(len(METHOD_DISPLAY_MAP)))
        ax.set_xticklabels([METHOD_DISPLAY_MAP[m] for m in sorted(METHOD_DISPLAY_MAP.keys())])
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        
        # Adjust legend title based on error type
        if error_type == 'amplitude_damping':
            legend_title = r"Decoherence Rate (T1 Coherence Time)"
        else:  # phase_damping
            legend_title = r"Decoherence Rate (T2 Coherence Time)"
        
        ax.legend(title=legend_title, loc='upper right',
                 fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"qec_performance_{error_type}.png"))
        plt.close()

# --- Plot 5: Fidelity vs Gamma for Different Initial States ---
def plot_fidelity_by_initial_state(df, plots_subdir="5_fidelity_by_state"):
    """
    Plot fidelity vs gamma for different initial states, comparing performance.
    """
    print("\n--- Plot 5: Fidelity by Initial State ---")
    plot_dir = os.path.join(PLOTS_DIR, plots_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Filter to keep only gamma values ≥ 10^4 for cleaner plots
    df_filtered = df[df['gamma'] >= 1e4].copy()
    if df_filtered.empty:
        print("  No valid gamma values for initial state plots.")
        return
    
    # Get unique values
    error_types = sorted(df_filtered['error_type'].unique())
    qec_methods = sorted(df_filtered['qec_method'].unique())
    
    # Set up colors for states
    initial_states = sorted(df_filtered['initial_state'].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(initial_states)))
    state_colors = {state: colors[i] for i, state in enumerate(initial_states)}
    
    # Plot for each error type and QEC method
    for error_type in error_types:
        for qec_method in qec_methods:
            # Filter data
            subset = df_filtered[(df_filtered['error_type'] == error_type) & 
                               (df_filtered['qec_method'] == qec_method)]
            
            if subset.empty:
                continue
                
            # Create figure with taller size
            fig = plt.figure(figsize=(12, 8.5))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            
            # Plot for each initial state
            for state in initial_states:
                state_df = subset[subset['initial_state'] == state]
                
                if state_df.empty:
                    continue
                    
                # Aggregate by gamma
                avg_by_gamma = state_df.groupby('gamma').agg(
                    avg_fid=('avg_fidelity', 'mean'),
                    std_fid=('avg_fidelity', 'std')
                ).reset_index()
                
                # Sort by gamma
                avg_by_gamma = avg_by_gamma.sort_values('gamma')
                
                if len(avg_by_gamma) < 2:
                    continue
                
                # Plot line
                state_display = state
                if state in ['+', '-']:
                    state_display = f"$|{state}\\rangle$"
                else:
                    state_display = f"$|{state}\\rangle$"
                    
                line, = ax.plot(avg_by_gamma['gamma'], avg_by_gamma['avg_fid'],
                              marker='o', markersize=8, linewidth=2,
                              color=state_colors[state], 
                              label=state_display)
                
                # Add error bars
                ax.fill_between(avg_by_gamma['gamma'],
                              avg_by_gamma['avg_fid'] - avg_by_gamma['std_fid'],
                              avg_by_gamma['avg_fid'] + avg_by_gamma['std_fid'],
                              color=state_colors[state], alpha=0.1)
            
            # Use log scale for x-axis
            ax.set_xscale('log')
            
            # Calculate T1/T2 values for secondary x-axis
            gamma_ticks = [1e4, 1e5, 1e6, 1e7]
            ax.set_xticks(gamma_ticks)
            
            # Use LaTeX formatting for gamma tick labels
            gamma_labels = []
            for g in gamma_ticks:
                if g in [1e4, 1e5, 1e6, 1e7]:
                    exponent = int(math.log10(g))
                    gamma_labels.append(f"$10^{{{exponent}}}$")
                else:
                    gamma_labels.append(format_gamma(g, precision=4))
            ax.set_xticklabels(gamma_labels)
            
            # Create secondary x-axis for T1/T2 times
            ax2 = ax.twiny()
            ax2.set_xscale('log')
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(gamma_ticks)
            ax2.set_xticklabels([gamma_to_time(g) for g in gamma_ticks])
            
            # Set up plot appearance
            error_display = ERROR_TYPE_MAP.get(error_type, error_type)
            qec_display = METHOD_DISPLAY_MAP.get(qec_method, qec_method)
            
            # Add title with significantly more padding
            ax.set_title(f"{error_display} - {qec_display} - Fidelity by Initial State", 
                       fontsize=16, pad=50)
            
            ax.set_xlabel(r"Decoherence Rate $\gamma$ (Hz)", fontsize=14)
            
            # Set appropriate coherence time label based on error type
            if error_type == 'amplitude_damping':
                coherence_time_label = r"T1 Coherence Time"
            else:  # phase_damping
                coherence_time_label = r"T2 Coherence Time"
            
            # Add more spacing for the top label
            ax2.set_xlabel(coherence_time_label, fontsize=14, labelpad=15)
            
            ax.set_ylabel(r"Average Fidelity $F$", fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(title="Initial State", loc='best', fontsize=10, framealpha=0.9)
            
            # Adjust layout with more explicit control
            plt.tight_layout()
            # Create significant space at the top
            plt.subplots_adjust(top=0.80)
            
            # Save without bbox_inches to prevent it from overriding our manual adjustment
            plt.savefig(os.path.join(plot_dir, f"{error_type}_{qec_method}_by_state.png"))
            plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    configure_plots()
    
    # Load the data
    all_distance_data = load_distance_qec_data()
    
    if not all_distance_data:
        print("No distance data loaded, cannot perform analysis.")
    else:
        # Convert list of dictionaries to DataFrame
        df_distance_metrics = pd.DataFrame(all_distance_data)
        print(f"Raw loaded configurations: {len(df_distance_metrics)}")
        
        # Filter out records with no or invalid gamma values
        valid_gamma_df = df_distance_metrics.dropna(subset=['gamma'])
        print(f"Filtered out {len(df_distance_metrics) - len(valid_gamma_df)} rows with missing gamma values")
        
        # Filter out gamma values ≤ 0 and very small values < 0.1
        # This more aggressive filtering prevents interpolation issues in plots
        df_filtered = valid_gamma_df[valid_gamma_df['gamma'] >= 0.1].copy()
        print(f"Filtered out {len(valid_gamma_df) - len(df_filtered)} additional rows with gamma < 0.1")
        
        df_distance_metrics = df_filtered
        print(f"Analyzing {len(df_distance_metrics)} experiment configurations after filtering.")
            
        if len(df_distance_metrics) > 0:
            print("\nExample loaded distance data point:")
            print(df_distance_metrics.iloc[0])
            
            # Diagnostic information
            print("\n=== Diagnostic Information ===")
            print(f"Unique distances: {sorted(df_distance_metrics['distance'].unique())}")
            print(f"Unique error types: {sorted(df_distance_metrics['error_type'].unique())}")
            print(f"Unique QEC methods: {sorted(df_distance_metrics['qec_method'].unique())}")
            print(f"Unique gamma values: {sorted(df_distance_metrics['gamma'].unique())}")
            
            # Call all the analysis functions
            plot_fidelity_by_error_type(df_distance_metrics)
            plot_fidelity_vs_gamma(df_distance_metrics)
            plot_fidelity_vs_distance(df_distance_metrics)
            plot_qec_performance(df_distance_metrics)
            plot_fidelity_by_initial_state(df_distance_metrics)
            
            print(f"\nAnalysis complete. Plots saved to: {PLOTS_DIR}") 