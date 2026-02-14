import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

LOGS_DIR = 'logs'
FIGS_DIR = 'figures'

# Read the CSV file
CSV_PATH = os.path.join(LOGS_DIR, 'full.csv')
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please ensure the file exists.")
df = pd.read_csv(CSV_PATH, sep=';')

os.makedirs(FIGS_DIR, exist_ok=True)

# Filter for TEST phase only
test_df = df[df['phase'] == 'TEST'].copy()

# Group by hidden_layer_size, learning_rate, and momentum, then average across seeds
grouped_loss = test_df.groupby(['hidden_layer_size', 'learning_rate', 'momentum'])['loss'].mean().reset_index()
grouped_accuracy = test_df.groupby(['hidden_layer_size', 'learning_rate', 'momentum'])['accuracy'].mean().reset_index()

for grouped, metric in [(grouped_loss, 'loss'), (grouped_accuracy, 'accuracy')]:

    # Get unique values for each parameter
    hidden_sizes = sorted(grouped['hidden_layer_size'].unique())
    learning_rates = sorted(grouped['learning_rate'].unique())
    momentums = sorted(grouped['momentum'].unique())

    # Create a combination label for each learning_rate + momentum pair
    grouped['combo'] = grouped.apply(
        lambda row: f"LR={row['learning_rate']}, M={row['momentum']}", axis=1
    )

    # Get unique combinations
    combinations = sorted(grouped['combo'].unique())

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Set the width of bars and positions
    n_combos = len(combinations)
    n_sizes = len(hidden_sizes)
    bar_width = 0.8 / n_combos
    x = np.arange(n_sizes)

    # Map learning rates to base colors
    base_colors = ['blue', 'red', 'green']
    lr_to_color = {lr: base_colors[i] for i, lr in enumerate(learning_rates)}

    # Normalize momentum values to [0.5, 1] for intensity (0.5 = lighter, 1 = full color)
    momentum_min, momentum_max = min(momentums), max(momentums)
    def get_intensity(momentum):
        if momentum_max == momentum_min:
            return 1.0
        return 0.5 + 0.5 * (momentum - momentum_min) / (momentum_max - momentum_min)

    def adjust_color(color, intensity):
        # Blend color with white based on intensity
        rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1, 1, 1])
        return rgb * intensity + white * (1 - intensity)

    # Plot bars for each combination
    for i, combo in enumerate(combinations):
        combo_data = grouped[grouped['combo'] == combo]

        # Parse learning rate and momentum from combo string
        lr = float(combo.split(',')[0].split('=')[1])
        m = float(combo.split(',')[1].split('=')[1])
        base_color = lr_to_color[lr]
        intensity = get_intensity(m)
        color = adjust_color(base_color, intensity)

        
        # Create a list of metrics aligned with hidden_sizes
        metrics = []
        for size in hidden_sizes:
            size_data = combo_data[combo_data['hidden_layer_size'] == size]
            if len(size_data) > 0:
                metrics.append(size_data[metric].values[0])
            else:
                metrics.append(0)  # If no data for this combination
        
        # Calculate position for this group of bars
        positions = x + (i - n_combos/2 + 0.5) * bar_width
        
        # Plot the bars
        ax.bar(positions, metrics, bar_width, label=combo, color=color, alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Hidden Layer Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average Test {metric.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Test {metric.capitalize()} by Hidden Layer Size and Hyperparameters\n(Averaged across seeds)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_sizes)
    ax.legend(title='Learning Rate & Momentum', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars (optional, can be commented out if too cluttered)
    for i, combo in enumerate(combinations):
        combo_data = grouped[grouped['combo'] == combo]
        metrics = []
        for size in hidden_sizes:
            size_data = combo_data[combo_data['hidden_layer_size'] == size]
            if len(size_data) > 0:
                metrics.append(size_data[metric].values[0])
            else:
                metrics.append(0)
        
        positions = x + (i - n_combos/2 + 0.5) * bar_width
        
        for pos, value in zip(positions, metrics):
            if value > 0:  # Only show label if there's data
                ax.text(pos, value + 0.02, f'{value:.3f}', ha='center', va='bottom', 
                    fontsize=7, rotation=90) # TODO: CAMBIARE rotation A 0 PER ORIZZONTALE

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'test_{metric}.png'), dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {os.path.join(FIGS_DIR, f'test_{metric}.png')}")
    print(f"\nData summary:")
    print(f"Hidden layer sizes: {hidden_sizes}")
    print(f"Learning rates: {learning_rates}")
    print(f"Momentum values: {momentums}")
    print(f"Total combinations: {n_combos}")
    plt.show()