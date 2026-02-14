import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

LOGS_DIR = 'logs'
FIGS_DIR = 'figures/15_epochs'

# Read the CSV file
CSV_PATH = os.path.join(LOGS_DIR, 'full.csv')
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please ensure the file exists.")
df = pd.read_csv(CSV_PATH, sep=';')

os.makedirs(FIGS_DIR, exist_ok=True)


phase_name_mapping = {
    'TRAIN': 'Training',
    'VAL': 'Validation'
}


for phase in ['TRAIN', 'VAL']:
    os.makedirs(os.path.join(FIGS_DIR, phase), exist_ok=True)

    for metric in ['loss', 'accuracy']:
        os.makedirs(os.path.join(FIGS_DIR, phase, metric), exist_ok=True)

        # Filter for current phase only
        phase_df = df[df['phase'] == phase].copy()

        # Get unique hidden layer sizes
        hidden_sizes = sorted(phase_df['hidden_layer_size'].unique())

        for FIXED_HIDDEN_SIZE in hidden_sizes:
            # Filter for the fixed hidden layer size
            size_df = phase_df[phase_df['hidden_layer_size'] == FIXED_HIDDEN_SIZE].copy()

            # Group by epoch, learning_rate, and momentum, then average across seeds
            grouped_mean = size_df.groupby(['epoch', 'learning_rate', 'momentum'])[metric].mean().reset_index()
            grouped_std = size_df.groupby(['epoch', 'learning_rate', 'momentum'])[metric].std().reset_index()

            grouped = pd.merge(
                grouped_mean,
                grouped_std,
                on=['epoch', 'learning_rate', 'momentum'],
                suffixes=('', '__std')
            )

            # Get unique values
            learning_rates = sorted(grouped['learning_rate'].unique())
            momentums = sorted(grouped['momentum'].unique())
            epochs = sorted(grouped['epoch'].unique())

            # Create a combination label for each learning_rate + momentum pair
            grouped['combo'] = grouped.apply(
                lambda row: f"LR={row['learning_rate']}, M={row['momentum']}", axis=1
            )

            # Get unique combinations
            combinations = sorted(grouped['combo'].unique())

            # Set up the plot
            fig, ax = plt.subplots(figsize=(12, 7))


            # Choose custom base colors for learning rates
            custom_colors = ['blue', 'red', 'green']  # Extend if you have more learning rates
            lr_to_color = {lr: custom_colors[i] for i, lr in enumerate(learning_rates)}
            momentum_min, momentum_max = min(momentums), max(momentums)
            def get_intensity(momentum):
                if momentum_max == momentum_min:
                    return 1.0
                return 0.5 + 0.5 * (momentum - momentum_min) / (momentum_max - momentum_min)

            def adjust_color(color, intensity):
                rgb = np.array(mcolors.to_rgb(color))
                white = np.array([1, 1, 1])
                return rgb * intensity + white * (1 - intensity)




            # Plot lines for each combination
            for i, combo in enumerate(combinations):
                combo_data = grouped[grouped['combo'] == combo].sort_values('epoch')
                lr = float(combo.split(',')[0].split('=')[1])
                m = float(combo.split(',')[1].split('=')[1])
                base_color = lr_to_color[lr]
                intensity = get_intensity(m)
                color = adjust_color(base_color, intensity)

                # Plot mean line
                ax.plot(combo_data['epoch'], combo_data[metric], 
                        marker='o', 
                        linestyle='-',
                        linewidth=2,
                        markersize=6,
                        label=combo, 
                        color=color,
                        alpha=0.8)
                
                # Plot std deviation as shaded region
                ax.fill_between(
                    combo_data['epoch'],
                    combo_data[metric] - combo_data[f'{metric}__std'],
                    combo_data[metric] + combo_data[f'{metric}__std'],
                    color=color,
                    alpha=0.2,
                    label=None
                )


            # Customize the plot
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Average {phase_name_mapping[phase]} {metric.capitalize()}', fontsize=12, fontweight='bold')
            ax.set_title(f'{phase_name_mapping[phase]} {metric.capitalize()} over Epochs for Hidden Layer Size = {FIXED_HIDDEN_SIZE}\n(Averaged across seeds)', 
                        fontsize=14, fontweight='bold')
            ax.legend(title='Learning Rate & Momentum', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(epochs)

            # Set y-axis to start from 0 for better visualization
            ax.set_ylim(bottom=0)

            plt.tight_layout()
            plt.savefig(os.path.join(FIGS_DIR, phase, metric, f'{FIXED_HIDDEN_SIZE}.png'), 
                        dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {os.path.join(FIGS_DIR, phase, metric, f'{FIXED_HIDDEN_SIZE}.png')}")
            print("\nData summary:")
            print(f"Hidden layer size: {FIXED_HIDDEN_SIZE}")
            print(f"Epochs: {epochs}")
            print(f"Learning rates: {learning_rates}")
            print(f"Momentum values: {momentums}")
            print(f"Total combinations: {len(combinations)}")

            # Print final training loss for each combination
            print(f"\nFinal {phase.capitalize()} {metric} (at last epoch) for each combination:")
            for combo in combinations:
                combo_data = grouped[grouped['combo'] == combo].sort_values('epoch')
                final_loss = combo_data.iloc[-1][metric]
                print(f"  {combo}: {final_loss:.4f}")

            # plt.show()