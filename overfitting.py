import os
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = 'logs'
FIGS_DIR = 'figures/overfitting'

os.makedirs(FIGS_DIR, exist_ok=True)

# Load the CSV file
df = pd.read_csv(os.path.join(LOGS_DIR, 'overfitting.csv'), delimiter=';')

# Set the parameters to filter by
HIDDEN_LAYER_SIZES = df['hidden_layer_size'].unique()
SEEDS = df['seed'].unique()

for HIDDEN_LAYER_SIZE in HIDDEN_LAYER_SIZES:
    for SEED in SEEDS:

        # Filter the data
        filtered_df = df[(df['hidden_layer_size'] == HIDDEN_LAYER_SIZE) & 
                        (df['seed'] == SEED)]

        # Sort by epoch to ensure proper plotting
        filtered_df = filtered_df.sort_values('epoch')

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['epoch'], filtered_df['loss'], marker='o', linewidth=2, markersize=8)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Loss vs Epoch\n(hidden_layer_size={HIDDEN_LAYER_SIZE}, seed={SEED})', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add some styling
        plt.tight_layout()

        # Show the plot
        # plt.show()

        # Save the plot
        fig_filename = f'loss_epoch_h{HIDDEN_LAYER_SIZE}_s{SEED}.png'
        plt.savefig(os.path.join(FIGS_DIR, fig_filename))

        # Print summary statistics
        print(f"\nSummary for hidden_layer_size={HIDDEN_LAYER_SIZE}, seed={SEED}:")
        print(f"Number of epochs: {len(filtered_df)}")
        print(f"Epoch range: {filtered_df['epoch'].min()} to {filtered_df['epoch'].max()}")
        print(f"Loss range: {filtered_df['loss'].min():.6f} to {filtered_df['loss'].max():.6f}")
        print(f"Final loss: {filtered_df['loss'].iloc[-1]:.6f}")