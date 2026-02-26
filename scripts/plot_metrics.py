import re
import csv
import argparse
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from log file.')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to the log file')
    parser.add_argument('--output', type=str, 
                        default='training_metrics_plot.png',
                        help='Name of the output plot file')
    args = parser.parse_args()

    log_file = args.log_file
    output_filename = args.output

    # Define the plots directory within TruthRL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    truth_rl_dir = os.path.dirname(script_dir)
    plots_dir = os.path.join(truth_rl_dir, 'plots')
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, output_filename)

    # Step level metrics
    step_inds = []
    step_accuracies = []
    step_kls = []
    step_pg_losses = []

    # Validation metrics
    val_steps = []
    val_accuracies = []

    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    # Regex patterns for the current log format (handling np.float64 and step: prefix)
    step_pattern = r'step:(\d+)'
    # Use a non-capturing group for optional np.float64 wrapper
    num_pattern = r'(?:np\.float64\()?([-\d.e+]+)\)?'
    
    acc_regex = re.compile(r'training/accuracy:' + num_pattern)
    kl_regex = re.compile(r'actor/ppo_kl:' + num_pattern)
    loss_regex = re.compile(r'actor/pg_loss:' + num_pattern)
    val_acc_regex = re.compile(r'val-aux/unknown/accuracy/mean@1:' + num_pattern)

    with open(log_file, 'r') as f:
        for line in f:
            # Check if line contains a step indicator
            step_match = re.search(step_pattern, line)
            if not step_match:
                continue
            
            step_val = int(step_match.group(1))

            # Extract Training Metrics
            acc_m = acc_regex.search(line)
            kl_m = kl_regex.search(line)
            loss_m = loss_regex.search(line)
            
            if acc_m:
                step_inds.append(step_val)
                step_accuracies.append(float(acc_m.group(1)))
                step_kls.append(float(kl_m.group(1)) if kl_m else 0.0)
                step_pg_losses.append(float(loss_m.group(1)) if loss_m else 0.0)
            
            # Extract Validation Metrics
            val_acc_m = val_acc_regex.search(line)
            if val_acc_m:
                val_steps.append(step_val)
                val_accuracies.append(float(val_acc_m.group(1)))

    if not step_inds and not val_steps:
        print(f"Warning: No metrics found in {log_file}. Check the regex patterns and log format.")
        return

    # Plot
    plt.figure(figsize=(12, 12))

    # 1. Training Accuracy
    plt.subplot(4, 1, 1)
    if step_inds:
        plt.plot(step_inds, step_accuracies, color='skyblue', alpha=0.6, label='Step')
        # Add a smoothed version if there are enough points
        if len(step_accuracies) > 10:
            window = 10
            smoothed = [sum(step_accuracies[i:i+window])/window for i in range(len(step_accuracies)-window+1)]
            plt.plot(step_inds[window-1:], smoothed, color='blue', linewidth=2, label='Smoothed')
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # 2. Validation Accuracy
    plt.subplot(4, 1, 2)
    if val_steps:
        plt.plot(val_steps, val_accuracies, color='purple', marker='s', linewidth=2, label='Validation Accuracy')
    plt.title('Validation Accuracy (Test Set Generalization)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # 3. PPO KL
    plt.subplot(4, 1, 3)
    if step_inds:
        plt.plot(step_inds, step_kls, color='salmon', alpha=0.6, label='Step')
    plt.title('PPO KL Divergence (Stability)')
    plt.ylabel('KL Divergence')
    plt.grid(True)
    plt.legend()

    # 4. Policy Loss
    plt.subplot(4, 1, 4)
    if step_inds:
        plt.plot(step_inds, step_pg_losses, color='lightgreen', alpha=0.6, label='Step')
    plt.title('Policy Gradient Loss (Convergence)')
    plt.xlabel('Step')
    plt.ylabel('PG Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    print(f"Parsed {len(step_inds)} step logs and {len(val_steps)} validation checkpoints.")
    if val_steps:
        print(f"Latest Validation Accuracy: {val_accuracies[-1]:.4f} at step {val_steps[-1]}")

if __name__ == "__main__":
    main()
