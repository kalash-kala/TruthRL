import re
import matplotlib.pyplot as plt
import csv

log_file = 'train_qwen2_5_vl_3b_6gpu.log'

# Epoch level metrics
epoch_steps = []
epoch_accuracies = []
epoch_kls = []
epoch_pg_losses = []

# Validation metrics
val_steps = []
val_accuracies = []

# Step level metrics
step_inds = []
step_accuracies = []
step_kls = []
step_pg_losses = []

with open(log_file, 'r') as f:
    for line in f:
        # Check if line contains metric logging format
        if 'step:' in line:
            step_match = re.search(r'step:(\d+)', line)
            if not step_match:
                continue
            step_val = int(step_match.group(1))

            if 'epoch/' in line:
                # Training epoch metrics
                acc_match = re.search(r'epoch/training/accuracy:([0-9.]+)', line)
                kl_match = re.search(r'epoch/actor/ppo_kl:([0-9.]+)', line)
                loss_match = re.search(r'epoch/actor/pg_loss:([-\d.]+)', line)
                
                if acc_match:
                    epoch_steps.append(step_val)
                    epoch_accuracies.append(float(acc_match.group(1)))
                    epoch_kls.append(float(kl_match.group(1)) if kl_match else 0.0)
                    epoch_pg_losses.append(float(loss_match.group(1)) if loss_match else 0.0)
                
                # Validation metrics
                val_acc_match = re.search(r'epoch/val-aux/unknown/accuracy/mean@1:([0-9.]+)', line)
                if val_acc_match:
                    val_steps.append(step_val)
                    val_accuracies.append(float(val_acc_match.group(1)))
            
            elif 'actor/' in line:
                # Step-level noisy metrics
                acc_match = re.search(r'training/accuracy:([0-9.]+)', line)
                kl_match = re.search(r' - actor/ppo_kl:([0-9.]+)', line)
                loss_match = re.search(r' - actor/pg_loss:([-\d.]+)', line)
                
                if acc_match:
                    step_inds.append(step_val)
                    step_accuracies.append(float(acc_match.group(1)))
                    step_kls.append(float(kl_match.group(1)) if kl_match else 0.0)
                    step_pg_losses.append(float(loss_match.group(1)) if loss_match else 0.0)

# Plot
plt.figure(figsize=(12, 12))

# 1. Training Accuracy
plt.subplot(4, 1, 1)
plt.plot(step_inds, step_accuracies, color='skyblue', alpha=0.4, label='Step (Noisy)')
plt.plot(epoch_steps, epoch_accuracies, color='blue', marker='o', linewidth=2, label='Epoch Avg')
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# 2. Validation Accuracy
plt.subplot(4, 1, 2)
if val_steps:
    plt.plot(val_steps, val_accuracies, color='purple', marker='s', linewidth=2, label='Validation Accuracy')
    # Overlay training epoch acc for comparison
    plt.plot(epoch_steps, epoch_accuracies, color='blue', linestyle='--', alpha=0.3, label='Training baseline')
plt.title('Validation Accuracy (Test Set Generalization)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# 3. PPO KL
plt.subplot(4, 1, 3)
plt.plot(step_inds, step_kls, color='salmon', alpha=0.6, label='Step')
plt.plot(epoch_steps, epoch_kls, color='red', marker='o', linewidth=2, label='Epoch Avg')
plt.title('PPO KL Divergence (Stability)')
plt.ylabel('KL Divergence')
plt.grid(True)
plt.legend()

# 4. Policy Loss
plt.subplot(4, 1, 4)
plt.plot(step_inds, step_pg_losses, color='lightgreen', alpha=0.6, label='Step')
plt.plot(epoch_steps, epoch_pg_losses, color='green', marker='o', linewidth=2, label='Epoch Avg')
plt.title('Policy Gradient Loss (Convergence)')
plt.xlabel('Step')
plt.ylabel('PG Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics_plot.png')
print(f"Parsed {len(step_inds)} step logs, {len(epoch_steps)} epoch logs, and {len(val_steps)} validation checkpoints.")
if len(val_steps) > 0:
    print(f"Latest Validation Accuracy: {val_accuracies[-1]:.4f} at step {val_steps[-1]}")
