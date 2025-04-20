import numpy as np
import matplotlib.pyplot as plt
import torch
from homework4 import CNP

# Load and preprocess data
states = np.load('assets/states.npy', allow_pickle=True)
dataset = []

for episode in states:
    for t_idx, step in enumerate(episode):
        t = t_idx / (len(episode) - 1) if len(episode) > 1 else 0.0  # Normalize time to [0,1]
        ey, ez, oy, oz, h = step
        dataset.append({
            't': t,
            'ey': ey,
            'ez': ez,
            'oy': oy,
            'oz': oz,
            'h': h
        })

# Convert to tensors
data_t = torch.tensor([d['t'] for d in dataset]).float().unsqueeze(1)
data_h = torch.tensor([d['h'] for d in dataset]).float().unsqueeze(1)
data_y = torch.tensor([[d['ey'], d['ez'], d['oy'], d['oz']] for d in dataset]).float()

# Create CNMP model
class CNMPWrapper(CNP):
    def __init__(self):
        super().__init__(in_shape=(2, 4),  # (t,h) input, (ey,ez,oy,oz) output
                         hidden_size=128,
                         num_hidden_layers=3,
                         min_std=0.1)

    def prepare_inputs(self, t, h, y=None):
        # Combine t and h for query/observation x
        x = torch.cat([t, h], dim=-1)
        return x, y

    def forward(self, observation_x, observation_y, target_x, observation_mask=None):
        # Prepare observation input for the base CNP
        observation = torch.cat([observation_x, observation_y], dim=-1)
        return super().forward(observation, target_x, observation_mask)

model = CNMPWrapper()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training function with improved batching
def train(model, data_t, data_h, data_y, epochs=1000, batch_size=64):
    losses = []
    n_data = len(data_t)
    for epoch in range(epochs):
        permutation = torch.randperm(n_data)
        for i in range(0, n_data, batch_size):
            indices = permutation[i:i + batch_size]
            batch_t = data_t[indices]
            batch_h = data_h[indices]
            batch_y = data_y[indices]

            # Randomly split into context and target
            n_context = torch.randint(10, batch_size - 10, (1,)).item() if batch_size > 20 else 10

            # Prepare inputs - x is (t,h), y is (ey,ez,oy,oz)
            context_x = torch.cat([batch_t[:n_context], batch_h[:n_context]], dim=-1)
            context_y = batch_y[:n_context]

            target_x = torch.cat([batch_t[n_context:], batch_h[n_context:]], dim=-1)
            target_y = batch_y[n_context:]

            # Forward pass
            mean, std = model(context_x.unsqueeze(0),  # Add batch dimension
                                context_y.unsqueeze(0),
                                target_x.unsqueeze(0))

            # Compute loss
            if target_y.numel() > 0:
                dist = torch.distributions.Normal(mean, std)
                loss = -dist.log_prob(target_y.unsqueeze(0)).mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {np.mean(losses[-len(indices):]):.4f}")

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('assets/training_loss.png')
    plt.show()
    return losses

# Improved evaluation function
def evaluate(model, dataset, n_tests=100):
    ef_errors = []
    obj_errors = []
    n_dataset = len(dataset)

    for _ in range(n_tests):
        # Randomly select context and target sizes
        n_context = np.random.randint(10, n_dataset // 2)
        n_target = np.random.randint(10, n_dataset - n_context)

        # Randomly sample points
        idx = np.random.choice(n_dataset, n_context + n_target, replace=False)
        context_idx = idx[:n_context]
        target_idx = idx[n_context:]

        # Prepare context inputs
        context_t = torch.tensor([dataset[i]['t'] for i in context_idx]).float().unsqueeze(1)
        context_h = torch.tensor([dataset[i]['h'] for i in context_idx]).float().unsqueeze(1)
        context_y = torch.tensor([[dataset[i]['ey'], dataset[i]['ez'],
                                    dataset[i]['oy'], dataset[i]['oz']]
                                   for i in context_idx]).float()

        # Prepare target inputs
        target_t = torch.tensor([dataset[i]['t'] for i in target_idx]).float().unsqueeze(1)
        target_h = torch.tensor([dataset[i]['h'] for i in target_idx]).float().unsqueeze(1)
        target_y = torch.tensor([[dataset[i]['ey'], dataset[i]['ez'],
                                    dataset[i]['oy'], dataset[i]['oz']]
                                   for i in target_idx]).float()

        # Forward pass
        with torch.no_grad():
            context_x = torch.cat([context_t, context_h], dim=-1)
            target_x = torch.cat([target_t, target_h], dim=-1)

            mean, _ = model(context_x.unsqueeze(0),
                                context_y.unsqueeze(0),
                                target_x.unsqueeze(0))

            # Calculate errors
            ef_error = ((mean[0, :, :2] - target_y[:, :2]) ** 2).mean().item()
            obj_error = ((mean[0, :, 2:] - target_y[:, 2:]) ** 2).mean().item()

            ef_errors.append(ef_error)
            obj_errors.append(obj_error)

    return np.mean(ef_errors), np.std(ef_errors), np.mean(obj_errors), np.std(obj_errors)

# Improved trajectory plotting
def plot_cnmp_trajectories(model, dataset, num_trajectories=5, num_points=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    model.eval()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_ee = axes[0]
    ax_obj = axes[1]

    # Get unique heights from dataset
    unique_heights = sorted(list({d['h'] for d in dataset}))
    selected_heights = np.random.choice(unique_heights, min(num_trajectories, len(unique_heights)), replace=False)

    # End-effector plot
    ax_ee.set_xlabel("e_y")
    ax_ee.set_ylabel("e_z")
    ax_ee.set_title("Predicted End-Effector Trajectories")
    ax_ee.grid(True)

    # Object plot
    ax_obj.set_xlabel("o_y")
    ax_obj.set_ylabel("o_z")
    ax_obj.set_title("Predicted Object Trajectories")
    ax_obj.grid(True)

    for h_val in selected_heights:
        # Filter dataset for this height
        h_data = [d for d in dataset if abs(d['h'] - h_val) < 1e-6]
        if len(h_data) < 10:  # Skip if not enough data
            continue

        # Create time points
        t_values = np.linspace(0, 1, num_points)

        # Create context points (actual data points)
        context_idx = np.random.choice(len(h_data), min(20, len(h_data)), replace=False)
        context_t = torch.tensor([h_data[i]['t'] for i in context_idx]).float().unsqueeze(1)
        context_h = torch.tensor([h_data[i]['h'] for i in context_idx]).float().unsqueeze(1)
        context_y = torch.tensor([[h_data[i]['ey'], h_data[i]['ez'],
                                    h_data[i]['oy'], h_data[i]['oz']]
                                   for i in context_idx]).float()

        # Create target points (evenly spaced in time)
        target_t = torch.tensor(t_values).float().unsqueeze(1)
        target_h = torch.full((num_points, 1), h_val, dtype=torch.float32)

        # Forward pass
        with torch.no_grad():
            context_x = torch.cat([context_t, context_h], dim=-1)
            target_x = torch.cat([target_t, target_h], dim=-1)

            mean, _ = model(context_x.unsqueeze(0),
                                context_y.unsqueeze(0),
                                target_x.unsqueeze(0))

            ey, ez = mean[0, :, 0].numpy(), mean[0, :, 1].numpy()
            ax_ee.plot(ey, ez, alpha=0.5, label=f"h={h_val:.3f}")

            oy, oz = mean[0, :, 2].numpy(), mean[0, :, 3].numpy()
            ax_obj.plot(oy, oz, alpha=0.5, label=f"h={h_val:.3f}")

    ax_ee.legend()
    ax_obj.legend()
    fig.tight_layout()
    plt.savefig('assets/cnmp_predicted_trajectories.png')
    plt.show()

# Plot error bars
def plot_errors(ef_mean, ef_std, obj_mean, obj_std):
    labels = ['End-Effector', 'Object']
    means = [ef_mean, obj_mean]
    stds = [ef_std, obj_std]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means, yerr=stds, capsize=10, alpha=0.7)
    plt.ylabel('Mean Squared Error')
    plt.title('Prediction Errors')
    plt.grid(True, axis='y')
    plt.savefig('assets/cnmp_errors.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Train the model
    train_losses = train(model, data_t, data_h, data_y, epochs=50)

    # Evaluate
    ef_mean, ef_std, obj_mean, obj_std = evaluate(model, dataset)
    print(f"End-effector MSE: {ef_mean:.4f} ± {ef_std:.4f}")
    print(f"Object MSE: {obj_mean:.4f} ± {obj_std:.4f}")

    # Plot results
    plot_errors(ef_mean, ef_std, obj_mean, obj_std)
    plot_cnmp_trajectories(model, dataset, num_trajectories=5)

    # Save model
    torch.save(model.state_dict(), 'assets/cnmp_model.pth')