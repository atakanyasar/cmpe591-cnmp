import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# load states.npy
states = np.load('states.npy', allow_pickle=True)

dataset = []
heights = []

for episode in states:  # shape: (100, 5)
    height = episode[0, 4]  # grab h from first timestep
    heights.append(height)

    for t_idx, step in enumerate(episode):
        ey, ez, oy, oz, h = step  # h will be same as episode height
        dataset.append({
            "t": t_idx / 100,  # normalize time to [0, 1]
            "ey": ey,
            "ez": ez,
            "oy": oy,
            "oz": oz,
            "h": h
        })

heights = np.array(heights)

class CNMP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),  # (t, h)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # output: (ey, ez, oy, oz)
        )

    def forward(self, t, h):
        x = torch.cat([t, h], dim=1)  # shape: (batch, 2)
        return self.model(x)  # shape: (batch, 4)
    

def train_model(model, dataset, epochs=500, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    data = [
        (
            torch.tensor([[d['t']]], dtype=torch.float32),
            torch.tensor([[d['h']]], dtype=torch.float32),
            torch.tensor([[d['ey'], d['ez'], d['oy'], d['oz']]], dtype=torch.float32)
        )
        for d in dataset
    ]

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            t_batch = torch.cat([b[0] for b in batch], dim=0)
            h_batch = torch.cat([b[1] for b in batch], dim=0)
            y_batch = torch.cat([b[2] for b in batch], dim=0)

            pred = model(t_batch, h_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

def evaluate_model(model, dataset, n_tests=100):
    ef_errors = []
    obj_errors = []

    for _ in range(n_tests):
        traj = np.random.choice(dataset, 1)[0]
        height = traj["h"]

        # Sample context and query sizes
        context_size = np.random.randint(1, 10)
        query_size = np.random.randint(1, 10)

        # Sample queries
        query_points = np.random.choice(dataset, query_size)

        for point in query_points:
            t = torch.tensor([[point["t"]]], dtype=torch.float32)
            h = torch.tensor([[point["h"]]], dtype=torch.float32)
            y_true = np.array([point["ey"], point["ez"], point["oy"], point["oz"]])

            with torch.no_grad():
                pred = model(t, h).numpy().squeeze()

            ef_pred = pred[:2]
            ef_true = y_true[:2]
            obj_pred = pred[2:]
            obj_true = y_true[2:]

            ef_errors.append(np.mean((ef_pred - ef_true) ** 2))
            obj_errors.append(np.mean((obj_pred - obj_true) ** 2))

    return np.mean(ef_errors), np.std(ef_errors), np.mean(obj_errors), np.std(obj_errors)


def plot_random_trajectories(model, num_trajectories=5, num_points=100, h_range=(0.03, 0.1), seed=None):
    """
    Generate and plot multiple random trajectories for randomly sampled object heights.

    Args:
        model: Trained CNMP model.
        num_trajectories: Number of different random h values (trajectories).
        num_points: Number of time steps per trajectory.
        h_range: Tuple (min, max) for sampling h.
        seed: Optional random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    model.eval()
    t_values = np.linspace(0, 1, num_points).reshape(-1, 1)
    t_tensor = torch.tensor(t_values, dtype=torch.float32)

    plt.figure(figsize=(12, 5))

    # End-effector plot
    plt.subplot(1, 2, 1)
    for _ in range(num_trajectories):
        h_val = np.random.uniform(*h_range)
        h_tensor = torch.full((num_points, 1), h_val, dtype=torch.float32)
        with torch.no_grad():
            pred = model(t_tensor, h_tensor).numpy()
        ey, ez = pred[:, 0], pred[:, 1]
        plt.plot(ey, ez, label=f"h={h_val:.2f}")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("End-Effector Trajectories")
    plt.grid(True)
    plt.legend()

    # Object plot
    plt.subplot(1, 2, 2)
    for _ in range(num_trajectories):
        h_val = np.random.uniform(*h_range)
        h_tensor = torch.full((num_points, 1), h_val, dtype=torch.float32)
        with torch.no_grad():
            pred = model(t_tensor, h_tensor).numpy()
        oy, oz = pred[:, 2], pred[:, 3]
        plt.plot(oy, oz, label=f"h={h_val:.2f}")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("Object Trajectories")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('cnmp_trajectories.png')
    plt.show()


def plot_errors(ef_mean, ef_std, obj_mean, obj_std):
    labels = ['End-Effector', 'Object']
    means = [ef_mean, obj_mean]
    stds = [ef_std, obj_std]

    plt.bar(labels, means, yerr=stds, capsize=10)
    plt.ylabel('MSE')
    plt.title('Prediction Error (CNMP)')
    plt.savefig('cnmp_errors.png')
    plt.show()


model = CNMP()
train_model(model, dataset)
ef_mean, ef_std, obj_mean, obj_std = evaluate_model(model, dataset)
plot_errors(ef_mean, ef_std, obj_mean, obj_std)

# Save the model
torch.save(model.state_dict(), 'cnmp_model.pth')

# Generate a sample trajectory
plot_random_trajectories(model, num_trajectories=5, num_points=100, seed=1)
