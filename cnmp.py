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
