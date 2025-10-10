import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

class ClusterSplitter(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(2, 10)

        self.layer_2 = nn.Linear(10, 1)

    def forward(self, x):
        out = torch.relu(self.layer_1(x))
        out = torch.sigmoid(self.layer_2(out))
        return out

#Provided by an AI system for visualisation of the task
def plot_decision_boundary(model, x, y, epoch):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    x_axis = np.arange(x_min, x_max, 0.1)
    y_axis = np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x_axis, y_axis)

    grid_points_np = np.c_[xx.ravel(), yy.ravel()]
    grid_points_torch = torch.from_numpy(grid_points_np).float()

    with torch.no_grad():
        predictions = model(grid_points_torch)

    predictions_grid = predictions.view(xx.shape)
    plt.contourf(xx, yy, predictions_grid.detach().numpy(), cmap='coolwarm', alpha=0.5)
    plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c=y.detach().numpy(), cmap='coolwarm')
    #plt.show()
    plt.savefig(f"images/epoch-{epoch}.png")

angles = torch.rand(100, 1) * 2 * math.pi

radii = torch.rand(100, 1) + 4

x_coordinates = radii * torch.cos(angles)

y_coordinates = radii * torch.sin(angles)

outer_circle = torch.cat((x_coordinates, y_coordinates), dim = 1)

angles = torch.rand(100, 1) * 2 * math.pi

radii = torch.rand(100, 1) * 1.5

x_coordinates = radii * torch.cos(angles)

y_coordinates = radii * torch.sin(angles)

inner_circle = torch.cat((x_coordinates, y_coordinates), dim = 1)

x = torch.cat((inner_circle, outer_circle))

y = torch.cat((torch.zeros(100, 1), torch.ones(100, 1)))

epochs = 300

criterion = nn.BCELoss()

model = ClusterSplitter()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(epochs):
    y_pred = model(x)

    optimizer.zero_grad()

    loss = criterion(y_pred, y)

    loss.backward()

    optimizer.step()

    plot_decision_boundary(model, x, y, epoch)

    if epoch % 30 == 0:
        print(f"completed {epoch} epochs with error of {loss}")

