\# Circle Separation Neural Network



\*\*Posted on October 10, 2025 | Programming Project\*\*



!\[Decision Boundary Evolution](images/animated.gif)



Excited to share a fun machine learning project I built using PyTorch! This neural network learns to separate points from two concentric circles—one inner and one outer—through binary classification. It's a great way to visualize how decision boundaries evolve during training.



\## Project Overview



The dataset consists of 200 points: 100 randomly generated inside a small circle (radius up to 1.5) labeled as 0, and 100 outside in a larger ring (radius 4 to 5) labeled as 1. The model trains over 300 epochs, and I captured the decision boundary at each step to see the learning process unfold.



\*\*Tech Stack:\*\* PyTorch (Neural Network \& Training), Matplotlib (Visualization), NumPy (Data Generation)



\## Model Architecture



The network is a simple feedforward model: a linear layer from 2 inputs (x, y coordinates) to 10 hidden units with ReLU activation, followed by another linear layer to 1 output with sigmoid for binary classification.



```python

class ClusterSplitter(nn.Module):

&nbsp;   def \_\_init\_\_(self):

&nbsp;       super().\_\_init\_\_()

&nbsp;       self.layer\_1 = nn.Linear(2, 10)

&nbsp;       self.layer\_2 = nn.Linear(10, 1)



&nbsp;   def forward(self, x):

&nbsp;       out = torch.relu(self.layer\_1(x))

&nbsp;       out = torch.sigmoid(self.layer\_2(out))

&nbsp;       return out

```



\## Training Process



Using Binary Cross-Entropy Loss and SGD optimizer with lr=0.1. Here's a snippet of the training loop, which also generates plots of the decision boundary:



```python

for epoch in range(epochs):

&nbsp;   y\_pred = model(x)

&nbsp;   optimizer.zero\_grad()

&nbsp;   loss = criterion(y\_pred, y)

&nbsp;   loss.backward()

&nbsp;   optimizer.step()

&nbsp;   plot\_decision\_boundary(model, x, y, epoch)

&nbsp;   if epoch % 30 == 0:

&nbsp;       print(f"completed {epoch} epochs with error of {loss}")

```



Sample training output (loss decreasing over epochs):

```

completed 0 epochs with error of 0.6931

completed 30 epochs with error of 0.5123

completed 60 epochs with error of 0.2345

... (continues to converge around epoch 150)

```



\## Key Insights



\- The decision boundary starts chaotic and gradually forms a clear separation between the circles.

\- Visualizing every epoch highlights the non-linear learning of the network.

\- This project reinforced my understanding of activation functions and loss landscapes in classification tasks.



\## Challenges \& Learnings



Initially, the model struggled with the circular separation due to the linear layers, but the hidden layer with ReLU allowed it to approximate the non-linear boundary. I experimented with learning rates—too high caused oscillations. Overall, it's a solid intro to PyTorch for geometric data tasks.



