import numpy as np
import torch


class Network(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class Classifier:
    def __init__(self, hidden_size=10, epochs=1000, batch_size=10, random_state=42):
        self.model = None
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.model = Network(input_size=X.shape[1], hidden_size=self.hidden_size, output_size=np.max(y) + 1)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long)), dtype=torch.float))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(self.epochs):
            for X, y in dataloader:
                pred = self.model(X)
                loss = torch.nn.MSELoss()(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, X):
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float)).argmax(dim=1).numpy()
