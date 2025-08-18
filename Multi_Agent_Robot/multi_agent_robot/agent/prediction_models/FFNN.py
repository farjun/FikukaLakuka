import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

from Multi_Agent_Robot.multi_agent_robot.data.data_utils import DataUtils
from Multi_Agent_Robot.multi_agent_robot.env.types import RobotActions

WEIGHTS_CACHE_DIR = Path(__file__).parent / "weights/"

class RockSampleAgent:
    def __init__(self, input_size, num_actions, lr=0.001):
        self.model = self.NNModel(input_size, num_actions)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        os.makedirs(WEIGHTS_CACHE_DIR, exist_ok=True)

    class NNModel(nn.Module):
        def __init__(self, input_size, num_actions):
            super(RockSampleAgent.NNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, num_actions)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def train(self, states, actions, num_epochs=10, batch_size=32):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        dataset = TensorDataset(states_tensor, actions_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        accuracy = self.evaluate(test_loader)
        print(f'Accuracy: {accuracy}%')

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def save_weights(self, file_path):
        with open(WEIGHTS_CACHE_DIR / file_path, 'wb') as f:
            pickle.dump(self.model.state_dict(), f)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path):
        with open(WEIGHTS_CACHE_DIR/file_path, 'rb') as f:
            self.model.load_state_dict(pickle.load(f))
        print(f"Model weights loaded from {file_path}")



def train_model():
    features_iter = DataUtils().get_state_action_features_iter()
    df = pd.concat(features_iter)
    agent_rock_beliefs = np.asarray(df['agent_rock_beliefs'].apply(lambda x: [float(p.split(":")[1]) for p in eval(x)]).tolist())
    cur_agent_locations = np.asarray(df['cur_agent_location'].apply(lambda x: eval(x)).tolist())
    states_np = np.hstack([agent_rock_beliefs, cur_agent_locations])

    actions_np = np.asarray(df['action'].apply(lambda x : list(iter(RobotActions)).index(eval(x)[0])).tolist())

    # Example usage
    # Assuming `states_np` is a numpy array of state features and `actions_np` is the corresponding action labels
    input_size = states_np.shape[1]
    num_actions = len(torch.unique(torch.Tensor(actions_np)))

    agent = RockSampleAgent(input_size, num_actions)
    agent.load_weights("agent_weights.pkl")
    agent.train(states_np, actions_np, num_epochs=10, batch_size=32)
    agent.save_weights("agent_weights.pkl")

if __name__ == '__main__':
    train_model()