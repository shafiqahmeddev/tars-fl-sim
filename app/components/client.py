import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, Any
from app.shared.interfaces import IClient, IAttack

class Client(IClient):
    """Emulates a Federated Learning client device."""
    def __init__(self, client_id: int, model: nn.Module, data_loader: DataLoader, device: str, is_byzantine: bool = False, attack: IAttack = None):
        self.client_id = client_id
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.is_byzantine = is_byzantine
        self.attack = attack

    def train(self, global_model_state: Dict[str, Any], local_epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """Performs local training and returns the new model state."""
        self.model.load_state_dict(global_model_state)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(local_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        update = self.model.state_dict()

        if self.is_byzantine and self.attack:
            return self.attack.apply(update, **kwargs)
        
        return update
