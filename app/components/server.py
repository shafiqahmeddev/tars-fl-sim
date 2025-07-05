import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, Any, Tuple
from app.shared.interfaces import IServer

class Server(IServer):
    """Emulates the central server in Federated Learning."""
    def __init__(self, global_model: nn.Module, validation_loader: DataLoader, device: str):
        self.global_model = global_model.to(device)
        self.validation_loader = validation_loader
        self.device = device

    def get_global_model_state(self) -> Dict[str, Any]:
        return self.global_model.state_dict()

    def set_global_model_state(self, new_state: Dict[str, Any]):
        self.global_model.load_state_dict(new_state)

    def evaluate_model(self, model_state: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluates a given model state on the validation set."""
        temp_model = self.global_model
        temp_model.load_state_dict(model_state)
        temp_model.eval()
        
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = temp_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.validation_loader.dataset)
        accuracy = 100. * correct / len(self.validation_loader.dataset)
        return accuracy, test_loss
