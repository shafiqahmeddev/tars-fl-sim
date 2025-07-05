import torch
from collections import OrderedDict
from typing import Dict, Any
from app.shared.interfaces import IAttack

class SignFlippingAttack(IAttack):
    """Flips the sign of every weight in the update."""
    def apply(self, update: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        malicious_update = OrderedDict()
        for key, value in update.items():
            malicious_update[key] = -1 * value
        return malicious_update

class GaussianAttack(IAttack):
    """Adds Gaussian noise to the update."""
    def __init__(self, std_dev: float):
        self.std_dev = std_dev

    def apply(self, update: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        malicious_update = OrderedDict()
        for key, value in update.items():
            noise = torch.randn_like(value) * self.std_dev
            malicious_update[key] = value + noise
        return malicious_update

class PretenseAttack(IAttack):
    """Acts benign for a number of rounds before launching a sub-attack."""
    def __init__(self, start_round: int, sub_attack: IAttack):
        self.start_round = start_round
        self.sub_attack = sub_attack

    def apply(self, update: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        current_round = kwargs.get('current_round', 0)
        if current_round < self.start_round:
            return update  # Behave honestly
        return self.sub_attack.apply(update, **kwargs)

# Note: Label-flipping is implemented by modifying the client's data loader,
# not as a direct update manipulation.
