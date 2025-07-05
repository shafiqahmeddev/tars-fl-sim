from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from torch.utils.data import Dataset

class IClient(ABC):
    """Interface for a Federated Learning client."""
    @abstractmethod
    def train(self, global_model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Trains the local model and returns the updated state dict."""
        pass

class IServer(ABC):
    """Interface for the Central Server."""
    @abstractmethod
    def evaluate_model(self, model_state: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluates a model state on the validation set, returning (accuracy, loss)."""
        pass

class IAgent(ABC):
    """Interface for the decision-making agent (TARS)."""
    @abstractmethod
    def choose_action(self, state: tuple, current_round: int) -> int:
        """Chooses an aggregation rule based on the current state."""
        pass

    @abstractmethod
    def update_q_table(self, state: tuple, action: int, reward: float, next_state: tuple):
        """Updates the Q-table based on the outcome of an action."""
        pass

class IAttack(ABC):
    """Interface for a poisoning attack strategy."""
    @abstractmethod
    def apply(self, update: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Applies a malicious modification to a model update."""
        pass
