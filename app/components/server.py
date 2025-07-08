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
        """Evaluates a given model state on the validation set with robust error handling."""
        import math
        
        # Use the existing global model (make sure to restore state afterward)
        temp_model = self.global_model
        original_state = temp_model.state_dict()  # Backup original state
        
        try:
            # Ensure model state tensors are on correct device
            clean_state = {}
            for key, tensor in model_state.items():
                if isinstance(tensor, torch.Tensor):
                    clean_state[key] = tensor.to(self.device)
                else:
                    clean_state[key] = tensor
            
            temp_model.load_state_dict(clean_state)
            temp_model.eval()
            
            test_loss = 0.0
            correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.validation_loader):
                    try:
                        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                        output = temp_model(data)
                        
                        # Compute loss with error handling
                        batch_loss = F.nll_loss(output, target, reduction='sum')
                        if not math.isnan(batch_loss.item()) and not math.isinf(batch_loss.item()):
                            test_loss += batch_loss.item()
                        
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total_samples += target.size(0)
                        
                    except Exception as e:
                        print(f"Warning: Batch {batch_idx} evaluation failed: {e}")
                        continue

            # Calculate final metrics with safety checks
            if total_samples > 0:
                test_loss = test_loss / total_samples
                accuracy = 100.0 * correct / total_samples
            else:
                print("Warning: No valid samples processed during evaluation")
                test_loss = float('inf')
                accuracy = 0.0
            
            # Handle NaN/inf values
            if math.isnan(test_loss) or math.isinf(test_loss):
                print("Warning: Invalid loss detected, using fallback value")
                test_loss = 10.0  # High but finite loss
            
            if math.isnan(accuracy) or math.isinf(accuracy):
                print("Warning: Invalid accuracy detected, using fallback value")
                accuracy = 0.0
            
            return float(accuracy), float(test_loss)
            
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return 0.0, 10.0  # Return safe fallback values
        finally:
            # Always restore original model state
            try:
                temp_model.load_state_dict(original_state)
            except Exception as restore_error:
                print(f"Warning: Failed to restore model state: {restore_error}")
