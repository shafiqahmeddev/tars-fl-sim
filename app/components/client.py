import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, Any
from app.shared.interfaces import IClient, IAttack
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

class Client(IClient):
    """Enhanced Federated Learning client with advanced training techniques and device consistency."""
    def __init__(self, client_id: int, model: nn.Module, data_loader: DataLoader, device: str, 
                 is_byzantine: bool = False, attack: IAttack = None, config: Dict[str, Any] = None):
        self.client_id = client_id
        
        # Ensure device consistency
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU for client {client_id}")
            device = 'cpu'
        
        self.device = device
        self.model = model.to(device)
        self.data_loader = data_loader
        self.is_byzantine = is_byzantine
        self.attack = attack
        self.config = config or {}
        
        # Validate device configuration
        self._validate_device_setup()

    def _validate_device_setup(self):
        """Validate device configuration and model placement."""
        try:
            # Check if model is on correct device
            model_device = next(self.model.parameters()).device
            expected_device = torch.device(self.device)
            
            # Compare device objects instead of strings to handle cuda:0 vs cuda
            if model_device != expected_device:
                print(f"Info: Moving model from {model_device} to {expected_device}")
                self.model = self.model.to(self.device)
            else:
                print(f"✅ Client {self.client_id} model correctly placed on {model_device}")
        except Exception as e:
            print(f"Device validation failed for client {self.client_id}: {e}")

    def train(self, global_model_state: Dict[str, Any], local_epochs: int = None, **kwargs) -> Dict[str, Any]:
        """Performs enhanced local training with improved optimization and device consistency."""
        # Ensure model state is loaded on correct device
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        # Ensure model is on correct device after loading state
        self.model = self.model.to(self.device)
        
        # Get training parameters from config
        if local_epochs is None:
            local_epochs = self.config.get('local_epochs', 1)
        
        lr = self.config.get('client_lr', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        optimizer_type = self.config.get('client_optimizer', 'adam')
        
        # Create optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = None
        if self.config.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
        
        # Enhanced mixed precision training setup with device validation
        use_amp = self.config.get('use_amp', False) and AMP_AVAILABLE and self.device == 'cuda'
        scaler = None
        if use_amp:
            try:
                scaler = GradScaler('cuda')
            except Exception as e:
                print(f"Warning: Failed to initialize GradScaler for client {self.client_id}: {e}")
                use_amp = False
        
        # Training loop with enhanced features
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Ensure data is on correct device
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if use_amp and scaler is not None:
                    with autocast('cuda'):
                        output = self.model(data)
                        loss = F.nll_loss(output, target)
                        
                        # Add L2 regularization if not using weight_decay
                        if self.config.get('l2_reg', 0.0) > 0:
                            l2_reg = torch.tensor(0.).to(self.device)
                            for param in self.model.parameters():
                                l2_reg += torch.norm(param)
                            loss += self.config['l2_reg'] * l2_reg
                    
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping with scaler
                    if self.config.get('grad_clip', 0.0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    
                    # Add L2 regularization if not using weight_decay
                    if self.config.get('l2_reg', 0.0) > 0:
                        l2_reg = torch.tensor(0.).to(self.device)
                        for param in self.model.parameters():
                            l2_reg += torch.norm(param)
                        loss += self.config['l2_reg'] * l2_reg
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.get('grad_clip', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    
                    optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
            
            # Update learning rate (FIXED: scheduler step after optimizer step)
            if scheduler is not None:
                scheduler.step()
        
        # Calculate average loss for this client
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get model state dict and ensure device consistency
        update = self.model.state_dict()
        
        # Ensure all tensors in update are on the same device
        reference_device = self.device
        for key, tensor in update.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != torch.device(reference_device):
                update[key] = tensor.to(reference_device)
        
        # Store training statistics
        update['_client_stats'] = {
            'client_id': self.client_id,
            'avg_loss': avg_loss,
            'num_samples': len(self.data_loader.dataset),
            'local_epochs': local_epochs,
            'device': str(reference_device)
        }

        if self.is_byzantine and self.attack:
            # Apply attack but preserve stats
            stats = update.pop('_client_stats')
            attacked_update = self.attack.apply(update, **kwargs)
            attacked_update['_client_stats'] = stats
            return attacked_update
        
        return update
