import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, Any
from app.shared.interfaces import IClient, IAttack
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

class Client(IClient):
    """Enhanced Federated Learning client with advanced training techniques."""
    def __init__(self, client_id: int, model: nn.Module, data_loader: DataLoader, device: str, 
                 is_byzantine: bool = False, attack: IAttack = None, config: Dict[str, Any] = None):
        self.client_id = client_id
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.is_byzantine = is_byzantine
        self.attack = attack
        self.config = config or {}

    def train(self, global_model_state: Dict[str, Any], local_epochs: int = None, **kwargs) -> Dict[str, Any]:
        """Performs enhanced local training with improved optimization."""
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
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
        if self.config.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
        
        # Mixed precision training setup
        use_amp = self.config.get('use_amp', False) and AMP_AVAILABLE and self.device == 'cuda'
        scaler = GradScaler() if use_amp else None
        
        # Training loop with enhanced features
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if use_amp:
                    with autocast():
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
            
            # Update learning rate
            if self.config.get('use_scheduler', False):
                scheduler.step()
        
        # Calculate average loss for this client
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        update = self.model.state_dict()
        
        # Store training statistics
        update['_client_stats'] = {
            'client_id': self.client_id,
            'avg_loss': avg_loss,
            'num_samples': len(self.data_loader.dataset),
            'local_epochs': local_epochs
        }

        if self.is_byzantine and self.attack:
            # Apply attack but preserve stats
            stats = update.pop('_client_stats')
            attacked_update = self.attack.apply(update, **kwargs)
            attacked_update['_client_stats'] = stats
            return attacked_update
        
        return update
