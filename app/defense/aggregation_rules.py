import torch
from collections import OrderedDict
from typing import List, Dict, Any

def _get_update_vectors(updates: List[Dict[str, Any]]) -> List[torch.Tensor]:
    """Helper to flatten model state dicts into vectors."""
    # Filter out non-tensor keys like '_client_stats'
    tensor_keys = [key for key in updates[0].keys() if key != '_client_stats' and isinstance(updates[0][key], torch.Tensor)]
    return [torch.cat([update[key].view(-1) for key in tensor_keys]) for update in updates]

def fed_avg(updates: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Standard Federated Averaging."""
    agg_update = OrderedDict()
    # Filter out non-tensor keys like '_client_stats'
    tensor_keys = [key for key in updates[0].keys() if key != '_client_stats' and isinstance(updates[0][key], torch.Tensor)]
    
    for key in tensor_keys:
        tensors = [update[key] for update in updates]
        # Convert to float if needed for mean operation
        if tensors[0].dtype in [torch.int64, torch.int32, torch.long]:
            tensors = [t.float() for t in tensors]
        agg_update[key] = torch.stack(tensors).mean(dim=0)
    return agg_update

def krum(updates: List[Dict[str, Any]], num_malicious: int, **kwargs) -> Dict[str, Any]:
    """Krum aggregation rule."""
    if not updates:
        return OrderedDict()
    
    update_vectors = _get_update_vectors(updates)
    num_clients = len(update_vectors)
    num_benign = num_clients - num_malicious
    
    distances = torch.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i, num_clients):
            dist = torch.norm(update_vectors[i] - update_vectors[j]) ** 2
            distances[i, j] = distances[j, i] = dist

    scores = torch.zeros(num_clients)
    for i in range(num_clients):
        sorted_dists, _ = torch.sort(distances[i])
        # Sum distances to the n-f-2 closest clients
        scores[i] = sorted_dists[1:num_benign-1].sum()
        
    best_client_idx = torch.argmin(scores).item()
    
    # Return only the tensor keys from the best client
    best_update = updates[best_client_idx]
    result = OrderedDict()
    tensor_keys = [key for key in best_update.keys() if key != '_client_stats' and isinstance(best_update[key], torch.Tensor)]
    for key in tensor_keys:
        result[key] = best_update[key]
    
    return result

def trimmed_mean(updates: List[Dict[str, Any]], trim_ratio: float, **kwargs) -> Dict[str, Any]:
    """Trimmed Mean aggregation rule."""
    num_to_trim = int(len(updates) * trim_ratio)
    agg_update = OrderedDict()
    
    # Filter out non-tensor keys like '_client_stats'
    tensor_keys = [key for key in updates[0].keys() if key != '_client_stats' and isinstance(updates[0][key], torch.Tensor)]
    
    for key in tensor_keys:
        tensors = [update[key] for update in updates]
        # Convert to float if needed for mean operation
        if tensors[0].dtype in [torch.int64, torch.int32, torch.long]:
            tensors = [t.float() for t in tensors]
        stacked_tensors = torch.stack(tensors)
        sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)
        
        trimmed_tensors = sorted_tensors[num_to_trim:-num_to_trim] if num_to_trim > 0 else sorted_tensors
        agg_update[key] = trimmed_tensors.mean(dim=0)
        
    return agg_update

def median(updates: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Element-wise Median aggregation rule."""
    agg_update = OrderedDict()
    # Filter out non-tensor keys like '_client_stats'
    tensor_keys = [key for key in updates[0].keys() if key != '_client_stats' and isinstance(updates[0][key], torch.Tensor)]
    
    for key in tensor_keys:
        tensors = [update[key] for update in updates]
        # Convert to float if needed for median operation
        if tensors[0].dtype in [torch.int64, torch.int32, torch.long]:
            tensors = [t.float() for t in tensors]
        stacked_tensors = torch.stack(tensors)
        agg_update[key], _ = torch.median(stacked_tensors, dim=0)
    return agg_update

def fl_trust(updates: List[Dict[str, Any]], server_update: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """FLTrust aggregation rule with enhanced device and dtype consistency."""
    if not updates or not server_update:
        return OrderedDict()
    
    # Filter out non-tensor keys like '_client_stats'
    tensor_keys = [key for key in server_update.keys() if key != '_client_stats' and isinstance(server_update[key], torch.Tensor)]
    
    if not tensor_keys:
        return OrderedDict()
    
    # Detect device from first tensor (prefer GPU if available)
    reference_device = None
    for key in tensor_keys:
        if server_update[key].is_cuda:
            reference_device = server_update[key].device
            break
    if reference_device is None:
        reference_device = server_update[tensor_keys[0]].device
    
    # Convert server tensors to float32 on reference device for consistent computation
    server_tensors = []
    for key in tensor_keys:
        tensor = server_update[key].to(device=reference_device, dtype=torch.float32)
        server_tensors.append(tensor.view(-1))
    
    server_vec = torch.cat(server_tensors)
    server_norm = torch.norm(server_vec)
    
    # Calculate trust scores with device consistency
    trust_scores = []
    for update in updates:
        # Convert client tensors to float32 on reference device
        client_tensors = []
        for key in tensor_keys:
            tensor = update[key].to(device=reference_device, dtype=torch.float32)
            client_tensors.append(tensor.view(-1))
        
        client_vec = torch.cat(client_tensors)
        # Cosine similarity as trust score
        score = torch.relu(torch.nn.functional.cosine_similarity(server_vec, client_vec, dim=0))
        trust_scores.append(score)
    
    # Ensure trust scores are on reference device
    trust_scores = torch.stack(trust_scores).to(device=reference_device)
    normalized_scores = trust_scores / (trust_scores.sum() + 1e-9)
    
    # Perform weighted aggregation with device consistency
    agg_update = OrderedDict()
    for key in tensor_keys:
        # Store original properties for final conversion
        original_dtype = server_update[key].dtype
        original_device = server_update[key].device
        
        # Initialize weighted_sum as float32 on reference device
        weighted_sum = torch.zeros_like(
            server_update[key], 
            dtype=torch.float32, 
            device=reference_device
        )
        
        # Accumulate weighted updates
        for i, update in enumerate(updates):
            update_tensor = update[key].to(device=reference_device, dtype=torch.float32)
            score = normalized_scores[i]
            weighted_sum += update_tensor * score
        
        # Convert back to original dtype and device
        if original_dtype in [torch.int64, torch.int32, torch.long]:
            # Round for integer types
            weighted_sum = weighted_sum.round()
        
        # Move to original device and convert to original dtype
        agg_update[key] = weighted_sum.to(device=original_device, dtype=original_dtype)
        
    return agg_update
