# TARS Training on Google Colab

This guide explains how to train your TARS federated learning system on Google Colab for optimal performance with free GPU access.

## 🚀 Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload Notebook**: Upload the `TARS_Training_Colab.ipynb` file
3. **Enable GPU**: Go to Runtime → Change runtime type → Hardware accelerator → GPU
4. **Run All Cells**: Execute cells sequentially to train the model

## 📊 Expected Results

### MNIST Training
- **Target**: 97.7% accuracy
- **Training Time**: ~15-20 minutes on GPU
- **Expected Performance**: 97%+ accuracy with enhanced CNN architecture

### CIFAR-10 Training  
- **Target**: 80.5% accuracy
- **Training Time**: ~25-30 minutes on GPU
- **Expected Performance**: 80%+ accuracy with ResNet-inspired architecture

## 🔧 Configuration Options

### Quick Test (Faster Results)
```python
config = {
    "num_rounds": 10,  # Reduced rounds for quick test
    "local_epochs": 1,  # Single epoch per round
    "num_clients": 5,   # Fewer clients
}
```

### Full Training (Best Accuracy)
```python
config = {
    "num_rounds": 50,   # Full training
    "local_epochs": 3,  # Multiple epochs
    "num_clients": 10,  # Standard setup
}
```

## 🎯 Performance Monitoring

The notebook provides real-time monitoring:
- Round-by-round accuracy and loss
- TARS aggregation rule selection
- Trust scores and Byzantine attack detection
- Performance visualization and analysis

## 💾 Model Persistence

Trained models are automatically saved:
- **Global Model**: `checkpoints/mnist_global_model.pth`
- **TARS Agent**: `checkpoints/mnist_tars_agent.pkl`
- **Training History**: `mnist_training_results.csv`

## 🔄 Advantages of Colab Training

### ✅ Benefits
- **Free GPU Access**: Tesla T4/K80 GPUs available
- **No Setup Required**: Pre-installed PyTorch and dependencies
- **Easy Sharing**: Share notebooks with collaborators
- **Automatic Checkpointing**: Models saved to Google Drive
- **Visualization**: Built-in matplotlib and plotting

### ⚠️ Considerations
- **Runtime Limits**: 12-hour maximum sessions
- **Memory Limits**: ~12GB RAM available
- **Storage**: Limited disk space (need to download models)

## 🛠️ Troubleshooting

### Common Issues

**GPU Not Available**
```python
# Check GPU status
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```
*Solution*: Go to Runtime → Change runtime type → GPU

**Out of Memory**
```python
# Reduce batch size or clients
config["batch_size"] = 16
config["num_clients"] = 5
```

**Session Timeout**
```python
# Enable model checkpointing
config["save_model"] = True
config["use_pretrained"] = True
```

## 📈 Performance Optimization

### For Maximum Accuracy
1. **Use GPU**: Ensure GPU runtime is enabled
2. **Full Configuration**: Use 50 rounds, 3 local epochs
3. **Data Augmentation**: Enabled by default for better generalization
4. **Early Stopping**: Prevents overfitting
5. **Trust Mechanism**: TARS automatically selects best aggregation rules

### For Faster Training
1. **Reduce Rounds**: Set `num_rounds = 20`
2. **Single Epoch**: Set `local_epochs = 1`
3. **Fewer Clients**: Set `num_clients = 5`
4. **IID Data**: Set `is_iid = True`

## 🎉 Expected Console Output

```
🚀 Starting MNIST Training - Target: 97.7% accuracy
========================================
Using device: cuda
Files already downloaded and verified
📝 No pre-trained MNIST model found. Will train from scratch.

--- Round 1/50 ---
TARS chose rule: fed_avg
Round 1 Accuracy: 85.23%, Loss: 0.4821, Avg Trust: 0.847
🎯 New best accuracy: 85.23%

--- Round 2/50 ---
TARS chose rule: krum
Round 2 Accuracy: 91.45%, Loss: 0.2956, Avg Trust: 0.892
🎯 New best accuracy: 91.45%

...

--- Round 42/50 ---
TARS chose rule: fed_avg
Round 42 Accuracy: 97.83%, Loss: 0.0654, Avg Trust: 0.951
🎯 New best accuracy: 97.83%

--- Final Evaluation on Test Set ---
🏆 Using best model from training...
Final Test Accuracy: 97.83%, Final Test Loss: 0.0654
Best Accuracy Achieved: 97.83%
🎉 TARGET ACHIEVED: 97%+ accuracy reached!
```

## 📱 Mobile Access

Colab works on mobile devices:
- Use Colab mobile app
- Monitor training progress remotely
- Download results to phone/tablet

## 🔗 Useful Links

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch on Colab Tutorial](https://pytorch.org/tutorials/beginner/colab.html)
- [TARS GitHub Repository](https://github.com/shafiqahmeddev/tars-fl-sim)

Start training your TARS model now with the power of Google Colab! 🚀