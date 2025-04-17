import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add parent directory to path so that we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SVRG optimizer
from utils.svrg_optimizer import SVRG_k, SVRG_Snapshot

def check_mps_availability():
    """檢查MPS是否可用"""
    if not torch.backends.mps.is_available():
        print("MPS is not available. This might be because:")
        if not torch.backends.mps.is_built():
            print("- PyTorch was not built with MPS support. Please use PyTorch version >= 1.12.0")
        else:
            print("- This device is not an Apple Silicon device or MPS is not enabled.")
        return False
    return True

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def test_svrg_optimizer_on_mps():
    """測試在MPS設備上使用SVRG優化器進行訓練"""
    print("Testing SVRG optimizer on MPS...")
    
    # 檢查MPS是否可用
    if not check_mps_availability():
        print("Skipping MPS test, will run on CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device('mps')
        print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a simple dataset
    batch_size = 16
    n_samples = 128
    x = torch.randn(n_samples, 10)
    y = (torch.sum(x[:, :5], dim=1, keepdim=True) > 0).float()
    
    # Move data to MPS if available
    if device.type == 'mps':
        x = x.to(device)
        y = y.to(device)
    
    # Create data loader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model and move to MPS
    model = SimpleModel()
    model.to(device)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # SVRG optimizer
    optimizer = SVRG_k(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # SVRG snapshot optimizer
    snapshot = SVRG_Snapshot(model.parameters())
    
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Calculate full gradient for snapshot
        snapshot.set_param_groups(optimizer.get_param_groups())
        model.zero_grad()
        
        # Compute the gradient over all samples (snapshot)
        full_grad_loss = 0
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            full_grad_loss += loss.item()
        
        full_grad_loss /= len(dataloader)
        print(f"Full gradient loss: {full_grad_loss:.4f}")
        
        # Set snapshot gradient to optimizer
        optimizer.set_u(snapshot.get_param_groups())
        
        # Train with SVRG step
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Reset snapshot model parameters to be the same as current model
            snapshot.set_param_groups(optimizer.get_param_groups())
            
            # Forward pass through main model
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass for model
            model.zero_grad()
            loss.backward()
            
            # Compute gradient on the same data for snapshot model
            snapshot_outputs = model(inputs)  # Use the same model but different parameters
            snapshot_loss = criterion(snapshot_outputs, targets)
            
            # Backward pass for snapshot
            snapshot.zero_grad()
            snapshot_loss.backward()
            
            # Update weights using SVRG
            optimizer.step(snapshot.get_param_groups())
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Average training loss: {avg_loss:.4f}")
    
    # Evaluate the final model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"SVRG optimizer test on {device} completed.")
    
    return accuracy > 60  # Return True if accuracy is reasonable

if __name__ == "__main__":
    success = test_svrg_optimizer_on_mps()
    print(f"Test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1) 