"""
Deep Learning Model Comparison for Y1 and Y2 Prediction
5 Different Neural Network Architectures - IMPROVED VERSION

Training: 80% (64,000 samples)
Testing: 20% (16,000 samples)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
import sys
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_preprocess_data(train_path):
    """Load and preprocess the training data."""
    print("\n" + "="*60)
    print("Loading and Preprocessing Data")
    print("="*60)
    
    df = pd.read_csv(train_path)
    print(f"Total dataset shape: {df.shape}")
    
    feature_cols = ['time', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    target_cols = ['Y1', 'Y2']
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    input_dim = X.shape[1]
    print(f"Number of features: {input_dim}")
    print(f"Number of targets: {len(target_cols)}")
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]} (80%)")
    print(f"Testing samples: {X_test.shape[0]} (20%)")
    
    return X_train, X_test, y_train, y_test, scaler, input_dim


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=256):
    """Create PyTorch DataLoaders."""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# IMPROVED Model Definitions - 5 Different Architectures
# ============================================================================

class Model1_WideMLP(nn.Module):
    """Model 1: Wide MLP with Dropout regularization"""
    def __init__(self, input_dim, output_dim=2):
        super(Model1_WideMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class Model2_VeryDeepMLP(nn.Module):
    """Model 2: Very Deep MLP with Dropout regularization"""
    def __init__(self, input_dim, output_dim=2, dropout=0.25):
        super(Model2_VeryDeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class Model3_BatchNormDeepMLP(nn.Module):
    """Model 3: Deep MLP with Batch Normalization + Dropout"""
    def __init__(self, input_dim, output_dim=2, dropout=0.2):
        super(Model3_BatchNormDeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class Model4_RegularizedMLP(nn.Module):
    """Model 4: MLP with Dropout + LayerNorm for regularization"""
    def __init__(self, input_dim, output_dim=2, dropout_rate=0.3):
        super(Model4_RegularizedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ImprovedResidualBlock(nn.Module):
    """Improved Residual Block with pre-activation, wider layers, and dropout"""
    def __init__(self, dim, hidden_dim=None, dropout=0.2):
        super(ImprovedResidualBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
        
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.block(x)


class Model5_DeepResNet(nn.Module):
    """Model 5: Deep ResNet-style MLP with skip connections and dropout"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=512, num_blocks=6, dropout=0.2):
        super(Model5_DeepResNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Stack multiple residual blocks with dropout
        self.res_blocks = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_model(model, train_loader, test_loader, epochs=150, lr=0.001, patience=40):
    """Train a model with cosine annealing and early stopping."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)  # Increased weight decay
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(test_loader.dataset)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.6f}, Test: {val_loss:.6f}, LR: {current_lr:.2e}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, test_loader, y_test):
    """Evaluate model and compute metrics for Y1 and Y2."""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Calculate metrics for Y1
    mse_y1 = mean_squared_error(y_test[:, 0], predictions[:, 0])
    mae_y1 = mean_absolute_error(y_test[:, 0], predictions[:, 0])
    r2_y1 = r2_score(y_test[:, 0], predictions[:, 0])
    
    # Calculate metrics for Y2
    mse_y2 = mean_squared_error(y_test[:, 1], predictions[:, 1])
    mae_y2 = mean_absolute_error(y_test[:, 1], predictions[:, 1])
    r2_y2 = r2_score(y_test[:, 1], predictions[:, 1])
    
    # Accuracy: percentage of predictions within threshold of 0.1
    threshold = 0.1
    acc_y1 = np.mean(np.abs(y_test[:, 0] - predictions[:, 0]) < threshold) * 100
    acc_y2 = np.mean(np.abs(y_test[:, 1] - predictions[:, 1]) < threshold) * 100
    
    return {
        'MSE_Y1': mse_y1, 'MAE_Y1': mae_y1, 'R2_Y1': r2_y1, 'Acc_Y1': acc_y1,
        'MSE_Y2': mse_y2, 'MAE_Y2': mae_y2, 'R2_Y2': r2_y2, 'Acc_Y2': acc_y2
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    start_time = time.time()
    
    # Load data
    train_path = 'AMLData/train.csv'
    X_train, X_test, y_train, y_test, scaler, input_dim = load_and_preprocess_data(train_path)
    
    # Create dataloaders with smaller batch size for better gradient estimates
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=256)
    
    # Define improved models
    models = {
        'Model 1: Wide MLP': Model1_WideMLP(input_dim),
        'Model 2: Very Deep MLP': Model2_VeryDeepMLP(input_dim),
        'Model 3: BatchNorm Deep MLP': Model3_BatchNormDeepMLP(input_dim),
        'Model 4: Regularized MLP': Model4_RegularizedMLP(input_dim),
        'Model 5: Deep ResNet': Model5_DeepResNet(input_dim, hidden_dim=512, num_blocks=6)
    }
    
    # Train and evaluate all models
    results = {}
    
    for name, model in models.items():
        print("\n" + "="*60)
        print(f"Training {name}")
        print("="*60)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")
        
        model_start = time.time()
        trained_model = train_model(model, train_loader, test_loader, epochs=150, lr=0.001, patience=40)
        model_time = time.time() - model_start
        
        metrics = evaluate_model(trained_model, test_loader, y_test)
        results[name] = metrics
        
        print(f"\n{name} Results (Training time: {model_time:.1f}s):")
        print(f"  Y1 - MSE: {metrics['MSE_Y1']:.6f}, MAE: {metrics['MAE_Y1']:.6f}, R²: {metrics['R2_Y1']:.4f}, Acc: {metrics['Acc_Y1']:.2f}%")
        print(f"  Y2 - MSE: {metrics['MSE_Y2']:.6f}, MAE: {metrics['MAE_Y2']:.6f}, R²: {metrics['R2_Y2']:.4f}, Acc: {metrics['Acc_Y2']:.2f}%")
    
    # ========================================================================
    # Summary Comparison
    # ========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY (IMPROVED MODELS)")
    print("="*80)
    
    print("\n" + "-"*80)
    print(f"{'Model':<30} {'Acc_Y1 (%)':<15} {'Acc_Y2 (%)':<15} {'R²_Y1':<12} {'R²_Y2':<12}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['Acc_Y1']:<15.2f} {metrics['Acc_Y2']:<15.2f} {metrics['R2_Y1']:<12.4f} {metrics['R2_Y2']:<12.4f}")
    
    print("-"*80)
    
    print("\n" + "-"*80)
    print(f"{'Model':<30} {'MSE_Y1':<12} {'MSE_Y2':<12} {'MAE_Y1':<12} {'MAE_Y2':<12}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['MSE_Y1']:<12.6f} {metrics['MSE_Y2']:<12.6f} {metrics['MAE_Y1']:<12.6f} {metrics['MAE_Y2']:<12.6f}")
    
    print("-"*80)
    
    # Find best models
    best_acc_y1 = max(results.items(), key=lambda x: x[1]['Acc_Y1'])
    best_acc_y2 = max(results.items(), key=lambda x: x[1]['Acc_Y2'])
    best_r2_y1 = max(results.items(), key=lambda x: x[1]['R2_Y1'])
    best_r2_y2 = max(results.items(), key=lambda x: x[1]['R2_Y2'])
    
    print("\n" + "="*80)
    print("BEST MODELS:")
    print("="*80)
    print(f"Best Acc_Y1: {best_acc_y1[0]} ({best_acc_y1[1]['Acc_Y1']:.2f}%)")
    print(f"Best Acc_Y2: {best_acc_y2[0]} ({best_acc_y2[1]['Acc_Y2']:.2f}%)")
    print(f"Best R²_Y1:  {best_r2_y1[0]} ({best_r2_y1[1]['R2_Y1']:.4f})")
    print(f"Best R²_Y2:  {best_r2_y2[0]} ({best_r2_y2[1]['R2_Y2']:.4f})")
    print(f"\nTotal execution time: {total_time:.1f}s")
    
    # Save results to JSON file
    import json
    results_for_json = {name: {k: float(v) for k, v in metrics.items()} for name, metrics in results.items()}
    with open('results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nResults saved to results.json")
    
    return results


if __name__ == "__main__":
    results = main()
