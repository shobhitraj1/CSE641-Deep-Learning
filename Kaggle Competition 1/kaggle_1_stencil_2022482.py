import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

torch.manual_seed(2022482)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2022482)

torch.set_default_dtype(torch.float64)

class CompetitionDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing competition data.
    Participants should modify this to load their specific dataset.
    """
    def __init__(self, data_path):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the dataset
        """
        
        data = pd.read_csv(data_path)
        
        self.F1 = data['F1'].values.astype('int64')
        self.F2 = data['F2'].values.astype('int64')
        self.F3 = data['F3'].values.astype('int32').reshape(-1, 1)
        
        if 'OUT' in data.columns:
            self.labels = data['OUT'].values.astype('float64').reshape(-1, 1)
            self.has_labels = True
        else:
            self.has_labels = False
            
        self.indexes = data['INDEX'].values.astype('int32').reshape(-1, 1)
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        
        return len(self.F1)
    
    def __getitem__(self, idx):
        """
        Generate one sample of data.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (input data, label)
        """
        
        if self.has_labels:
            return (self.F1[idx], self.F2[idx], self.F3[idx]), torch.FloatTensor(self.labels[idx]), self.indexes[idx]
        else:
            return (self.F1[idx], self.F2[idx], self.F3[idx]), self.indexes[idx]


class CompetitionModel(nn.Module):
    """
    Base neural network model for the competition.
    Participants should modify the architecture as needed.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output
        """
        super().__init__()
        
        self.f1_linear = nn.Linear(1, 9) # learned projection for F1
        self.f2_linear = nn.Linear(1, 3) # learned projection for F2
        
        combined_input_dim = 6 * 2 + 1  # F1 projection + F2 projection + F3
        
        layers = []
        prev_dim = combined_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU() if i % 2 == 0 else nn.SiLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        
        if isinstance(x, tuple):
            F1, F2, F3 = x
            
            F1 = F1.double().unsqueeze(1)
            F2 = F2.double().unsqueeze(1)
            F3 = F3.double()
            
            f1_proj = self.f1_linear(F1)
            f2_proj = self.f2_linear(F2)
            
            x = torch.cat([f1_proj, f2_proj, F3], dim=1)
        
        return self.model(x)


class CompetitionCriterion(nn.Module):
    """
    Custom loss function for the competition.
    Combines MSE with additional penalty or custom metrics.
    """
    def __init__(self, mse_weight=0.8, complexity_penalty=0.2):
        """
        Initialize the custom loss criterion.
        
        Args:
            mse_weight (float): Weight for Mean Squared Error
            complexity_penalty (float): Weight for model complexity penalty
        """
        super(CompetitionCriterion, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.complexity_penalty = complexity_penalty
    
    def forward(self, predictions, targets, model=None):
        """
        Compute the custom loss.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            model (nn.Module, optional): Neural network model for complexity calculation
        
        Returns:
            torch.Tensor: Computed loss value
        """
        # Calculate base Mean Squared Error
        mse = self.mse_loss(predictions, targets)
        
        return mse


def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    """
    Train the neural network model.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        num_epochs (int, optional): Number of training epochs. Defaults to 50.
    
    Returns:
        dict: Training history with train losses
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    history = {
        'train_loss': []
    }

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    best_mse = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            F1, F2, F3 = features
            F1 = F1.to(device).long()
            F2 = F2.to(device).long()
            F3 = F3.to(device).double()
            features = (F1, F2, F3)
                
            labels = labels.to(device).double()
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels, model)
            loss = criterion(outputs, labels, model)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()         
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        scheduler.step(train_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.6f}, LR: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')
        
        # checkpointing
        if train_loss < best_mse:
            best_mse = train_loss
            torch.save(model.state_dict(), "best_model_27.pth")
            print(f"Checkpoint Saved: Best Train MSE = {best_mse:.6f}")
    
    model.load_state_dict(torch.load("best_model_27.pth"))
    
    return history

def test_model(model_path, test_data_path, output_path='predictions.csv'):
    """
    Test a trained model and generate predictions in a specified CSV format.
    
    Args:
        model_path (str): Path to the saved model weights
        test_data_path (str): Path to the test dataset CSV
        output_path (str, optional): Path to save prediction results. Defaults to 'predictions.csv'
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = CompetitionDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CompetitionModel(
        input_dim=None,
        hidden_dims=[128, 256, 256, 128, 64, 32],
        output_dim=1
    )
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for features, indexes in test_loader:
            F1, F2, F3 = features
            F1 = F1.to(device).long()
            F2 = F2.to(device).long()
            F3 = F3.to(device).double()
            features = (F1, F2, F3)
            
            outputs = model(features)
            
            indexes = indexes.cpu().numpy().squeeze()
            outputs = outputs.cpu().numpy().squeeze()
            
            predictions.extend(zip(indexes, outputs))
    
    output_df = pd.DataFrame(predictions, columns=['ID', 'OUT'])
    
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(output_df)}")
    
    return output_df


def main():
    """
    Main function to set up and run the training pipeline.
    Participants should customize this based on their specific requirements.
    """
    input_dim = None  
    hidden_dims = [128, 256, 256, 128, 64, 32]
    output_dim = 1
    
    train_dataset = CompetitionDataset('/kaggle/input/kaggle-challenge-1-data/kaggle_1_train.csv') 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = CompetitionModel(input_dim, hidden_dims, output_dim)
    
    criterion = CompetitionCriterion(mse_weight=1.0, complexity_penalty=3e-5)
    optimizer = optim.RAdam(model.parameters(), lr=0.0007, weight_decay=3e-5)
    
    history = train_model(model, train_loader, criterion, optimizer, num_epochs=300)
    
    # Save model
    torch.save(model.state_dict(), 'best_model_27.pth')
    
    # Generate predictions
    test_model('best_model_27.pth', '/kaggle/input/kaggle-challenge-1-data/kaggle_1_test.csv', 'predictions.csv')


if __name__ == '__main__':
    main()
