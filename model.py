import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Load and preprocess the data
df = pd.read_csv("enriched/de_train_enriched.csv")
X = df.iloc[:, :-1].values  # Assuming that the last column is the target variable
y = df.iloc[:, -1].values  # Assuming that the last column is the target variable
cell_types = df['cell_type'].values  # Replace 'cell_type' with your actual cell type column name

# Indices based on cell type
indices_t_nk = np.where((cell_types == 'T') | (cell_types == 'NK'))[0]
indices_b_myeloid = np.where((cell_types == 'B') | (cell_types == 'Myeloid'))[0]

# Train-Test split for T and NK cells
indices_train_t_nk, indices_test_t_nk = train_test_split(indices_t_nk, test_size=0.2, random_state=42)

# Further split the training set of T and NK cells into Train and Validation
indices_train_t_nk, indices_val_t_nk = train_test_split(indices_train_t_nk, test_size=0.1, random_state=42)

# Use 10% of B and Myeloid cells for training
n_train_b_myeloid = int(0.1 * len(indices_b_myeloid))
indices_train_b_myeloid = np.random.choice(indices_b_myeloid, size=n_train_b_myeloid, replace=False)
indices_test_b_myeloid = np.setdiff1d(indices_b_myeloid, indices_train_b_myeloid)

# Final index lists for train, validation, and test sets
indices_train = np.concatenate([indices_train_t_nk, indices_train_b_myeloid])
indices_val = indices_val_t_nk
indices_test = np.concatenate([indices_test_t_nk, indices_test_b_myeloid])

# Actual data splits
X_train, y_train = X[indices_train], y[indices_train]
X_val, y_val = X[indices_val], y[indices_val]
X_test, y_test = X[indices_test], y[indices_test]

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


# Base Learner
class BaseLearner(nn.Module):
    def __init__(self):
        super(BaseLearner, self).__init__()
        self.fc1 = nn.Linear(27, 256)  # Assuming the input features are 27, adjust if different
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1)  # Assuming single target regression, adjust if different

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Meta Learner
class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(512 + 1, 256)  # Concatenation of base output and cell type
        self.fc2 = nn.Linear(256, 1)  # Assuming single target regression, adjust if different

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate models and move them to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_learner = BaseLearner().to(device)
meta_learner = MetaLearner().to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(base_learner.parameters()) + list(meta_learner.parameters()), lr=0.001)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter()

# Training Loop with tqdm progress bar
for epoch in tqdm(range(100), desc="Training Epochs"):
    # Transfer data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass through base learner
    base_learner_output = base_learner(X_train)

    # Concatenate base learner output with cell type
    cell_type = torch.tensor([1 if ct in ['T', 'NK'] else 0 for ct in cell_types[indices_train]],
                             dtype=torch.float32).to(device)
    meta_learner_input = torch.cat((base_learner_output, cell_type.unsqueeze(1)), dim=1)

    # Forward pass through meta learner
    meta_learner_output = meta_learner(meta_learner_input)

    # Calculate loss
    loss = criterion(meta_learner_output, y_train.unsqueeze(1))

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Log to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

# Save the models
torch.save(base_learner.state_dict(), "base_learner.pth")
torch.save(meta_learner.state_dict(), "meta_learner.pth")

# Close the SummaryWriter
writer.close()
