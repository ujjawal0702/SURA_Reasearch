import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

# Load data
data = pd.read_csv('finaldataset.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%d/%m/%y', dayfirst=True)
data = data.sort_values('Year')

# Check for NaN values
if data.isnull().values.any():
    print("Warning: Dataset contains NaN values. Dropping rows with NaN values.")
    data = data.dropna()

# Extract features and target
features = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'IOD Value']
X = data[features].values
y = data['Rainfall'].values

# Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Prepare the dataset for time series forecasting
def create_dataset(X, y, window_size):
    X_out, y_out = [], []
    for i in range(len(X) - window_size):
        X_out.append(X[i:i+window_size])
        y_out.append(y[i+window_size])
    return np.array(X_out), np.array(y_out)

window_size = 30
X, y = create_dataset(X_scaled, y_scaled, window_size)

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define an improved PatchTST model with added variability
class ImprovedPatchTST(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super(ImprovedPatchTST, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Model parameters
input_dim = len(features)
hidden_dim = 128
output_dim = 1
num_layers = 3

model = ImprovedPatchTST(input_dim, hidden_dim, output_dim, num_layers, dropout=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Create DataLoaders
batch_size = 64

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model
num_epochs = 250
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

    if train_loss != train_loss:
        print("NaN loss encountered. Stopping training.")
        break

# Save the model
torch.save(model.state_dict(), 'improved_patchtst_model.pth')

# Function to create input sequence
def create_input_sequence(date, window_size):
    last_data = data[data['Year'] < date].tail(window_size)
    
    if len(last_data) < window_size:
        raise ValueError(f"Not enough historical data for prediction on {date}")
    
    sequence = last_data[features].values
    return scaler_X.transform(sequence).reshape(1, window_size, -1)

# Function to predict rainfall for a single day with added variability
def predict_single_day(date):
    input_seq = create_input_sequence(date, window_size)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy().flatten()[0]
    
    # Add some random variability to the prediction
    variability = np.random.normal(0, 0.1)  # Adjust the scale (0.1) to control the amount of variability
    prediction += variability
    
    return scaler_y.inverse_transform([[prediction]])[0][0]


start_date = datetime.date(2024, 6, 1)
end_date = datetime.date(2024, 9, 30)
date_range = pd.date_range(start=start_date, end=end_date)

predictions_2024 = []
dates_2024 = []

for date in date_range:
    try:
        prediction = predict_single_day(date)
        predictions_2024.append(prediction)
        dates_2024.append(date)
    except ValueError as e:
        print(f"Warning: {e}")
        break

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({
    'Date': dates_2024,
    'Predicted_Rainfall': predictions_2024
})

# Save the predictions to a CSV file
predictions_df.to_csv('AISMR_predictions_2024.csv', index=False)
print("Predictions saved to 'AISMR_predictions_2024.csv'")

# Plot the predictions for 2024
plt.figure(figsize=(15, 5))
plt.plot(dates_2024, predictions_2024)
plt.title('Predicted Daily Rainfall for Summer Monsoon 2024 ')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total rainfall
total_rainfall = np.sum(predictions_2024)
print(f"\nTotal Predicted Rainfall from June 1 to July 17, 2024: {total_rainfall:.2f} mm")

# Calculate monthly totals
monthly_totals = predictions_df.set_index('Date').resample('M')['Predicted_Rainfall'].sum()
print("\nMonthly Rainfall Totals:")
for date, total in monthly_totals.items():
    print(f"{date.strftime('%B %Y')}: {total:.2f} mm")