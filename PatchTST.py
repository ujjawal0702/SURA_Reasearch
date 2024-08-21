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
data = pd.read_csv('newdata.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y', dayfirst=True)
data = data.sort_values('Date')

# Check for NaN values
if data.isnull().values.any():
    print("Warning: Dataset contains NaN values. Dropping rows with NaN values.")
    data = data.dropna()

# Extract features and target
X = data['Rainfall'].values
y = data['Rainfall'].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).flatten()
y_scaled = X_scaled  # Since X and y are the same in this case

# Prepare the dataset for time series forecasting
def create_dataset(X, y, window_size):
    X_out, y_out = [], []
    for i in range(len(X) - window_size):
        X_out.append(X[i:i+window_size])
        y_out.append(y[i+window_size])
    return np.array(X_out), np.array(y_out)

window_size = 30
X, y = create_dataset(X_scaled, y_scaled, window_size)

# Reshape X to have 3 dimensions: [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define an improved PatchTST model
class ImprovedPatchTST(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super(ImprovedPatchTST, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

input_dim = 1  # Only rainfall as input
hidden_dim = 64
output_dim = 1
num_layers = 2

model = ImprovedPatchTST(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Create DataLoaders
batch_size = 32

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

    # Early stopping
    if train_loss != train_loss:  # Check for NaN loss
        print("NaN loss encountered. Stopping training.")
        break

# Save the model
torch.save(model.state_dict(), 'improved_patchtst_model_rainfall_only.pth')

# Load the trained model
model.load_state_dict(torch.load('improved_patchtst_model_rainfall_only.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    train_preds = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    test_preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()

# Inverse scale the predictions
train_preds = scaler.inverse_transform(train_preds.reshape(-1, 1)).flatten()
test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate errors
mse_train = mean_squared_error(y_train, train_preds)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_preds)
r2_train = r2_score(y_train, train_preds)

mse_test = mean_squared_error(y_test, test_preds)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_preds)
r2_test = r2_score(y_test, test_preds)

print(f'Train MSE: {mse_train:.4f}, Train RMSE: {rmse_train:.4f}, Train MAE: {mae_train:.4f}, Train R2: {r2_train:.4f}')
print(f'Test MSE: {mse_test:.4f}, Test RMSE: {rmse_test:.4f}, Test MAE: {mae_test:.4f}, Test R2: {r2_test:.4f}')

# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(data['Date'][window_size:train_size+window_size], y_train, label='Train Data')
plt.plot(data['Date'][train_size+window_size:], y_test, label='Test Data')
plt.plot(data['Date'][window_size:train_size+window_size], train_preds, label='Train Predictions')
plt.plot(data['Date'][train_size+window_size:], test_preds, label='Test Predictions')
plt.xlabel('Date')
plt.ylabel('Rainfall')
plt.legend()
plt.title('All India Summer Monsoon Rainfall Prediction (Univariate)')
plt.show()

def create_input_sequence(date, window_size):
    last_data = data[data['Date'] < date].tail(window_size)
    
    if len(last_data) < window_size:
        raise ValueError(f"Not enough historical data for prediction on {date}")
    
    sequence = last_data['Rainfall'].values
    return scaler.transform(sequence.reshape(-1, 1)).reshape(1, window_size, 1)

# Function to predict rainfall for a single day
def predict_single_day(date):
    input_seq = create_input_sequence(date, window_size)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy().flatten()[0]
    
    return scaler.inverse_transform([[prediction]])[0][0]

# Generate predictions for summer monsoon months of 2021
start_date = datetime.date(2021, 6, 1)
end_date = datetime.date(2021, 9, 30)
date_range = pd.date_range(start=start_date, end=end_date)

predictions_2021 = []
dates_2021 = []

for date in date_range:
    try:
        prediction = predict_single_day(date)
        predictions_2021.append(prediction)
        dates_2021.append(date)
    except ValueError as e:
        print(f"Warning: {e}")
        break

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({
    'Date': dates_2021,
    'Predicted_Rainfall': predictions_2021
})

# Save the predictions to a CSV file
predictions_df.to_csv('AISMR_predictions_2021_rainfall_only.csv', index=False)
print("Predictions saved to 'AISMR_predictions_2021_rainfall_only.csv'")

# Plot the predictions for 2021
plt.figure(figsize=(15, 5))
plt.plot(dates_2021, predictions_2021)
plt.title('Predicted Daily Rainfall for Summer Monsoon 2021')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total rainfall over the 4 months
total_rainfall = np.sum(predictions_2021)
print(f"\nTotal Predicted Rainfall for Summer Monsoon 2021: {total_rainfall:.2f} mm")

# Calculate monthly totals
monthly_totals = predictions_df.set_index('Date').resample('M')['Predicted_Rainfall'].sum()
print("\nMonthly Rainfall Totals:")
for date, total in monthly_totals.items():
    print(f"{date.strftime('%B %Y')}: {total:.2f} mm")