# ERNET_ML

``` verilog
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ----------------- Data Fetching -----------------
def get_stock_data(ticker='AAPL', start='2017-01-01', end='2022-12-31'):
    df = yf.download(ticker, start=start, end=end)
    return df['Close'].values.reshape(-1, 1)

# ----------------- Quantum Circuit -----------------
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i % inputs.shape[0]], wires=i)
        qml.RY(weights[i][0], wires=i)
        qml.RZ(weights[i][1], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])  # ring entanglement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_qubits, 2)}

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    return quantum_circuit(inputs, weights)

qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ----------------- QLSTM Model -----------------
class QLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QLSTM, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_qubits)
        self.quantum = qlayer
        self.lstm = nn.LSTM(n_qubits, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q_out = []

        for t in range(seq_len):
            xt = x[:, t, :]                          # shape: (batch_size, input_dim)
            xt = self.fc1(xt)                        # shape: (batch_size, n_qubits)
            qt = torch.stack([self.quantum(sample) for sample in xt])  # per sample
            q_out.append(qt)

        q_out = torch.stack(q_out, dim=1)            # shape: (batch_size, seq_len, n_qubits)
        out, _ = self.lstm(q_out)                    # shape: (batch_size, seq_len, hidden_dim)
        out = self.dropout(out)
        out = self.fc2(out[:, -1, :])                # last time step
        return out

# ----------------- Sequence Preprocessing -----------------
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ----------------- Training -----------------
def train_model(model, X_train, y_train, epochs=150):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# ----------------- Main Execution -----------------
if __name__ == '__main__':
    # Load and scale data
    data = get_stock_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    seq_length = 10
    X, y = create_sequences(data_scaled, seq_length)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(scaler.transform(y.reshape(-1, 1)), dtype=torch.float32)

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model and training
    model = QLSTM(input_dim=1, hidden_dim=12, output_dim=1)
    train_model(model, X_train, y_train, epochs=150)

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.numpy())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test_inv, label='Actual')
    plt.legend()
    plt.title("Stock Price Prediction with Optimized QLSTM")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid()
    plt.show()

```
