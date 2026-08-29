import os
import numpy as np
import torch
import torch.nn as nn

from src.data_loader import generate_demo_data
from src.preprocessing import preprocess_imu
from src.dead_reckoning import dead_reckoning
from src.ai_adapter import AIAdapter


print("=" * 60)
print("        AI-IMU LSTM TRAINING")
print("=" * 60)


# ============================================================
# 1. Generate and preprocess IMU data
# ============================================================

df = generate_demo_data()

df = preprocess_imu(df)

print("\n[1] IMU data prepared")
print("Samples:", len(df))


# ============================================================
# 2. Physics-based Dead Reckoning
# ============================================================

dr_position, velocity = dead_reckoning(df)

print("[2] Physics DR calculated")


# ============================================================
# Create Ground Truth from actual dataset

ground_truth = df[
    ["gt_x", "gt_y", "gt_z"]
].values.astype(np.float32)

print("[3] Ground truth loaded from dataset")


# ============================================================
# 4. Calculate residual/error
# ============================================================

residual = (
    ground_truth - dr_position
)

print("[3] Residual targets created")


# ============================================================
# 5. Prepare IMU features
# ============================================================

features = df[
    ["ax", "ay", "az", "gx", "gy", "gz"]
].values.astype(np.float32)

residual = residual.astype(np.float32)


# Normalize features
feature_mean = features.mean(axis=0)
feature_std = features.std(axis=0) + 1e-8

features = (
    features - feature_mean
) / feature_std


# ============================================================
# 6. Create sequences
# ============================================================

sequence_length = 20

X = []
Y = []

for i in range(
    sequence_length,
    len(features)
):

    X.append(
        features[
            i-sequence_length:i
        ]
    )

    Y.append(
        residual[i]
    )


X = np.array(X)
Y = np.array(Y)


print("[4] Sequences created")
print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# ============================================================
# Train / Validation split
# ============================================================

rng = np.random.default_rng(42)

indices = np.arange(len(X))

rng.shuffle(indices)

split = int(
    0.8 * len(indices)
)

train_indices = indices[:split]
val_indices = indices[split:]

X_train = X[train_indices]
Y_train = Y[train_indices]

X_val = X[val_indices]
Y_val = Y[val_indices]


# Convert to PyTorch tensors

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

Y_train = torch.tensor(
    Y_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

Y_val = torch.tensor(
    Y_val,
    dtype=torch.float32
)


# ============================================================
# 8. Create LSTM model
# ============================================================

model = AIAdapter(
    input_size=6,
    hidden_size=32
)

print("\n[5] LSTM model created")
print(model)


# ============================================================
# 9. Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. Training
# ============================================================

epochs = 100

print("\n[6] Training started...\n")


for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    prediction = model(
        X_train
    )

    loss = criterion(
        prediction,
        Y_train
    )

    loss.backward()

    optimizer.step()


    # Validation

    model.eval()

    with torch.no_grad():

        val_prediction = model(
            X_val
        )

        val_loss = criterion(
            val_prediction,
            Y_val
        )


    if (
        epoch == 0
        or (epoch + 1) % 5 == 0
    ):

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Train Loss: {loss.item():.6f} "
            f"| Val Loss: {val_loss.item():.6f}"
        )


# ============================================================
# 11. Save model
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = (
    "models/ai_adapter.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


print("\n[7] Training completed!")

print(
    "Model saved at:",
    model_path
)

print("\nAI model is ready!")