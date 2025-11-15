import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL_PATH = "3.pth"


# Attention Block — 512-dim (matches checkpoint)
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim=512):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim  # ✅ store it as instance variable
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # ✅ fixed
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)



# CNN + BiLSTM + Attention (matches 3.pth exactly)
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN_BiLSTM_Attention, self).__init__()

        # CNN (input 12-lead, kernel_size=7, output 128)
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # BiLSTM (input=128, hidden=256)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention over 512-dim embeddings
        self.attention = AttentionBlock(hidden_dim=512)

        # Fully connected layer (512 → 9)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [batch, 12, seq_len]
        cnn_out = self.cnn(x)                # [batch, 128, seq_len']
        cnn_out = cnn_out.permute(0, 2, 1)   # [batch, seq_len', 128]
        lstm_out, _ = self.bilstm(cnn_out)   # [batch, seq_len', 512]
        attn_out = self.attention(lstm_out)  # [batch, seq_len', 512]
        pooled = torch.mean(attn_out, dim=1) # [batch, 512]
        out = self.fc(pooled)                # [batch, 9]
        return out


def load_model(device='cpu'):
    model = CNN_BiLSTM_Attention(num_classes=9)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    return model


def predict_signal(model, ecg_tensor, device='cpu'):
    with torch.no_grad():
        outputs = model(ecg_tensor.to(device))
        pred = torch.argmax(outputs, dim=1).item()
    return pred
