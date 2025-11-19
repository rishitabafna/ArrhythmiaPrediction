# model_utils.py
import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = "model2.pth"   # change to your model file if needed
DEVICE = "cpu"


# -------------------- SELF ATTENTION --------------------
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_outputs, mask=None):
        Q = self.query(lstm_outputs)
        K = self.key(lstm_outputs)
        V = self.value(lstm_outputs)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lstm_outputs.size(-1) ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = self.dropout(attn_out)

        context = attn_out.mean(dim=1)
        return context, attn_weights


# -------------------- CNN + BiLSTM + SELF ATTENTION --------------------
class CNN_BiLSTM_SelfAttention(nn.Module):
    def __init__(
        self,
        in_channels=12,
        cnn_channels=[16, 32, 64, 128, 256],
        kernel_size=7,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=9,
        dropout=0.4,
    ):
        super().__init__()

        convs = []
        prev_ch = in_channels
        for ch in cnn_channels:
            convs.append(nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size // 2))
            convs.append(nn.BatchNorm1d(ch))
            convs.append(nn.ReLU())
            convs.append(nn.MaxPool1d(kernel_size=2))
            prev_ch = ch

        self.cnn = nn.Sequential(*convs)

        self.bilstm = nn.LSTM(
            input_size=prev_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.attention = SelfAttention(hidden_dim=lstm_hidden * 2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x, mask=None):
        out = self.cnn(x)                         # [B, C, T]
        out = out.permute(0, 2, 1).contiguous()   # [B, T, C]

        lstm_out, _ = self.bilstm(out)
        context, attn = self.attention(lstm_out, mask)

        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn


# -------------------- LOAD MODEL --------------------
def load_model(device=DEVICE):
    model = CNN_BiLSTM_SelfAttention().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Model loaded!")
    return model


# -------------------- PREDICT SIGNAL --------------------
def predict_signal(model, ecg_tensor, device=DEVICE, threshold=0.5):
    with torch.no_grad():
        logits, _ = model(ecg_tensor.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = (probs >= threshold).astype(int)
    return preds.tolist(), probs.tolist()