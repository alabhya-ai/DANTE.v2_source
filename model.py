# This file contains the DANTE.v2 model definition (InsiderClassifier)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition:

class LSTM_Encoder(nn.Module):
    def __init__(self, padding_idx=None, input_size=251, embedding_dim=40, lstm_hidden_size=40, num_layers=3, dropout_rate=0.5):
        super(LSTM_Encoder, self).__init__()

        # Model hyperparameters/constants
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=padding_idx)
        lstm_input_size = embedding_dim

        # One-hot encoder fallback (optional)
        self.one_hot_encoder = F.one_hot
        
        # Core LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True)
            
        self.dropout = nn.Dropout(dropout_rate)
        # Decoder maps hidden_size back to the input vocab size (input_size)
        self.decoder = nn.Linear(lstm_hidden_size, input_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, sequence):
        # sequence shape: (batch_size, seq_len)
        
        # 1. Input Processing (Embedding or One-Hot)
        if self.embedding:
            x = self.embedding(sequence)
        else:
            x = self.one_hot_encoder(sequence,
                num_classes=self.input_size).float()
        # x shape: (batch_size, seq_len, embed_dim)
        
        # 2. LSTM Forward Pass
        x, _ = self.lstm_encoder(x)
        # x shape: (batch_size, seq_len, lstm_hidden_size)

        # 3. Output for Training (Reconstruction) or Inference (Hidden State)
        if self.training:
            x = self.dropout(x)
            x = self.decoder(x)
            x = self.log_softmax(x)
            # Output for reconstruction loss: (batch_size, seq_len, input_size)
            return x
        else:
            # Output for Classifier: (batch_size, seq_len, lstm_hidden_size)
            return x
        

class CNN_Classifier(nn.Module):
    def __init__(self, seq_length=250, lstm_hidden_size=40):
        super(CNN_Classifier, self).__init__()

        self.seq_length = seq_length
        self.lstm_hidden_size = lstm_hidden_size        
        
        final_seq_dim = self.seq_length // 4
        final_hidden_dim = self.lstm_hidden_size // 4
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        # Calculate the required linear input size dynamically
        linear_input_size = 64 * final_seq_dim * final_hidden_dim
        
        self.flatten = lambda x: x.view(x.size(0),-1)
        self.linear = nn.Linear(linear_input_size, 2) # Output 2 classes (malicious/not malicious)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        
        return x

class InsiderClassifier(nn.Module):
    def __init__(self, lstm_checkpoint, device='cuda'):
        super(InsiderClassifier, self).__init__()
        
        # Assuming similar architecture as in the original code:
        lstm_hidden_size = 40
        seq_length = 250 # Adjusted (max_length=250)
        
        self.lstm_encoder = LSTM_Encoder(lstm_hidden_size=lstm_hidden_size)
        self.lstm_encoder.requires_grad = False
        self.lstm_encoder.eval()
        self.load_encoder(lstm_checkpoint, device=device) # Pass device here!

        self.sigmoid = nn.Sigmoid()
        self.cnn_classifier = CNN_Classifier(seq_length=seq_length, lstm_hidden_size=lstm_hidden_size)
        
        # Move the entire model to the correct device upon initialization
        self.to(device) 

    def train(self, mode=True):
        # Only the CNN classifier is trained, the encoder stays in eval mode
        self.training = mode
        self.sigmoid.train(mode)
        self.cnn_classifier.train(mode)
        
        # Crucially, the encoder MUST remain in evaluation mode and have grad disabled.
        self.lstm_encoder.eval() 
        self.lstm_encoder.requires_grad = False
        return self

    def load_encoder(self, checkpoint, device):
        # Map location ensures the checkpoint is loaded correctly regardless of current device
        self.lstm_encoder.load_state_dict(
            torch.load(
                checkpoint,
                map_location=torch.device(device)),
            strict=True
            )
        # Move the encoder to the target device after loading its state
        self.lstm_encoder.to(device)
        return self

    def forward(self, x):
        # Ensure input data is on the same device as the model
        device = next(self.parameters()).device 
        x = x.to(device)

        with torch.no_grad():
            # The encoder is on the correct device due to __init__ and load_encoder
            hidden_state = self.lstm_encoder(x) 
            hidden_state = self.sigmoid(hidden_state)
        
        # hidden_state shape: (batch_size, seq_len, lstm_hidden_size)
        # CNN expects (N, C, H, W). We add the channel dimension (C=1)
        scores = self.cnn_classifier(hidden_state.unsqueeze(1)) 

        return scores
