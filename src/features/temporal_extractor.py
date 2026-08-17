import torch
import torch.nn as nn
from torchvision import models


class TemporalClassifier(nn.Module):
    """
    CNN + LSTM temporal branch.

    A per-frame ResNet-18 backbone (ImageNet-pretrained, fine-tuned -- NOT
    frozen) extracts spatial features from each frame in the sequence
    independently; an LSTM then models how those features evolve across the
    16-frame sequence. The LSTM's final hidden state is the temporal
    embedding.

    train_temporal.py uses a lower learning rate for cnn_backbone than for
    the LSTM/classifier head (standard fine-tuning practice), so the
    pretrained features aren't destroyed by a few epochs on a comparatively
    small dataset before the newly-initialized LSTM has learned to use them.

    forward(x) returns (embedding, logits), matching SemanticClassifier's
    contract exactly:
      - embedding: [B, embedding_dim] — the representation Phase 3 (Feature
        Fusion) will concatenate with the semantic embedding. Defaults to
        256 to match SemanticClassifier and config.yaml's
        models.temporal.features_dim.
      - logits: [B, 2] — real/fake scores, for standalone training/eval of
        this branch before fusion exists.

    Input: x of shape [B, T, 3, 224, 224] (T=16 by default), already
    face-cropped and ImageNet-normalized — see temporal_dataset.py.
    """

    def __init__(self, cnn_feature_dim=512, embedding_dim=256, lstm_layers=1, num_classes=2):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Drop the final classification layer — we only want pooled features.
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> [B, 512, 1, 1]

        self.cnn_feature_dim = cnn_feature_dim

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=embedding_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape

        # Run the CNN backbone on every frame in one batched pass:
        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        x = x.view(batch_size * num_frames, channels, height, width)

        # Backbone is fine-tuned (not frozen) -- gradients flow through it,
        # so no torch.no_grad() here. This does mean B*T frames' worth of
        # backbone activations are kept for backprop instead of just the
        # LSTM's; if you hit CUDA OOM at the configured batch size, lowering
        # temporal batch_size is the first thing to try (see config.yaml).
        features = self.cnn_backbone(x)  # [B*T, 512, 1, 1]

        features = features.view(batch_size, num_frames, self.cnn_feature_dim)  # [B, T, 512]

        # LSTM over the frame sequence. We only need the final hidden state.
        _, (hidden, _) = self.lstm(features)  # hidden: [num_layers, B, embedding_dim]
        embedding = hidden[-1]  # [B, embedding_dim] — last layer's final hidden state

        logits = self.classifier_head(embedding)

        return embedding, logits
