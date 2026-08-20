import torch.nn as nn
from torchvision import models


class SemanticClassifier(nn.Module):
    """
    EfficientNet-B0 based semantic branch.

    forward(x) returns (embedding, logits):
      - embedding: the feature vector immediately before the final
        classification layer (shape [B, embedding_dim]). This is the
        representation Phase 3 (Feature Fusion) will concatenate with the
        temporal branch's embedding.
      - logits: raw classification scores over [real, fake]. Existing
        callers that only need a prediction can keep doing
        `preds = torch.argmax(logits, dim=1)` unchanged.

    embedding_dim defaults to 256 to match config.yaml's
    models.semantic.features_dim, which Phase 3's fusion config already
    assumes. Changing this from the previous hidden size of 512 means the
    model must be retrained from this point forward.

    The backbone is ImageNet-pretrained but NOT frozen -- it is fine-tuned
    jointly with the embedding/classifier heads. See train_semantic.py's
    optimizer setup: the backbone uses a lower learning rate than the heads
    (standard fine-tuning practice) so a few epochs on a comparatively small
    dataset don't wash out the pretrained low/mid-level features before the
    heads have learned to use them.
    """

    def __init__(self, num_classes=2, embedding_dim=256):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features

        # Replace torchvision's classifier with Identity so
        # self.backbone(x) returns pooled EfficientNet features directly,
        # and we own the embedding/classification split ourselves.
        self.backbone.classifier = nn.Identity()

        self.embedding_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        pooled_features = self.backbone(x)
        embedding = self.embedding_head(pooled_features)
        logits = self.classifier_head(embedding)
        return embedding, logits
