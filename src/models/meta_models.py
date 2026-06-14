# src/models/meta_models.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def _prepare_backbone(base_cnn: nn.Module) -> int:
    """Strip the classification head off a backbone (in place) and return its
    feature dimension. Shared by every metadata-fusion model so the
    backbone-introspection logic lives in one place."""
    if hasattr(base_cnn, "get_classifier"):
        feat_dim = base_cnn.get_classifier().in_features
        base_cnn.reset_classifier(0, "")  # keep spatial features (global_pool='')
    elif hasattr(base_cnn, "fc"):
        feat_dim = base_cnn.fc.in_features
        base_cnn.fc = nn.Identity()
    elif hasattr(base_cnn, "classifier"):
        clf = base_cnn.classifier
        if isinstance(clf, nn.Linear):
            feat_dim = clf.in_features
            base_cnn.classifier = nn.Identity()
        elif isinstance(clf, nn.Sequential):
            feat_dim = next((l.in_features for l in clf if isinstance(l, nn.Linear)), None)
            if feat_dim is None:
                raise ValueError("Could not determine feature dim from Sequential classifier.")
            base_cnn.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported classifier structure to get feature_dim.")
    else:
        raise ValueError("Cannot determine feature dimension / reset classifier for backbone.")
    logger.info(f"Base CNN feature dimension: {feat_dim}")
    return feat_dim


def _backbone_feature_map(base_cnn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the backbone's spatial feature map ``(B, C, H, W)`` for ``x``,
    preferring timm's ``forward_features``."""
    if hasattr(base_cnn, "forward_features"):
        feats = base_cnn.forward_features(x)
    else:
        feats = base_cnn(x)
    return feats

class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256, dropout_p: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)
        logger.info(f"MetadataMLP initialized: In={input_dim}, Hidden={hidden_dim}, Out={output_dim}, Dropout={dropout_p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

class CNNWithMetadata(nn.Module):
    def __init__(self,
                 base_cnn_model: nn.Module,
                 num_classes: int,
                 metadata_input_dim: int,
                 meta_mlp_hidden_dim: int = 256,
                 meta_mlp_output_dim: int = 256,
                 meta_dropout_p: float = 0.4,
                 post_concat_dim: int = 1024,
                 post_concat_dropout_p: float = 0.4):
        super().__init__()
        self.base_cnn_model = base_cnn_model
        self.metadata_input_dim = metadata_input_dim

        self.cnn_feature_dim = _prepare_backbone(self.base_cnn_model)

        self.metadata_mlp = MetadataMLP(
            input_dim=metadata_input_dim,
            hidden_dim=meta_mlp_hidden_dim,
            output_dim=meta_mlp_output_dim,
            dropout_p=meta_dropout_p
        )
        
        combined_feature_dim = self.cnn_feature_dim + meta_mlp_output_dim
        
        self.post_concat_fc = nn.Linear(combined_feature_dim, post_concat_dim)
        self.post_concat_bn = nn.BatchNorm1d(post_concat_dim)
        self.post_concat_relu = nn.ReLU()
        self.post_concat_dropout = nn.Dropout(p=post_concat_dropout_p)
        
        self.final_classifier = nn.Linear(post_concat_dim, num_classes)
        
        logger.info(f"CNNWithMetadata initialized: CNN feats={self.cnn_feature_dim}, Meta feats={meta_mlp_output_dim}, "
                    f"Combined={combined_feature_dim}, PostConcat={post_concat_dim}, NumClasses={num_classes}")

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        # If base_cnn_model.forward_features exists (common in timm)
        if hasattr(self.base_cnn_model, 'forward_features'):
            cnn_features = self.base_cnn_model.forward_features(image_input)
            # Timm models with reset_classifier(0,'') might do GAP in forward_features or need it after
            # For many timm models, after forward_features, if global_pool='', features are [B, C, H, W]
            # If global_pool was set e.g. to 'avg', it's [B,C].
            # Let's ensure it's [B,C] via adaptive avg pooling if needed.
            if cnn_features.ndim == 4: # B, C, H, W
                cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1)).squeeze(-1).squeeze(-1)
        else: # Otherwise, assume self.base_cnn_model(image_input) gives features due to Identity classifier
            cnn_features = self.base_cnn_model(image_input)
            # Output might already be pooled if base_cnn has GAP before Identity, or might need pooling.
            if cnn_features.ndim == 4: # B, C, H, W
                 cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1)).squeeze(-1).squeeze(-1)


        metadata_features = self.metadata_mlp(metadata_input)
        
        combined_features = torch.cat((cnn_features, metadata_features), dim=1)
        
        x = self.post_concat_fc(combined_features)
        x = self.post_concat_bn(x)
        x = self.post_concat_relu(x)
        x = self.post_concat_dropout(x)
        
        output_logits = self.final_classifier(x)
        return output_logits

    def set_base_cnn_trainable(self, trainable: bool):
        for param in self.base_cnn_model.parameters():
            param.requires_grad = trainable
        status = "trainable" if trainable else "frozen"
        logger.info(f"Base CNN model parameters set to {status}.")


class MetaBlock(nn.Module):
    """Attention-based metadata fusion (Pacheco & Krohling, 2021).

    Metadata produces per-channel gate/shift terms that modulate the CNN feature
    map before pooling, so the network emphasises image features conditioned on
    the patient context. Outperforms plain concatenation in the literature.
    """
    def __init__(self, n_channels: int, metadata_dim: int):
        super().__init__()
        self.fb = nn.Linear(metadata_dim, n_channels)
        self.gb = nn.Linear(metadata_dim, n_channels)

    def forward(self, V: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # V: (B, C, H, W) feature map; meta: (B, M)
        t1 = self.fb(meta).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        t2 = self.gb(meta).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return torch.sigmoid(torch.tanh(V * t1) + t2)


class CNNWithMetaBlock(nn.Module):
    """Backbone + MetaBlock attention fusion + linear head.

    Drop-in alternative to :class:`CNNWithMetadata` with the same
    ``forward(image, metadata)`` signature, selectable via ``meta_fusion=metablock``.
    """
    def __init__(self, base_cnn_model: nn.Module, num_classes: int,
                 metadata_input_dim: int, classifier_dropout_p: float = 0.0):
        super().__init__()
        self.base_cnn_model = base_cnn_model
        self.metadata_input_dim = metadata_input_dim
        self.cnn_feature_dim = _prepare_backbone(self.base_cnn_model)

        self.metablock = MetaBlock(self.cnn_feature_dim, metadata_input_dim)
        self.dropout = nn.Dropout(p=classifier_dropout_p)
        self.final_classifier = nn.Linear(self.cnn_feature_dim, num_classes)
        logger.info(f"CNNWithMetaBlock initialized: CNN feats={self.cnn_feature_dim}, "
                    f"meta_dim={metadata_input_dim}, NumClasses={num_classes}")

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        fmap = _backbone_feature_map(self.base_cnn_model, image_input)
        if fmap.ndim == 2:  # backbone already pooled -> restore spatial dims for the block
            fmap = fmap.unsqueeze(-1).unsqueeze(-1)
        fmap = self.metablock(fmap, metadata_input)
        pooled = nn.functional.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
        return self.final_classifier(self.dropout(pooled))

    def set_base_cnn_trainable(self, trainable: bool):
        for param in self.base_cnn_model.parameters():
            param.requires_grad = trainable
        status = "trainable" if trainable else "frozen"
        logger.info(f"Base CNN model parameters set to {status}.")


def build_meta_model(base_cnn_model: nn.Module, num_classes: int,
                     metadata_input_dim: int, fusion: str = "concat", **head_args) -> nn.Module:
    """Construct a metadata-fusion model. ``fusion='concat'`` -> CNNWithMetadata
    (feature concatenation); ``fusion='metablock'`` -> CNNWithMetaBlock (attention)."""
    fusion = (fusion or "concat").lower()
    if fusion in ("concat", "concatenation"):
        return CNNWithMetadata(base_cnn_model, num_classes, metadata_input_dim, **head_args)
    if fusion == "metablock":
        dropout = head_args.get("post_concat_dropout_p", head_args.get("meta_dropout_p", 0.0))
        return CNNWithMetaBlock(base_cnn_model, num_classes, metadata_input_dim,
                                classifier_dropout_p=dropout)
    raise ValueError(f"Unknown meta_fusion '{fusion}'. Use 'concat' or 'metablock'.")