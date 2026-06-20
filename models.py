from __future__ import annotations

"""AIC vision-offset regression models.

All models output a 6D correction vector:
    [dx_m, dy_m, dz_m, droll_rad, dpitch_rad, dyaw_rad]
in the base_link convention stored by PortOffsetCollect.
"""

import torch
from torch import nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - import-time environment guard
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


class TimmFeatureBackbone(nn.Module):
    """timm feature-map backbone with a small projection head.

    The referenced regularization notebook uses timm with ``num_classes=0`` and
    ``global_pool=""`` so the model exposes spatial feature maps. We keep that
    behavior here, then project the native channel count to ``feature_dim`` so
    bilinear pooling stays tractable.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for AIC bilinear CNN models. Install timm in "
                "the active environment."
            ) from _TIMM_IMPORT_ERROR

        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.raw_feature_dim = int(self.model.num_features)
        self.feature_dim = int(feature_dim)

        if self.raw_feature_dim == self.feature_dim:
            self.project = nn.Identity()
        else:
            self.project = nn.Sequential(
                nn.Conv2d(self.raw_feature_dim, self.feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=True),
            )

    def _to_nchw_feature_map(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if features.shape[1] == self.raw_feature_dim:
                return features
            if features.shape[-1] == self.raw_feature_dim:
                return features.permute(0, 3, 1, 2).contiguous()
            raise ValueError(f"Unsupported 4D timm feature shape: {tuple(features.shape)}")

        if features.ndim == 3:
            batch, tokens, channels = features.shape
            if channels != self.raw_feature_dim and tokens == self.raw_feature_dim:
                features = features.transpose(1, 2).contiguous()
                batch, tokens, channels = features.shape
            if channels != self.raw_feature_dim:
                raise ValueError(f"Unsupported 3D timm feature shape: {tuple(features.shape)}")
            side = int(tokens**0.5)
            if side * side == tokens:
                return features.transpose(1, 2).reshape(batch, channels, side, side)
            return features.transpose(1, 2).unsqueeze(-1)

        if features.ndim == 2:
            if features.shape[1] != self.raw_feature_dim:
                raise ValueError(f"Unsupported 2D timm feature shape: {tuple(features.shape)}")
            return features[:, :, None, None]

        raise ValueError(f"Unsupported timm feature shape: {tuple(features.shape)}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(image)
        else:
            features = self.model(image)
        features = self._to_nchw_feature_map(features)
        return self.project(features)


def bilinear_pool(feature_map: torch.Tensor) -> torch.Tensor:
    """Compute normalized bilinear outer-product descriptor."""
    batch, channels, height, width = feature_map.shape
    descriptors = feature_map.reshape(batch, channels, height * width)
    pooled = torch.bmm(descriptors, descriptors.transpose(1, 2)) # binlinear outer product
    pooled = pooled / float(height * width) # average pooling
    pooled = pooled.reshape(batch, channels * channels)
    pooled = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1e-5)  
    return F.normalize(pooled, dim=1)


class SimpleCNNRegressor(nn.Module):
    """Plain timm CNN baseline without view-specific branches.

    Training input is multiview [B, V, 3, H, W]. The same CNN is applied to
    every view, then the feature maps are averaged before global average
    pooling. A 4D [B, 3, H, W] path is kept only for quick debugging.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 6,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = TimmFeatureBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            feature_dim=feature_dim,
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            features = self.backbone(images)
        elif images.dim() == 5:
            batch, views, channels, height, width = images.shape
            flat = images.reshape(batch * views, channels, height, width)
            features = self.backbone(flat)
            _, feature_channels, feature_h, feature_w = features.shape
            features = features.reshape(
                batch,
                views,
                feature_channels,
                feature_h,
                feature_w,
            ).mean(dim=1)
        else:
            raise ValueError(f"Expected 4D or 5D image tensor, got {images.shape}")
        return self.head(features)


class SharedBilinearCNNRegressor(nn.Module):
    """Bilinear CNN without view-specific branches.

    Training input is multiview [B, V, 3, H, W]. The same feature extractor is
    applied to all views and the feature maps are averaged before one bilinear
    pooling operation. This intentionally does not distinguish left/center/right.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 6,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = TimmFeatureBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            feature_dim=feature_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim * feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            features = self.backbone(images)
        elif images.dim() == 5:
            batch, views, channels, height, width = images.shape
            flat = images.reshape(batch * views, channels, height, width)
            features = self.backbone(flat)
            _, feature_channels, feature_h, feature_w = features.shape
            features = features.reshape(
                batch,
                views,
                feature_channels,
                feature_h,
                feature_w,
            ).mean(dim=1)
        else:
            raise ValueError(f"Expected 4D or 5D image tensor, got {images.shape}")
        return self.head(bilinear_pool(features))


class MultiViewBilinearCNNRegressor(nn.Module):
    """View-aware Bilinear CNN.

    The model keeps separate left/center/right feature streams. Each stream
    performs its own bilinear outer product, then the descriptors are
    concatenated for final 6D correction regression.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 6,
        num_views: int = 3,
        share_backbone_weights: bool = True,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.share_backbone_weights = share_backbone_weights
        if share_backbone_weights:
            self.shared_backbone = TimmFeatureBackbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                feature_dim=feature_dim,
            )
            self.view_backbones = None
        else:
            self.shared_backbone = None
            self.view_backbones = nn.ModuleList(
                TimmFeatureBackbone(
                    backbone_name=backbone_name,
                    pretrained=pretrained,
                    feature_dim=feature_dim,
                )
                for _ in range(num_views)
            )
        descriptor_dim = num_views * feature_dim * feature_dim
        self.head = nn.Sequential(
            nn.Linear(descriptor_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 5:
            raise ValueError(
                "MultiViewBilinearCNNRegressor expects [B, V, 3, H, W] input"
            )
        batch, views, channels, height, width = images.shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {views}")

        descriptors = []
        for view_index in range(self.num_views):
            view_image = images[:, view_index, :, :, :]
            if self.share_backbone_weights:
                features = self.shared_backbone(view_image)
            else:
                features = self.view_backbones[view_index](view_image)
            descriptors.append(bilinear_pool(features))
        fused = torch.cat(descriptors, dim=1)
        return self.head(fused)


def build_model(
    name: str,
    feature_dim: int = 128,
    backbone_name: str = "efficientnetv2_rw_s",
    pretrained: bool = True,
) -> nn.Module:
    name = name.strip().lower()
    if name in {"simple", "simple_cnn"}:
        return SimpleCNNRegressor(
            feature_dim=feature_dim,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
    if name in {"bilinear", "shared_bilinear"}:
        return SharedBilinearCNNRegressor(
            feature_dim=feature_dim,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
    if name in {"multiview_bilinear", "mv_bilinear"}:
        return MultiViewBilinearCNNRegressor(
            feature_dim=feature_dim,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
    raise ValueError(
        "Unknown model name. Use one of: simple_cnn, shared_bilinear, "
        "multiview_bilinear"
    )
