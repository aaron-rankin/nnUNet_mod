import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def _foreground_onehot(target_bc1: torch.Tensor, num_classes: int,
                       ref: torch.Tensor) -> torch.Tensor:
    """One-hot the foreground classes 1..num_classes of a (B, 1, *spatial) label map.

    Uses `scatter_` rather than `F.one_hot(...).permute(0, 4, 1, 2, 3)`, which hardcodes a
    5D layout and therefore cannot handle 2D configurations. Ignore voxels must already be
    mapped to 0 by the caller.
    """
    oh = torch.zeros(
        (target_bc1.shape[0], num_classes + 1, *target_bc1.shape[2:]),
        device=ref.device, dtype=torch.float32,
    )
    oh.scatter_(1, target_bc1.long(), 1)
    return oh[:, 1:]  # drop background -> channels correspond to labels 1..num_classes


def _morphological_boundary_mask(seg_onehot: torch.Tensor, radius: int = 3,
                                 inplane_only: bool = False) -> torch.Tensor:
    """
    GPU morphological boundary shell (dilation minus erosion).

    Args:
        seg_onehot: (B, C, H, W) for 2D, or (B, C, Z, Y, X) for 3D; float in [0, 1]
        radius: morphological shell radius in voxels
        inplane_only: restrict the structuring element to the in-plane axes (3D only)

    Returns:
        Binary mask of same shape: 1 inside boundary shell, 0 elsewhere

    `inplane_only` exists because this dataset annotates only even z-slices; odd slices carry
    the ignore label and every loss maps ignore -> background before one-hot. A 3D erosion with
    any z-radius >= 1 therefore annihilates *every* foreground voxel (eroded is identically
    zero), so the "shell" becomes the entire structure and the loss silently degenerates:
    BoundaryCE to foreground-weighted CE, SurfaceDice to plain Dice. Restricting the structuring
    element to (1, k, k) keeps erosion well-defined. It is also the physically sensible choice
    here: at [4.0, 0.406, 0.406] mm spacing an isotropic voxel radius of 5 spans 20 mm
    through-plane against 2 mm in-plane.
    """
    k = 2 * radius + 1
    if seg_onehot.ndim == 4:
        pool, kern, pad = F.max_pool2d, k, radius
    elif seg_onehot.ndim == 5:
        pool = F.max_pool3d
        kern, pad = ((1, k, k), (0, radius, radius)) if inplane_only else (k, radius)
    else:
        raise ValueError(
            f"seg_onehot must be 4D (B,C,H,W) or 5D (B,C,Z,Y,X), got {seg_onehot.ndim}D"
        )
    # Dilate: max pooling (foreground expands)
    dilated = pool(seg_onehot, kernel_size=kern, stride=1, padding=pad)
    # Erode: min pooling = -max_pool(-x) (foreground shrinks)
    eroded = -pool(-seg_onehot, kernel_size=kern, stride=1, padding=pad)
    # Boundary = dilated - eroded
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary


class BoundaryCELoss(nn.Module):
    """
    Cross-entropy with per-voxel spatial reweighting.
    Voxels inside the morphological boundary shell receive weight `boundary_weight`.
    Interior and background voxels receive weight 1.0.
    
    Args:
        boundary_radius: morphological shell radius in voxels (default 3)
        boundary_weight: CE weight multiplier for boundary voxels (default 3.0)
        ignore_index: label value to exclude from loss (default -100)
    """
    
    def __init__(self, boundary_radius: int = 3, boundary_weight: float = 3.0,
                 ignore_index: int = -100, inplane_only: bool = False):
        super().__init__()
        self.boundary_radius = boundary_radius
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.inplane_only = inplane_only
        self.ce = RobustCrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                seg_onehot: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, *spatial) - network outputs (logits), 2D or 3D
            target: (B, 1, *spatial) or (B, *spatial) - class indices
            seg_onehot: (B, C, *spatial) - one-hot encoded target (optional, for efficiency)

        Returns:
            Weighted cross-entropy loss
        """
        # Normalise target to (B, 1, *spatial) so 2D and 3D share one code path.
        if target.ndim == net_output.ndim - 1:
            target = target.unsqueeze(1)

        # Handle ignore label BEFORE one-hot encoding (replace with 0 temporarily)
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target_safe = torch.where(mask, target, 0)
        else:
            mask = None
            target_safe = target

        if seg_onehot is None:
            seg_onehot = _foreground_onehot(target_safe, net_output.shape[1], net_output)

        boundary = _morphological_boundary_mask(
            seg_onehot, self.boundary_radius, self.inplane_only
        )

        # Compute weights: 1.0 base + (boundary_weight - 1.0) on boundaries.
        # Collapse across classes to get a per-voxel weight.
        per_voxel_boundary = boundary.max(dim=1, keepdim=True)[0]  # (B, 1, *spatial)
        weights = (1.0 + (self.boundary_weight - 1.0) * per_voxel_boundary)[:, 0]

        ce_loss = self.ce(net_output, target_safe[:, 0].long())  # (B, *spatial)

        if mask is not None:
            m = mask[:, 0].to(ce_loss.dtype)
            return (ce_loss * weights * m).sum() / torch.clip(m.sum(), min=1e-8)
        return (ce_loss * weights).mean()


class DC_and_BoundaryCE_loss(nn.Module):
    """
    weight_dice * SoftDice + weight_ce * BoundaryCE
    
    Replaces standard CE with BoundaryCELoss. DC term is unchanged (global overlap).
    This is non-redundant: Dice measures volumetric overlap; BoundaryCE measures
    spatially-weighted per-voxel log-likelihood.
    
    Args:
        soft_dice_kwargs: kwargs for SoftDiceLoss
        ce_kwargs: kwargs for BoundaryCELoss (boundary_radius, boundary_weight)
        weight_ce: weight for cross-entropy loss (default 1)
        weight_dice: weight for dice loss (default 1)
        ignore_label: label to ignore (default None)
        dice_class: dice loss class to use (default MemoryEfficientSoftDiceLoss)
    """
    
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        
        # Extract boundary-specific kwargs
        boundary_radius = ce_kwargs.get('boundary_radius', 3)
        boundary_weight = ce_kwargs.get('boundary_weight', 3.0)
        inplane_only = ce_kwargs.get('inplane_only', False)

        self.ce = BoundaryCELoss(
            boundary_radius=boundary_radius,
            boundary_weight=boundary_weight,
            ignore_index=ignore_label if ignore_label is not None else -100,
            inplane_only=inplane_only,
        )

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, *spatial) - logits, 2D or 3D
            target: (B, 1, *spatial) - class indices

        Returns:
            Combined loss
        """
        # Handle ignore label for Dice
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        # Compute Dice loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        # BoundaryCELoss builds its own one-hot; it is dimension-agnostic.
        if self.weight_ce != 0 and (self.ignore_label is None or (num_fg is not None and num_fg > 0)):
            ce_loss_val = self.ce(net_output, target)
        else:
            ce_loss_val = torch.tensor(0.0, device=net_output.device, dtype=net_output.dtype)

        result: torch.Tensor = self.weight_ce * ce_loss_val + self.weight_dice * dc_loss
        return result

class SurfaceDiceLoss(nn.Module):
    """
    Soft Dice computed only within the morphological boundary shell of each class.
    High variance due to sparse boundary region — smooth=1e-5 and batch_dice mitigate.
    
    Args:
        boundary_radius: shell radius in voxels (default 3)
        smooth: Dice smoothing factor (default 1e-5)
        batch_dice: aggregate dice across batch (default True)
        do_bg: include background class (default False)
        apply_nonlin: softmax or sigmoid function to apply to net_output
    """
    
    def __init__(self, boundary_radius: int = 3, smooth: float = 1e-5,
                 batch_dice: bool = True, do_bg: bool = False,
                 apply_nonlin=None, inplane_only: bool = False):
        super().__init__()
        self.boundary_radius = boundary_radius
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin
        self.inplane_only = inplane_only
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                seg_onehot: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, X, Y, Z) - network outputs (logits)
            target: (B, 1, X, Y, Z) or (B, X, Y, Z) - class indices  
            seg_onehot: (B, C, X, Y, Z) - one-hot encoded target (optional, for efficiency)
            loss_mask: (B, 1, X, Y, Z) - mask for valid voxels (optional)
            
        Returns:
            Surface Dice loss (lower is better)
        """
        # Convert target to one-hot if not provided
        if seg_onehot is None:
            num_classes = net_output.shape[1]
            target_flat = target.squeeze(1) if target.dim() == 5 else target
            if target_flat.ndim < net_output.ndim:
                target_flat = target_flat.view((target_flat.shape[0], 1, *target_flat.shape[1:]))

            # Handle ignore label BEFORE one-hot encoding
            if loss_mask is not None:
                target_flat_safe = torch.where(loss_mask.bool(), target_flat, torch.zeros_like(target_flat))
            else:
                target_flat_safe = target_flat

            # Use scatter_ into a (B, C, ...) bool tensor — same approach as MemoryEfficientSoftDiceLoss.
            # This avoids the double-strip bug that arises from F.one_hot with num_classes+1.
            seg_onehot = torch.zeros(
                (target_flat_safe.shape[0], num_classes, *target_flat_safe.shape[2:]),
                device=net_output.device, dtype=torch.float32
            )
            seg_onehot.scatter_(1, target_flat_safe.long(), 1)

            if not self.do_bg:
                seg_onehot = seg_onehot[:, 1:]  # Remove background class (channel 0) — only once
        elif not self.do_bg and seg_onehot.shape[1] > 1:
            seg_onehot = seg_onehot[:, 1:]  # Remove background class
        
        # Compute boundary mask
        boundary = _morphological_boundary_mask(
            seg_onehot, self.boundary_radius, self.inplane_only
        )

        # Honour the ignore label: zero the boundary region on ignored voxels so they
        # contribute nothing to tp/fp/fn below. Without this, ignored voxels are treated
        # as background (target_flat_safe set them to 0 above) and any foreground predicted
        # there is penalised as a false positive — which, with sparse (every-other-slice)
        # annotation, trains the network to blank the un-annotated slices ("comb" output).
        # loss_mask is (B, 1, *spatial) and broadcasts over the class dimension.
        if loss_mask is not None:
            boundary = boundary * loss_mask.to(boundary.dtype)

        # Apply nonlinearity (softmax)
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        
        # If not do_bg, remove background channel from net_output too
        if not self.do_bg:
            net_output = net_output[:, 1:]
        
        # Mask both prediction and target by boundary
        # This restricts Dice computation to boundary voxels only
        pred_boundary = net_output * boundary
        target_boundary = seg_onehot * boundary
        
        # Flatten for Dice computation
        if self.batch_dice:
            # Aggregate over batch, compute dice per class
            pred_boundary = pred_boundary.view(pred_boundary.shape[0], pred_boundary.shape[1], -1)
            target_boundary = target_boundary.view(target_boundary.shape[0], target_boundary.shape[1], -1)
            # Sum over batch and spatial
            tp = (pred_boundary * target_boundary).sum(dim=(0, 2))
            fp = (pred_boundary * (1 - target_boundary)).sum(dim=(0, 2))
            fn = ((1 - pred_boundary) * target_boundary).sum(dim=(0, 2))
        else:
            # Per-sample dice
            pred_boundary = pred_boundary.view(pred_boundary.shape[0], pred_boundary.shape[1], -1)
            target_boundary = target_boundary.view(target_boundary.shape[0], target_boundary.shape[1], -1)
            tp = (pred_boundary * target_boundary).sum(dim=2)
            fp = (pred_boundary * (1 - target_boundary)).sum(dim=2)
            fn = ((1 - pred_boundary) * target_boundary).sum(dim=2)
        
        # Compute Dice
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        
        # Return loss (1 - dice)
        return (1 - dice).mean()


class DC_and_SurfaceDice_BCE_loss(nn.Module):
    """
    weight_dice * SurfaceDice + weight_ce * BCE (or CE)
    
    Combines boundary-restricted Dice loss with standard (non-boundary-weighted) 
    cross-entropy. Surface Dice focuses on boundary accuracy while CE provides
    global per-voxel supervision.
    
    This differs from DC_and_BoundaryCE_loss which uses standard Dice + boundary-weighted CE.
    Here we use surface-restricted Dice + normal CE.
    
    Args:
        soft_dice_kwargs: kwargs for SurfaceDiceLoss
        ce_kwargs: kwargs for RobustCrossEntropyLoss
        weight_ce: weight for cross-entropy loss (default 1)
        weight_dice: weight for dice loss (default 1)
        ignore_label: label to ignore (default None)
    """
    
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, 
                 ignore_label=None):
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        
        # Extract boundary-specific kwargs for SurfaceDice
        boundary_radius = soft_dice_kwargs.pop('boundary_radius', 3)
        inplane_only = soft_dice_kwargs.pop('inplane_only', False)

        # SurfaceDice uses boundary mask internally
        self.dc = SurfaceDiceLoss(
            boundary_radius=boundary_radius,
            smooth=soft_dice_kwargs.get('smooth', 1e-5),
            batch_dice=soft_dice_kwargs.get('batch_dice', True),
            do_bg=soft_dice_kwargs.get('do_bg', False),
            apply_nonlin=softmax_helper_dim1,
            inplane_only=inplane_only,
        )
        
        # Normal CE (not boundary weighted)
        self.ce = RobustCrossEntropyLoss(reduction='mean', ignore_index=ignore_label if ignore_label is not None else -100)
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, X, Y, Z) - logits
            target: (B, 1, X, Y, Z) - class indices
            
        Returns:
            Combined loss
        """
        # Handle ignore label for loss mask
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None
        
        # Compute Surface Dice loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        
        # Compute normal CE loss
        ce_loss: torch.Tensor = torch.tensor(0.0, device=net_output.device, dtype=net_output.dtype)
        if self.weight_ce != 0 and (self.ignore_label is None or (num_fg is not None and num_fg > 0)):
            # Keep ignored voxels at the ignore label so RobustCrossEntropyLoss(ignore_index)
            # drops them. Previously they were relabelled to background (0), which defeated
            # ignore_index and penalised the (un-annotated) ignored slices as background.
            ce_target = target.squeeze(1).long()
            ce_loss = self.ce(net_output, ce_target)
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
