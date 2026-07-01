import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def _morphological_boundary_mask(seg_onehot: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """
    GPU morphological boundary shell.
    
    Args:
        seg_onehot: (B, C, X, Y, Z), float in [0, 1]
        radius: morphological shell radius in voxels
        
    Returns:
        Binary mask of same shape: 1 inside boundary shell, 0 elsewhere
    """
    k = 2 * radius + 1
    # Dilate: max pooling (foreground expands)
    dilated = F.max_pool3d(seg_onehot, kernel_size=k, stride=1, padding=radius)
    # Erode: min pooling = -max_pool(-x) (foreground shrinks)
    eroded = -F.max_pool3d(-seg_onehot, kernel_size=k, stride=1, padding=radius)
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
                 ignore_index: int = -100):
        super().__init__()
        self.boundary_radius = boundary_radius
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.ce = RobustCrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, 
                seg_onehot: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, X, Y, Z) - network outputs (logits)
            target: (B, 1, X, Y, Z) or (B, X, Y, Z) - class indices
            seg_onehot: (B, C, X, Y, Z) - one-hot encoded target (optional, for efficiency)
            
        Returns:
            Weighted cross-entropy loss
        """
        # Convert target to one-hot if not provided
        if seg_onehot is None:
            num_classes = net_output.shape[1]
            # target should be (B, X, Y, Z) or (B, 1, X, Y, Z)
            target_flat = target.squeeze(1) if target.dim() == 5 else target
            
            # Handle ignore label BEFORE one-hot encoding (replace with 0 temporarily)
            if self.ignore_index is not None:
                target_flat_safe = torch.where(target_flat == self.ignore_index, 0, target_flat)
            else:
                target_flat_safe = target_flat
            
            seg_onehot = F.one_hot(target_flat_safe.long(), num_classes=num_classes + 1)
            seg_onehot = seg_onehot[..., 1:].permute(0, 4, 1, 2, 3).float()  # Remove background
        
        # Compute boundary mask
        boundary = _morphological_boundary_mask(seg_onehot, self.boundary_radius)
        
        # Compute weights: 1.0 base + (boundary_weight - 1.0) on boundaries
        # We average across classes to get per-voxel weight
        per_voxel_boundary = boundary.max(dim=1, keepdim=True)[0]  # (B, 1, X, Y, Z)
        weights = 1.0 + (self.boundary_weight - 1.0) * per_voxel_boundary
        
        # Handle ignore label
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target_ce = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_ce = target
            mask = None
            num_fg = None
        
        # Compute per-voxel CE
        # target_ce should be (B, X, Y, Z) for CE
        if target_ce.dim() == 5:
            target_ce = target_ce.squeeze(1)
        ce_loss = self.ce(net_output, target_ce.long())  # (B, X, Y, Z)
        
        # Apply weights and reduce
        if self.ignore_index is not None and num_fg is not None and mask is not None:
            # mask is (B, 1, X, Y, Z), weights is (B, 1, X, Y, Z)
            mask_for_compute = mask.squeeze(1) if mask.dim() == 5 else mask
            weights_for_compute = weights.squeeze(1)
            weighted_loss = (ce_loss * weights_for_compute * mask_for_compute).sum() / torch.clip(mask_for_compute.sum(), min=1e-8)
        else:
            weights_for_compute = weights.squeeze(1)
            weighted_loss = (ce_loss * weights_for_compute).mean()
        
        return weighted_loss


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
        
        self.ce = BoundaryCELoss(
            boundary_radius=boundary_radius,
            boundary_weight=boundary_weight,
            ignore_index=ignore_label if ignore_label is not None else -100
        )
        
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, X, Y, Z) - logits
            target: (B, 1, X, Y, Z) - class indices
            
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
        
        # For CE, we need one-hot
        seg_onehot = None
        if self.weight_ce != 0 and (self.ignore_label is None or (num_fg is not None and num_fg > 0)):
            num_classes = net_output.shape[1]
            target_flat = target.squeeze(1) if target.dim() == 5 else target
            
            # Handle ignore label BEFORE one-hot encoding (replace with 0 temporarily)
            if self.ignore_label is not None:
                target_flat_safe = torch.where(target_flat == self.ignore_label, 0, target_flat)
            else:
                target_flat_safe = target_flat
            
            seg_onehot = F.one_hot(target_flat_safe.long(), num_classes=num_classes + 1)
            seg_onehot = seg_onehot[..., 1:].permute(0, 4, 1, 2, 3).float()
        
        ce_loss_val: torch.Tensor = self.ce(net_output, target, seg_onehot=seg_onehot) \
            if self.weight_ce != 0 and (self.ignore_label is None or (num_fg is not None and num_fg > 0)) else torch.tensor(0.0, device=net_output.device, dtype=net_output.dtype)
        
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
                 apply_nonlin=None):
        super().__init__()
        self.boundary_radius = boundary_radius
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin
    
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
        boundary = _morphological_boundary_mask(seg_onehot, self.boundary_radius)

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
        
        # SurfaceDice uses boundary mask internally
        self.dc = SurfaceDiceLoss(
            boundary_radius=boundary_radius,
            smooth=soft_dice_kwargs.get('smooth', 1e-5),
            batch_dice=soft_dice_kwargs.get('batch_dice', True),
            do_bg=soft_dice_kwargs.get('do_bg', False),
            apply_nonlin=softmax_helper_dim1
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
