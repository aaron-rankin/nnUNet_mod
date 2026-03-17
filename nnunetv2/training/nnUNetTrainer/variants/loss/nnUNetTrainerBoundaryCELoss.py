import numpy as np
import torch

from nnunetv2.training.loss.boundary_losses import DC_and_BoundaryCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerBoundaryCE(nnUNetTrainer):
    """
    Trainer using DC + BoundaryCE loss with boundary_weight=3.0.
    
    BoundaryCE shifts gradient mass toward boundary voxels without abandoning
    interior signal. This is particularly beneficial for classes with high
    surface-to-volume ratio like BF-SH.
    """
    
    BOUNDARY_RADIUS = 3
    BOUNDARY_WEIGHT = 3.0
    
    def _build_loss(self):
        """
        Build DC_and_BoundaryCE_loss wrapped in DeepSupervisionWrapper.
        """
        # Configure dice kwargs
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }
        
        # Configure CE kwargs with boundary parameters
        ce_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'boundary_weight': self.BOUNDARY_WEIGHT
        }
        
        # Create the loss
        if self.label_manager.has_regions:
            # For region-based labels, use BCE instead of CE
            # Fall back to standard DC_and_BCE_loss for regions
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            loss = DC_and_BCE_loss(
                {},
                soft_dice_kwargs,
                weight_ce=1,
                weight_dice=1,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            # Standard class-based labels use BoundaryCE
            loss = DC_and_BoundaryCE_loss(
                soft_dice_kwargs,
                ce_kwargs,
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        
        # Wrap in deep supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # We give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            
            # We don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            
            # Now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss


class nnUNetTrainerBoundaryCEHighWeight(nnUNetTrainerBoundaryCE):
    """
    Trainer using DC + BoundaryCE loss with boundary_weight=5.0.
    
    This is run only if boundary_weight=3.0 shows improvement,
    to test whether pushing harder improves further.
    """
    
    BOUNDARY_WEIGHT = 5.0
