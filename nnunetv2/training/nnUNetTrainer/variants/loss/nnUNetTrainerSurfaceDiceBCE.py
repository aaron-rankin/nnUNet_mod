import numpy as np
import torch

from nnunetv2.training.loss.boundary_losses import DC_and_SurfaceDice_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSurfaceDiceBCE(nnUNetTrainer):
    """
    Trainer using DC + SurfaceDice with normal BCE.
    
    SurfaceDice focuses gradient on boundary voxels only (unlike standard Dice 
    which covers all voxels). Combined with normal CE, this tests whether 
    boundary-restricted Dice signal improves BF-SH segmentation.
    
    This differs from nnUNetTrainerBoundaryCE which uses:
    - Standard Dice (all voxels) + Boundary-weighted CE (boundary voxels get higher weight)
    
    This trainer uses:
    - Surface Dice (boundary voxels only) + Normal CE (uniform per-voxel weight)
    """
    
    BOUNDARY_RADIUS = 3
    WEIGHT_DICE = 1  # Weight for SurfaceDice loss
    WEIGHT_CE = 1    # Weight for Cross-Entropy loss
    
    def _build_loss(self):
        """
        Build DC_and_SurfaceDice_BCE_loss wrapped in DeepSupervisionWrapper.
        """
        # Configure dice kwargs with boundary radius
        soft_dice_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
        }
        
        # Configure CE kwargs (normal CE, no boundary weighting)
        ce_kwargs = {}
        
        # Create the loss
        if self.label_manager.has_regions:
            # For region-based labels, fall back to standard DC_and_BCE_loss
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
            loss = DC_and_BCE_loss(
                {},
                {'batch_dice': self.configuration_manager.batch_dice, 
                 'do_bg': True, 
                 'smooth': 1e-5},
                weight_ce=1,
                weight_dice=1,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            # Standard class-based labels use SurfaceDice + normal CE
            loss = DC_and_SurfaceDice_BCE_loss(
                soft_dice_kwargs,
                ce_kwargs,
                weight_ce=self.WEIGHT_CE,
                weight_dice=self.WEIGHT_DICE,
                ignore_label=self.label_manager.ignore_label
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


class nnUNetTrainerSurfaceDiceBCE_LargerRadius(nnUNetTrainerSurfaceDiceBCE):
    """
    SurfaceDice + BCE with larger boundary radius (5 voxels instead of 3).
    
    This tests whether a thicker boundary shell improves results.
    """
    
    BOUNDARY_RADIUS = 5


class nnUNetTrainerSurfaceDice_Weight3(nnUNetTrainerSurfaceDiceBCE):
    """
    SurfaceDice + BCE with Dice weight = 3.0, CE weight = 1.0.
    
    This tests whether increasing the SurfaceDice gradient signal (3x vs 1x)
    improves boundary segmentation, particularly for BF-SH.
    
    Total loss = 3.0 * SurfaceDice + 1.0 * CE
    Boundary radius = 3 (default)
    """
    
    WEIGHT_DICE = 3.0
    WEIGHT_CE = 1.0


class nnUNetTrainerSurfaceDice_Weight5(nnUNetTrainerSurfaceDiceBCE):
    """
    SurfaceDice + BCE with Dice weight = 5.0, CE weight = 1.0.
    
    Tests a more aggressive weighting where SurfaceDice dominates (5x vs 1x).
    May improve boundary precision but risks overfitting to boundary voxels.
    
    Total loss = 5.0 * SurfaceDice + 1.0 * CE
    Boundary radius = 3 (default)
    """
    
    WEIGHT_DICE = 5.0
    WEIGHT_CE = 1.0


class nnUNetTrainerSurfaceDice_W3_Radius5(nnUNetTrainerSurfaceDice_Weight3):
    """
    SurfaceDice + BCE with Dice weight = 3.0 and boundary radius = 5.
    
    Tests whether combining higher Dice weight (3x) with expanded boundary
    region (5 voxels) improves BF-SH boundary segmentation.
    
    Total loss = 3.0 * SurfaceDice + 1.0 * CE
    Boundary radius = 5
    """
    
    BOUNDARY_RADIUS = 5


class nnUNetTrainerSurfaceDice_W5_Radius5(nnUNetTrainerSurfaceDice_Weight5):
    """
    SurfaceDice + BCE with Dice weight = 5.0 and boundary radius = 5.

    Tests aggressive Dice weighting (5x) combined with expanded boundary
    region (5 voxels) for maximum boundary focus on challenging cases.

    Total loss = 5.0 * SurfaceDice + 1.0 * CE
    Boundary radius = 5
    """

    BOUNDARY_RADIUS = 5


class nnUNetTrainerSurfaceDice_W5_Radius5_FixCheck(nnUNetTrainerSurfaceDice_W5_Radius5):
    """
    Short (50-epoch) run of W5_R5 used only to confirm the ignore-label fix in
    boundary_losses.py: with sparse (every-other-slice) annotation the previous
    loss produced "comb" predictions (blank odd slices). This should now fill the
    full volume. Writes to its own results dir so the original combed run is kept.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
