from typing import List, Union, Tuple

import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSurfaceDiceBCE import nnUNetTrainerSurfaceDice_W5_Radius5
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.boundary_losses import DC_and_SurfaceDice_BCE_loss


class nnUNetTrainerDA5_SurfaceDiceW5R5(nnUNetTrainerDA5):
    """
    Combined trainer using DA5 augmentation + SurfaceDice (W5 R5) loss.
    
    This combines the best performing configuration from experiments:
    - DA5 augmentation pipeline (B1 field inhomogeneity + CutOut regularization)
    - SurfaceDice loss with weight=5, radius=5 (best boundary Dice configuration)
    - Includes standard Z-mirror (keep default mirroring)
    
    Configuration based on experiments showing SurfaceDice_W5_R5 achieves 
    highest validation Dice (~0.916 on fold1).
    
    Run with:
        nnUNetv2_train 502 3d_fullres FOLD -tr nnUNetTrainerDA5_SurfaceDiceW5R5 -p nnUNetResEncUNetMPlans
    
    Key components:
    1. DA5 adds B1 field inhomogeneity simulation (not in default pipeline)
    2. BlankRectangleTransform (CutOut) for small dataset regularization  
    3. SurfaceDice W5 R5: 5x Dice weight, 5 voxel boundary radius
       - Focuses on boundary voxels only (unlike standard Dice)
       - Higher weight (5x) pushes boundary accuracy
       - Larger radius (5) captures thicker boundary shell
    """
    
    # SurfaceDice configuration (best performing from experiments)
    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 5.0  # Weight for SurfaceDice loss
    WEIGHT_CE = 1.0    # Weight for Cross-Entropy loss
    
    def _build_loss(self):
        """
        Build SurfaceDice loss with weight=5, radius=5 wrapped in DeepSupervisionWrapper.
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
        
        self.print_to_log_file(
            f'DA5_SurfaceDiceW5R5: Using DC_and_SurfaceDice_BCE_loss with '
            f'weight_dice={self.WEIGHT_DICE}, weight_ce={self.WEIGHT_CE}, '
            f'boundary_radius={self.BOUNDARY_RADIUS}'
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


class nnUNetTrainerDA5_SurfaceDiceW5R5_NoMirror(nnUNetTrainerDA5_SurfaceDiceW5R5):
    """
    Variant of DA5 + SurfaceDiceW5R5 with NO Z-mirror.
    
    Tests whether removing Z-mirror (keeping only XY mirroring) improves
    anatomical consistency for thigh MRI.
    """
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Override to restrict mirroring to in-plane only (no Z-axis flip).
        """
        # Get DA5 base configuration
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Override: only allow mirroring on axes 1, 2 (in-plane), not Z (axis 0)
        # This is anatomically correct for thigh MRI - muscles change along thigh length
        if len(mirror_axes) > 2:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes
        
        self.print_to_log_file(f'DA5_SurfaceDiceW5R5_NoMirror: Mirror axes restricted to {mirror_axes} (no Z-mirror)')
        
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerDA5_NoZFlip_SurfaceDiceW5R5(nnUNetTrainerDA5_SurfaceDiceW5R5):
    """
    DA5 + SurfaceDiceW5R5 without z-axis (superior-inferior) mirroring.

    Use when flipping along the z-axis is anatomically invalid (e.g. MSK, spine, brain
    with clear superior/inferior). Keeps the SurfaceDice W5R5 loss and DA5 pipeline
    from the parent trainer but removes z-flip augmentation.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Override to restrict mirroring to in-plane only (no Z-axis flip).
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'DA5_NoZFlip_SurfaceDiceW5R5: Mirror axes restricted to {mirror_axes} (no Z-mirror)'
        )

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerDA5_SurfaceDiceW5R5_Short(nnUNetTrainerDA5_SurfaceDiceW5R5):
    """
    DA5 + SurfaceDiceW5R5 with shorter training budget (250 epochs).
    Use if full 500 epochs shows diminishing returns.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerDA5_SurfaceDiceW3R5(nnUNetTrainerDA5):
    """
    Combined trainer using DA5 augmentation + SurfaceDice (W3 R5) loss.

    Same as nnUNetTrainerDA5_SurfaceDiceW5R5 but with a lower Dice weight
    (3x instead of 5x), keeping the wider 5-voxel boundary radius.

    Run with:
        nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5_SurfaceDiceW3R5 -p nnUNetResEncUNetMPlans
    """

    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 3.0  # Weight for SurfaceDice loss
    WEIGHT_CE = 1.0    # Weight for Cross-Entropy loss

    def _build_loss(self):
        """
        Build SurfaceDice loss with weight=3, radius=5 wrapped in DeepSupervisionWrapper.
        """
        soft_dice_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
        }

        ce_kwargs = {}

        if self.label_manager.has_regions:
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
            loss = DC_and_SurfaceDice_BCE_loss(
                soft_dice_kwargs,
                ce_kwargs,
                weight_ce=self.WEIGHT_CE,
                weight_dice=self.WEIGHT_DICE,
                ignore_label=self.label_manager.ignore_label
            )

        self.print_to_log_file(
            f'DA5_SurfaceDiceW3R5: Using DC_and_SurfaceDice_BCE_loss with '
            f'weight_dice={self.WEIGHT_DICE}, weight_ce={self.WEIGHT_CE}, '
            f'boundary_radius={self.BOUNDARY_RADIUS}'
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerDA5_NoZFlip_SurfaceDiceW3R5(nnUNetTrainerDA5_SurfaceDiceW3R5):
    """
    DA5 + SurfaceDiceW3R5 without z-axis (superior-inferior) mirroring.

    Use when flipping along the z-axis is anatomically invalid (e.g. MSK, spine, brain
    with clear superior/inferior). Keeps the SurfaceDice W3R5 loss and DA5 pipeline
    from the parent trainer but removes z-flip augmentation.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Override to restrict mirroring to in-plane only (no Z-axis flip).
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'DA5_NoZFlip_SurfaceDiceW3R5: Mirror axes restricted to {mirror_axes} (no Z-mirror)'
        )

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerDA5_SurfaceDiceW1R5(nnUNetTrainerDA5):
    """
    Combined trainer using DA5 augmentation + SurfaceDice (W1 R5) loss.

    Same as nnUNetTrainerDA5_SurfaceDiceW5R5 but with dice/CE weighted equally
    (1x instead of 5x), keeping the wider 5-voxel boundary radius. Matches the
    weighting used by nnUNetTrainerSurfaceDiceBCE_LargerRadius, recombined with DA5.

    Run with:
        nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5_SurfaceDiceW1R5 -p nnUNetResEncUNetMPlans
    """

    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 1.0  # Weight for SurfaceDice loss
    WEIGHT_CE = 1.0    # Weight for Cross-Entropy loss

    def _build_loss(self):
        """
        Build SurfaceDice loss with weight=1, radius=5 wrapped in DeepSupervisionWrapper.
        """
        soft_dice_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
        }

        ce_kwargs = {}

        if self.label_manager.has_regions:
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
            loss = DC_and_SurfaceDice_BCE_loss(
                soft_dice_kwargs,
                ce_kwargs,
                weight_ce=self.WEIGHT_CE,
                weight_dice=self.WEIGHT_DICE,
                ignore_label=self.label_manager.ignore_label
            )

        self.print_to_log_file(
            f'DA5_SurfaceDiceW1R5: Using DC_and_SurfaceDice_BCE_loss with '
            f'weight_dice={self.WEIGHT_DICE}, weight_ce={self.WEIGHT_CE}, '
            f'boundary_radius={self.BOUNDARY_RADIUS}'
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerDA5_NoZFlip_SurfaceDiceW1R5(nnUNetTrainerDA5_SurfaceDiceW1R5):
    """
    DA5 + SurfaceDiceW1R5 without z-axis (superior-inferior) mirroring.

    Use when flipping along the z-axis is anatomically invalid (e.g. MSK, spine, brain
    with clear superior/inferior). Keeps the SurfaceDice W1R5 loss and DA5 pipeline
    from the parent trainer but removes z-flip augmentation.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Override to restrict mirroring to in-plane only (no Z-axis flip).
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'DA5_NoZFlip_SurfaceDiceW1R5: Mirror axes restricted to {mirror_axes} (no Z-mirror)'
        )

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
