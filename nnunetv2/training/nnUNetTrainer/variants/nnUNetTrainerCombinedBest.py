from typing import List, Union, Tuple

import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerCombinedBest(nnUNetTrainerDA5):
    """
    Combined trainer using best elements from experiments:
    - DA5 augmentation pipeline (B1 field inhomogeneity + CutOut regularization)
    - No Z-mirror (anatomically correct for thigh MRI)
    - BoundaryCE loss (radius=5, weight=3) for BF-SH boundary improvement
    
    Recommended configuration for hamstring muscle segmentation (Dataset502).
    Combines strongest regularization (DA5), anatomically correct augmentation,
    and boundary-aware loss function.
    
    Run with:
        nnUNetv2_train 502 3d_fullres FOLD -tr nnUNetTrainerCombinedBest -p nnUNetResEncUNetMPlans
    
    Key improvements over baseline:
    1. DA5 adds B1 field inhomogeneity simulation (not in default pipeline)
    2. BlankRectangleTransform (CutOut) for small dataset regularization  
    3. No Z-mirror: anatomically correct (thigh muscles change along length)
    4. BoundaryCE: focuses gradients on boundary voxels, helps BF-SH (weakest class)
    """
    
    # BoundaryCE configuration
    BOUNDARY_RADIUS = 5
    BOUNDARY_WEIGHT = 3.0
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Override to use DA5 configuration but restrict mirroring to in-plane only (no Z-axis flip).
        """
        # Get DA5 base configuration
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Override: only allow mirroring on axes 1, 2 (in-plane), not Z (axis 0)
        # This is anatomically correct for thigh MRI - muscles change along thigh length
        mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes
        
        self.print_to_log_file(f'CombinedBest: Mirror axes restricted to {mirror_axes} (no Z-mirror)')
        
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    def _build_loss(self):
        """
        Build BoundaryCE loss with radius=5, weight=3.0.
        Wrap in DeepSupervisionWrapper for multi-scale training.
        """
        from nnunetv2.training.loss.boundary_losses import DC_and_BoundaryCE_loss
        
        # Configure dice kwargs using the same approach as base trainer
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }
        
        # BoundaryCE kwargs
        ce_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'boundary_weight': self.BOUNDARY_WEIGHT
        }
        
        # Create combined loss
        loss = DC_and_BoundaryCE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )
        
        # Compile loss if needed (matching base trainer)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        
        # Wrap in deep supervision (matching base trainer approach)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            
            # Normalize weights so they sum to 1
            weights = weights / weights.sum()
            
            loss = DeepSupervisionWrapper(loss, weights)
        
        self.print_to_log_file(
            f'CombinedBest: Using DC_and_BoundaryCE_loss with radius={self.BOUNDARY_RADIUS}, '
            f'weight={self.BOUNDARY_WEIGHT}'
        )
        
        return loss


class nnUNetTrainerCombinedBest_HighBoundaryWeight(nnUNetTrainerCombinedBest):
    """
    Variant with higher boundary weight (5.0 instead of 3.0).
    Run only if standard weight shows improvement but could benefit from more boundary focus.
    """
    BOUNDARY_WEIGHT = 5.0


class nnUNetTrainerCombinedBest_Short(nnUNetTrainerCombinedBest):
    """
    Combined trainer with shorter training budget (250 epochs).
    Use if full 500 epochs shows diminishing returns.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
