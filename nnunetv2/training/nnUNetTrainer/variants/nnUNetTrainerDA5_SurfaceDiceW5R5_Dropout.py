from typing import List, Union, Tuple

import numpy as np
import torch
import sys
from pathlib import Path
from copy import deepcopy

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.boundary_losses import DC_and_SurfaceDice_BCE_loss


class nnUNetTrainerDA5_SurfaceDiceW5R5_Dropout(nnUNetTrainerDA5):
    """
    Combined trainer: DA5 + SurfaceDice (W5 R5) + Dropout (p=0.02)
    
    Maximum regularization configuration combining:
    - DA5 augmentation (B1 field + CutOut)
    - SurfaceDice W5 R5 (best boundary Dice config)
    - Dropout p=0.02 (MC Dropout for uncertainty)
    
    Run with:
        nnUNetv2_train 502 3d_fullres FOLD -tr nnUNetTrainerDA5_SurfaceDiceW5R5_Dropout -p nnUNetResEncUNetMPlans
    """
    
    # SurfaceDice configuration
    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 5.0
    WEIGHT_CE = 1.0
    
    # Dropout configuration
    DROPOUT_P = 0.02
    
    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """
        Override to inject Dropout3d into the architecture.
        """
        from pydoc import locate
        
        # Deep copy to avoid modifying original
        kwargs = deepcopy(arch_init_kwargs)
        req_import = list(arch_init_kwargs_req_import)
        
        # Determine dropout class based on conv_op
        conv_op = kwargs.get('conv_op', '')
        if 'Conv3d' in str(conv_op):
            kwargs['dropout_op'] = 'torch.nn.Dropout3d'
        else:
            kwargs['dropout_op'] = 'torch.nn.Dropout2d'
        
        # Set dropout kwargs
        kwargs['dropout_op_kwargs'] = {'p': self.DROPOUT_P, 'inplace': True}
        
        # Add to required imports if not already there
        if 'dropout_op' not in req_import:
            req_import.append('dropout_op')
        
        # Call parent to build with modified kwargs
        return super().build_network_architecture(
            architecture_class_name, kwargs, req_import,
            num_input_channels, num_output_channels,
            enable_deep_supervision
        )
    
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
            f'DA5_SurfaceDiceW5R5_Dropout: Using DC_and_SurfaceDice_BCE_loss with '
            f'weight_dice={self.WEIGHT_DICE}, weight_ce={self.WEIGHT_CE}, '
            f'boundary_radius={self.BOUNDARY_RADIUS}, dropout_p={self.DROPOUT_P}'
        )
        
        # Wrap in deep supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # Exponential decay weights for deep supervision
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss


class nnUNetTrainerDA5_SurfaceDiceW5R5_Dropout_NoMirror(nnUNetTrainerDA5_SurfaceDiceW5R5_Dropout):
    """
    Variant with NO Z-mirror (anatomically correct for thigh MRI).
    """
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Override to restrict mirroring to in-plane only.
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Only allow mirroring on axes 1, 2 (in-plane), not Z (axis 0)
        if len(mirror_axes) > 2:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes
        
        self.print_to_log_file(
            f'DA5_SurfaceDiceW5R5_Dropout_NoMirror: Mirror axes={mirror_axes}'
        )
        
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerDA5_SurfaceDiceW5R5_Dropout_Short(nnUNetTrainerDA5_SurfaceDiceW5R5_Dropout):
    """
    Variant with shorter training (250 epochs).
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
