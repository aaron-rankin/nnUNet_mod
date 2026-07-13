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
from nnunetv2.training.loss.boundary_losses import DC_and_BoundaryCE_loss


class nnUNetTrainerDA5_BoundaryCEW3R3(nnUNetTrainerDA5):
    """
    DA5 augmentation + standard Dice combined with BoundaryCE (radius=3, weight=3).

    Same boundary_weight as nnUNetTrainerDA5_BoundaryCEW3R5 but with a thinner
    boundary shell (radius=3 instead of 5), to test the radius dimension of the
    sweep rather than just weight.

    Run with:
        nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5_BoundaryCEW3R3 -p nnUNetResEncUNetMPlans
    """

    BOUNDARY_RADIUS = 3
    BOUNDARY_WEIGHT = 3.0

    def _build_loss(self):
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        ce_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'boundary_weight': self.BOUNDARY_WEIGHT
        }

        loss = DC_and_BoundaryCE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(loss, weights)

        self.print_to_log_file(
            f'DA5_BoundaryCEW3R3: Using DC_and_BoundaryCE_loss with radius={self.BOUNDARY_RADIUS}, '
            f'weight={self.BOUNDARY_WEIGHT}'
        )

        return loss


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW3R3(nnUNetTrainerDA5_BoundaryCEW3R3):
    """
    DA5 + BoundaryCEW3R3 without z-axis (superior-inferior) mirroring.

    Use when flipping along the z-axis is anatomically invalid (e.g. MSK, spine, brain
    with clear superior/inferior). Keeps the BoundaryCE W3R3 loss and DA5 pipeline
    from the parent trainer but removes z-flip augmentation.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'DA5_NoZFlip_BoundaryCEW3R3: Mirror axes restricted to {mirror_axes} (no Z-mirror)'
        )

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerDA5_BoundaryCEW3R5(nnUNetTrainerDA5):
    """
    DA5 augmentation + standard Dice combined with BoundaryCE (radius=5, weight=3).

    Matches the boundary weight used by nnUNetTrainerCombinedBest, but keeps the
    Z-mirroring behavior of plain DA5 (see the NoZFlip variant below for that combo).

    Run with:
        nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5_BoundaryCEW3R5 -p nnUNetResEncUNetMPlans
    """

    BOUNDARY_RADIUS = 5
    BOUNDARY_WEIGHT = 3.0

    def _build_loss(self):
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        ce_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'boundary_weight': self.BOUNDARY_WEIGHT
        }

        loss = DC_and_BoundaryCE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(loss, weights)

        self.print_to_log_file(
            f'DA5_BoundaryCEW3R5: Using DC_and_BoundaryCE_loss with radius={self.BOUNDARY_RADIUS}, '
            f'weight={self.BOUNDARY_WEIGHT}'
        )

        return loss


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW3R5(nnUNetTrainerDA5_BoundaryCEW3R5):
    """
    DA5 + BoundaryCEW3R5 without z-axis (superior-inferior) mirroring.

    Use when flipping along the z-axis is anatomically invalid (e.g. MSK, spine, brain
    with clear superior/inferior). Keeps the BoundaryCE W3R5 loss and DA5 pipeline
    from the parent trainer but removes z-flip augmentation.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'DA5_NoZFlip_BoundaryCEW3R5: Mirror axes restricted to {mirror_axes} (no Z-mirror)'
        )

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerDA5_BoundaryCEW5R5(nnUNetTrainerDA5):
    """
    DA5 augmentation + standard Dice combined with BoundaryCE (radius=5, weight=5).

    Highest boundary weight in the sweep, testing whether pushing BoundaryCE further
    than nnUNetTrainerCombinedBest's weight=3 helps.

    Run with:
        nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5_BoundaryCEW5R5 -p nnUNetResEncUNetMPlans
    """

    BOUNDARY_RADIUS = 5
    BOUNDARY_WEIGHT = 5.0

    def _build_loss(self):
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        ce_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'boundary_weight': self.BOUNDARY_WEIGHT
        }

        loss = DC_and_BoundaryCE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(loss, weights)

        self.print_to_log_file(
            f'DA5_BoundaryCEW5R5: Using DC_and_BoundaryCE_loss with radius={self.BOUNDARY_RADIUS}, '
            f'weight={self.BOUNDARY_WEIGHT}'
        )

        return loss


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW5R5(nnUNetTrainerDA5_BoundaryCEW5R5):
    """
    DA5 + BoundaryCEW5R5 without z-axis (superior-inferior) mirroring.

    Use when flipping along the z-axis is anatomically invalid (e.g. MSK, spine, brain
    with clear superior/inferior). Keeps the BoundaryCE W5R5 loss and DA5 pipeline
    from the parent trainer but removes z-flip augmentation.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'DA5_NoZFlip_BoundaryCEW5R5: Mirror axes restricted to {mirror_axes} (no Z-mirror)'
        )

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
