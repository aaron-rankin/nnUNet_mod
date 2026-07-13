"""DA5 + no-Z-flip trainers with an *in-plane* boundary shell.

Why these exist
---------------
The original BoundaryCE / SurfaceDice trainers compute their morphological boundary shell with
an isotropic voxel-radius 3D structuring element. DS601 annotates only even z-slices; odd slices
carry the ignore label, and every loss maps ignore -> background before one-hot. A 3D erosion with
any z-radius >= 1 therefore removes *every* foreground voxel, so `boundary == dilated` and the
shell covers 100% of the structure. Under that degeneracy:

  * BoundaryCE reduces to foreground-weighted cross-entropy (a class-balance knob),
  * SurfaceDice reduces to approximately plain Dice.

Neither carries boundary information, which is why the original 3D grid showed no effect.

These trainers set ``inplane_only=True``, restricting the structuring element to (1, k, k). Erosion
is then well-defined and the shell is a genuine 1.2 mm (r=3) / 2.0 mm (r=5) rim covering ~18-40% of
the foreground. This is also the physically correct choice: at [4.0, 0.406, 0.406] mm spacing an
isotropic radius of 5 spans 20 mm through-plane against 2 mm in-plane.

The same classes serve 2D and 3D configurations — `_morphological_boundary_mask` dispatches on
tensor rank, and in 2D there is no z-axis to erode across, so the shell is well-defined either way.

Note: the shell radius is defined in voxels at each deep-supervision scale, so it corresponds to a
physically larger rim at coarser scales. That behaviour is inherited from the original trainers and
is left unchanged so the corrected runs stay comparable to them along every other axis.

Run with:
    nnUNetv2_train 601 3d_fullres FOLD -tr nnUNetTrainerDA5_NoZFlip_BoundaryCEW5R5_IP -p nnUNetResEncUNetMPlans
    nnUNetv2_train 601 2d         FOLD -tr nnUNetTrainerDA5_NoZFlip_BoundaryCEW5R5_IP -p nnUNetResEncUNetMPlans
"""

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.boundary_losses import (
    DC_and_BoundaryCE_loss,
    DC_and_SurfaceDice_BCE_loss,
)


class _NoZFlipMixin:
    """Drop craniocaudal mirroring: hamstring morphology varies along the thigh axis.

    Only meaningful for 3D patches; 2D patches are axial slices with no z-axis, so the DA5
    defaults already contain no z-mirror.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        if len(self.configuration_manager.patch_size) == 3:
            mirror_axes = (1, 2)
        self.inference_allowed_mirroring_axes = mirror_axes

        self.print_to_log_file(
            f'{type(self).__name__}: mirror axes {mirror_axes} (no Z-mirror)'
        )
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


def _wrap_deep_supervision(trainer, loss):
    if not trainer.enable_deep_supervision:
        return loss
    deep_supervision_scales = trainer._get_deep_supervision_scales()
    weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    if trainer.is_ddp and not trainer._do_i_compile():
        weights[-1] = 1e-6
    else:
        weights[-1] = 0
    weights = weights / weights.sum()
    return DeepSupervisionWrapper(loss, weights)


# ---------------------------------------------------------------------------
# BoundaryCE family: standard Dice + distance-reweighted CE
# ---------------------------------------------------------------------------

class _BoundaryCE_IP_Base(_NoZFlipMixin, nnUNetTrainerDA5):
    BOUNDARY_RADIUS = 5
    BOUNDARY_WEIGHT = 3.0

    def _build_loss(self):
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp,
        }
        ce_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'boundary_weight': self.BOUNDARY_WEIGHT,
            'inplane_only': True,
        }

        loss = DC_and_BoundaryCE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        self.print_to_log_file(
            f'{type(self).__name__}: DC_and_BoundaryCE_loss, in-plane shell, '
            f'radius={self.BOUNDARY_RADIUS}, weight={self.BOUNDARY_WEIGHT}'
        )
        return _wrap_deep_supervision(self, loss)


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW3R3_IP(_BoundaryCE_IP_Base):
    BOUNDARY_RADIUS = 3
    BOUNDARY_WEIGHT = 3.0


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW3R5_IP(_BoundaryCE_IP_Base):
    BOUNDARY_RADIUS = 5
    BOUNDARY_WEIGHT = 3.0


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW5R3_IP(_BoundaryCE_IP_Base):
    BOUNDARY_RADIUS = 3
    BOUNDARY_WEIGHT = 5.0


class nnUNetTrainerDA5_NoZFlip_BoundaryCEW5R5_IP(_BoundaryCE_IP_Base):
    BOUNDARY_RADIUS = 5
    BOUNDARY_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# SurfaceDice family: shell-restricted Dice + standard CE
# ---------------------------------------------------------------------------

class _SurfaceDice_IP_Base(_NoZFlipMixin, nnUNetTrainerDA5):
    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 5.0
    WEIGHT_CE = 1.0

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError(
                f'{type(self).__name__} expects class-based labels; this dataset uses regions. '
                'The surface-restricted Dice term has no defined behaviour there.'
            )

        soft_dice_kwargs = {
            'boundary_radius': self.BOUNDARY_RADIUS,
            'inplane_only': True,
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
        }

        loss = DC_and_SurfaceDice_BCE_loss(
            soft_dice_kwargs,
            {},
            weight_ce=self.WEIGHT_CE,
            weight_dice=self.WEIGHT_DICE,
            ignore_label=self.label_manager.ignore_label,
        )

        self.print_to_log_file(
            f'{type(self).__name__}: DC_and_SurfaceDice_BCE_loss, in-plane shell, '
            f'radius={self.BOUNDARY_RADIUS}, weight_dice={self.WEIGHT_DICE}, '
            f'weight_ce={self.WEIGHT_CE}'
        )
        return _wrap_deep_supervision(self, loss)


class nnUNetTrainerDA5_NoZFlip_SurfaceDiceW1R5_IP(_SurfaceDice_IP_Base):
    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 1.0


class nnUNetTrainerDA5_NoZFlip_SurfaceDiceW3R5_IP(_SurfaceDice_IP_Base):
    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 3.0


class nnUNetTrainerDA5_NoZFlip_SurfaceDiceW5R5_IP(_SurfaceDice_IP_Base):
    BOUNDARY_RADIUS = 5
    WEIGHT_DICE = 5.0
