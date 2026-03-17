from copy import deepcopy
from pydoc import locate

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDropout05(nnUNetTrainer):
    DROPOUT_P = 0.05

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        arch_init_kwargs = deepcopy(arch_init_kwargs)
        conv_op = locate(arch_init_kwargs['conv_op'])
        if conv_op.__name__ == 'Conv2d':
            arch_init_kwargs['dropout_op'] = 'torch.nn.Dropout2d'
        else:
            arch_init_kwargs['dropout_op'] = 'torch.nn.Dropout3d'
        arch_init_kwargs['dropout_op_kwargs'] = {'p': nnUNetTrainerDropout05.DROPOUT_P, 'inplace': True}
        
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision)
        
        
class nnUNetTrainerDropout02(nnUNetTrainer):
    DROPOUT_P = 0.02

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        arch_init_kwargs = deepcopy(arch_init_kwargs)
        conv_op = locate(arch_init_kwargs['conv_op'])
        if conv_op.__name__ == 'Conv2d':
            arch_init_kwargs['dropout_op'] = 'torch.nn.Dropout2d'
        else:
            arch_init_kwargs['dropout_op'] = 'torch.nn.Dropout3d'
        arch_init_kwargs['dropout_op_kwargs'] = {'p': nnUNetTrainerDropout02.DROPOUT_P, 'inplace': True}
        
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision)
        
        