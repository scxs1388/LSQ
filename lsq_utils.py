import torch
import torch.nn as nn
from toy_model import *
from lsq_module import LSQ_Conv2d, LSQ_Linear


# Quantization Hyper-parameters
weight_bit_width = 4
activation_bit_width = 4
per_channel = True

quant_module_mapping = {
    nn.Conv2d: LSQ_Conv2d,
    nn.Linear: LSQ_Linear,
}


def replace(module):
    # Conv2d
    if isinstance(module, nn.Conv2d):
        new_module = LSQ_Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dalition=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            bit_width=[weight_bit_width, activation_bit_width],
            per_channel=per_channel
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
    # Linear
    if isinstance(module, nn.Linear):
        new_module = LSQ_Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            bit_width=[weight_bit_width, activation_bit_width]
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
    return new_module


def find_and_replace(model):
    '''
    find and replace the parameterized modules in the model to QAT modules
    '''
    def find(model):
        for name, module in model.named_children():
            if len(module._modules) == 0:
                if type(module) in quant_module_mapping.keys():
                    model._modules[name] = replace(module)
            else:
                quantized_module = find(module)
                model._modules[name] = quantized_module
        return model
    return find(model)


def find_and_quantize(model):
    '''
    find and quantize the parameterized modules in the model
    '''
    def find(model):
        for name, module in model.named_children():
            if type(module) in quant_module_mapping.values():
                module.weight.data = module.weight_quantizer.quantize(module.weight.data)
                model._modules[name] = module
            else:
                if len(module._modules) > 0:
                    quantized_module = find(module)
                    model._modules[name] = quantized_module
        return model
    return find(model)
