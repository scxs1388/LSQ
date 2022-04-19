import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
from lsq_module import *


# Seaborn Style
sns.set(style='darkgrid')

module_name_mapping = {
    nn.Conv2d: 'conv2d',
    nn.Linear: 'linear',
    # nn.BatchNorm2d: 'bn',
    # nn.ReLU: 'relu',
    # nn.MaxPool2d: 'maxpool',
    # nn.Dropout: 'dropout',
    LSQ_Conv2d: 'lsq_conv2d',
    LSQ_Linear: 'lsq_linear',
}


def plot_weight(model, model_name):
    '''
    Plot the weights of the model.
    '''
    assert model.state_dict is not None, 'Model state dict has not been loaded yet.'
    assert model_name is not None, 'Model name has not been set.'

    save_dir = './figures'
    if not os.path.exists(os.path.join(save_dir, model_name, 'weight')):
        os.makedirs(os.path.join(save_dir, model_name, 'weight'))

    for name, module in model.named_modules():
        if type(module) in module_name_mapping.keys():
            print(f'Plotting weight of {name}')
            x = module.weight.detach().flatten().numpy()

            # Plot the histogram of the weights
            fig, ax = plt.subplots(1, 1, num='weight', figsize=(12, 12), sharex=True, sharey=True)
            palette = sns.color_palette("hls", 8)

            ax1 = sns.histplot(x, kde=True, bins=100, label=name, color=palette[5], edgecolor=palette[4])
            
            plt.savefig(os.path.join(save_dir, model_name, 'weight', f'{name}_{module_name_mapping[type(module)]}.png'))
            plt.clf()


def plot_activation(model, model_name, calibration_dataloader):
    '''
    Plot the activation of the model. Calibration data is required.
    not implemented yet.
    '''
    assert model.state_dict is not None, 'Model state dict has not been loaded yet.'
    assert model_name is not None, 'Model name has not been set.'
    assert calibration_dataloader is not None, 'Calibration dataloader has not been set.'
    return None
    # import torch.fx as fx

    # traced : fx.GraphModule = fx.symbolic_trace(model)
    # traced.graph.print_tabular()
    # return None


def get_log_info(datetime, batch=None, total_batch=None, epoch=None, total_epoch=None, show_bar=False, **kwargs):
    # timestamp
    log_info = f'{datetime.strftime("%Y-%m-%d %H:%M:%S")} - '
    # epochs
    if epoch is not None and total_epoch is not None:
        log_info += f'Epoch: [{epoch}/{total_epoch}], '
    # batches (process bar)
    if batch is not None and total_batch is not None:
        log_info += f'Iter: '
        if show_bar:
            bar_len = 25
            p = batch / total_batch
            log_info += f'[{"=" * int(p * bar_len)}{">" * (1 - int(p))}{" " * (bar_len - 1 + int(p) - int(p * bar_len))}]'
        log_info += f'[{batch}/{total_batch}]'
    # other info
    for i, (k, v) in enumerate(kwargs.items()):
        if i == 0 and batch is None:
            log_info += f'{k}: {v}'
        else:
            log_info += f', {k}: {v}'
    
    return log_info