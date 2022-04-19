import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as opt

import torchvision.models as models
from lsq_utils import *
from dataset_utils import *
from general_utils import *
from toy_model import *


# =============================================================================
# Configurations
# =============================================================================
# Devices
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3, 4, 5'
device_ids = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))))
multi_gpu = len(device_ids) > 1
assert len(device_ids) > 0, 'No GPU available'

# Training Hyper-parameters
epochs = 100
train_batch_size = 128
val_batch_size = 512
learning_rate = 1e-4

# Datasets
# dataset_getter = get_CIFAR10_dataset
# dataset_name = 'CIFAR10'
dataset_getter = get_ImageNet_dataset
dataset_name = 'ImageNet'
# dataset_getter = get_MSTAR_dataset
# dataset_name = 'MSTAR'

# Model
# Toy Model
# model_getter = MyModel()
# model_name = 'MyModel'

# ResNet-18
model_getter = models.resnet18(pretrained=False)
# model_getter = models.resnet18(pretrained=False, num_classes=10)
model_name = 'ResNet18'

# MobileNet-V2
# model_getter = models.mobilenet_v2(pretrained=False)
# model_name = 'MobileNetV2'

# Model Path
# Toy Model on CIFAR10
# pretrained_model_load_path = './model/pretrained/0.57100_0.93640_MyModel_CIFAR10_B128_E25.pth'
# full_precision_model_save_path = f'./model/saved/FP32_{model_name}_{dataset_name}_B{train_batch_size}_E{epochs}.pth'
# quantized_model_save_path = f'./model/saved/W{weight_bit_width}A{activation_bit_width}{"_PC" if per_channel else ""}_{model_name}_{dataset_name}_B{train_batch_size}_E{epochs}.pth'

# ResNet-18 on ImageNet
pretrained_model_load_path = './model/pretrained/resnet18-f37072fd.pth'
full_precision_model_save_path = f'./model/saved/FP32_{model_name}_{dataset_name}_B{train_batch_size}_E{epochs}.pth'
quantized_model_save_path = f'./model/saved/W{weight_bit_width}A{activation_bit_width}{"_PC" if per_channel else ""}_{model_name}_{dataset_name}_B{train_batch_size}_E{epochs}.pth'

# Seed
random_seed = 1

# Logger
log = True
log_path = f'./logs/{model_name}.log'

# Save Checkpoints
save_checkpoints = True


def train(model, model_save_path=None):
    # Task Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = opt.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)
    # DataLoader
    train_loader, _ = dataset_getter(batch_size=train_batch_size * len(device_ids))

    # Training Loop
    for epoch in range(1, epochs + 1):
        total, top1_correct = 0, 0
        now = datetime.now()
    
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Process Bar (Loss & Train Accuracy)
            total += labels.size(0)
            top1_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_top1_accuracy = format(top1_correct / total, '.5f')

            log_info = get_log_info(
                datetime=now,
                batch=i + 1,
                total_batch=len(train_loader),
                epoch=epoch,
                total_epoch=epochs,
                show_bar=True,
                Loss=format(loss.item(), '.5f'),
                TrainTop1Acc=train_top1_accuracy,
            )
            print(log_info, end='\r', flush=True)

        # Validation
        val_top1_accuracy, val_top5_accuracy = eval(model)
        log_info += f', ValTop1Acc: {val_top1_accuracy}, ValTop5Acc: {val_top5_accuracy}'
        print(log_info)

        # Logger
        if log:
            if not os.path.exists(f'./logs'):
                os.makedirs(f'./logs')
            log_info = get_log_info(
                datetime=now,
                epoch=epoch,
                total_epoch=epochs,
                Loss=format(loss.item(), '.5f'),
                TrainTop1Acc=train_top1_accuracy,
                ValTop1Acc=val_top1_accuracy,
                ValTop5Acc=val_top5_accuracy,
            )
            with open(log_path, 'a') as f:
                f.write(log_info + '\n')
            plot_weight(model, model_name=f'W{weight_bit_width}A{activation_bit_width}{"_PC" if per_channel else ""}_{model_name}_E{epoch}')

        # Save Checkpoint
        if save_checkpoints:
            if not os.path.exists(f'./model/checkpoint/{model_name}'):
                os.makedirs(f'./model/checkpoint/{model_name}')
            torch.save(model.state_dict(), f'./model/checkpoint/{model_name}/{val_top1_accuracy}_{val_top5_accuracy}_{model_name}_{dataset_name}_B{train_batch_size}_E{epoch + 1}.pth')

    # Save Model
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
    return model


def eval(model, model_load_path=None, verbose=False):
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
        print(f'Model loaded from {model_load_path}')
    if model.state_dict() is None:
        raise ValueError('Model not loaded')
    
    model.eval()
    _, val_loader = dataset_getter(batch_size=val_batch_size)
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, top1_predicted = torch.max(outputs, 1)
            _, top5_predicted = torch.topk(outputs, 5)
            top1_correct += torch.sum(top1_predicted == labels).item()
            top5_correct += torch.sum(top5_predicted == labels.unsqueeze(1).repeat(1, 5)).item()
            total += labels.size(0)
            val_top1_accuracy = format(top1_correct / total, '.5f')
            val_top5_accuracy = format(top5_correct / total, '.5f')
            
            if verbose:
                log_info = get_log_info(
                    datetime=datetime.now(),
                    batch=i + 1,
                    total_batch=len(val_loader),
                    show_bar=True,
                    ValTop1Acc=val_top1_accuracy,
                    ValTop5Acc=val_top5_accuracy,
                )
                print(log_info, end='\r', flush=True)
        # Line Feed
        if verbose:
            print()

    return val_top1_accuracy, val_top5_accuracy


def main():
    # Set random seed
    torch.manual_seed(random_seed)
    
    # Get model
    fp_model = model_getter
    
    # Load pretrained model state dict
    print(f'Loading pretrained model from {pretrained_model_load_path}')
    fp_model.load_state_dict(torch.load(pretrained_model_load_path))
    
    # Analyze weight distribution of the pretrained model
    print('Analyzing weight distribution of the pretrained model ...')
    plot_weight(fp_model, model_name)

    # Find and replace the parameterized modules in the model to QAT modules
    print('Replacing the parameterized modules in the model to QAT modules ...')
    fq_model = find_and_replace(fp_model)
    
    # set model to CUDA devices
    if multi_gpu:
        fq_model = nn.DataParallel(fq_model, device_ids=device_ids)
        fq_model = fq_model.cuda(device_ids[0])
    else:
        fq_model = fq_model.cuda()
    
    # Quantization-Aware Training
    print('Training ...')
    fq_model = train(fq_model, model_save_path=full_precision_model_save_path)
    
    # Evaluation
    print('Evaluating ...')
    _, _ = eval(fq_model, model_load_path=None, verbose=True)

    # Quantize model to 8-bit (byte or char)
    print('Quantizing ...')
    rq_model = find_and_quantize(fq_model)

    # Save quantized model
    print('Saving quantized model ...')
    torch.save(rq_model.state_dict(), quantized_model_save_path)
    print(f'Quantized model saved to {quantized_model_save_path}')

    # Analyze weight distribution of the quantized model
    rq_model = rq_model.cpu()
    print('Analyzing weight distribution of the quantized model ...')
    plot_weight(rq_model, f'Quantized_W{weight_bit_width}A{activation_bit_width}_{model_name}')

    print('Done!')


if __name__ == '__main__':
    main()
