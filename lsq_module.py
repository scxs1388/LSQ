import torch
import torch.nn as nn
import torch.nn.functional as F


# def gradscale(x, scale, per_channel=False):
#     '''
#     implemented according to the paper: https://arxiv.org/abs/1902.08153
#     x: quantization step size
#     scale: gradient scale factor
#     '''
#     y_out = x
#     y_grad = x * scale
#     y = y_out.detach() - y_grad.detach() + y_grad
#     return y


# def roundpass(x):
#     '''
#     implemented according to the paper: https://arxiv.org/abs/1902.08153
#     x: input tensor
#     '''
#     y_out = torch.round(x)
#     y_grad = x
#     y = y_out.detach() - y_grad.detach() + y_grad
#     return y


# def quantize(v, s, p, is_activation=False, per_channel=False):
#     '''
#     implemented according to the paper: https://arxiv.org/abs/1902.08153
#     v: input tensor
#     s: step size, a learnable parameter specific to weight or activation layer being quantized
#     p: quantization bits of precision
#     is_activation: whether this is an activation layer or not
#     '''
#     if is_activation:
#         qn = 0
#         qp = 2**p - 1
#         grad_scale_factor = 1.0 / ((v.numel() * qp) ** 0.5)
#     else:
#         qn = -2 ** (p - 1)
#         qp = 2 ** (p - 1) - 1
#         if per_channel:
#             grad_scale_factor = 1.0 / ((v[0].numel() * qp) ** 0.5)
#         else:
#             grad_scale_factor = 1.0 / ((v.numel() * qp) ** 0.5)

#     # quantize
#     s = gradscale(s, grad_scale_factor)
#     if per_channel:
#         v = v / s.repeat(1, v[0].numel()).reshape(v.shape)
#         v = torch.clamp(v, qn, qp)
#         v_bar = torch.round(v)
#         v_hat = v_bar * s.repeat(1, v[0].numel()).reshape(v.shape)
#     else:
#         v = v / s
#         v = torch.clamp(v, qn, qp)
#         v_bar = roundpass(v)
#         v_hat = v_bar * s
#     return v_hat
    
#     v = torch.round(v)
#     v = v / (2 ** p - 1)
#     return v


class LSQ_Quantizer(nn.Module):
    def __init__(self, bit_width=8, is_activation=False, per_channel=False, out_channels=None):
        super(LSQ_Quantizer, self).__init__()
        assert bit_width in [2, 3, 4, 8], 'quantization bit width must be one of [2, 3, 4, 8]'
        assert not (is_activation and per_channel) , 'activation quantization must be per-layer quantization'
        assert (not per_channel) or (per_channel and out_channels), 'out_channels must be specified when per_channel is true'

        self.bit_width = bit_width
        self.is_activation = is_activation
        self.per_channel = per_channel
        if self.per_channel:
            self.step_size = nn.Parameter(torch.ones(out_channels, 1))
        else:
            self.step_size = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # if x.dtype == torch.float32:
        x = self.quantize(x)
        x = self.dequantize(x)
        return x

    def initialize_step_size(self, x):
        if self.per_channel:
            self.step_size.data.copy_(2 * x.abs().mean(dim=list(range(1, x.dim()))).reshape(-1, 1) / ((2 ** (self.bit_width - 1) - 1) ** 0.5))
        else:
            self.step_size.data.copy_(2 * x.abs().mean() / ((2 ** (self.bit_width - 1) - 1) ** 0.5))

    def gradscale(self, x, scale):
        '''
        implemented according to the paper: https://arxiv.org/abs/1902.08153
        x: quantization step size
        scale: gradient scale factor
        '''
        y_out = x
        y_grad = x * scale
        y = y_out.detach() - y_grad.detach() + y_grad
        return y

    def roundpass(self, x):
        '''
        implemented according to the paper: https://arxiv.org/abs/1902.08153
        x: input tensor
        '''
        y_out = torch.round(x)
        y_grad = x
        y = y_out.detach() - y_grad.detach() + y_grad
        return y

    def quantize(self, v):
        '''
        implemented according to the paper: https://arxiv.org/abs/1902.08153
        v: input tensor
        '''
        if self.is_activation:
            qn = 0
            qp = 2 ** self.bit_width - 1
            grad_scale_factor = 1.0 / ((v.numel() * qp) ** 0.5)
        else:
            qn = -2 ** (self.bit_width - 1)
            qp = 2 ** (self.bit_width - 1) - 1
            if self.per_channel:
                grad_scale_factor = 1.0 / ((v[0].numel() * qp) ** 0.5)
            else:
                grad_scale_factor = 1.0 / ((v.numel() * qp) ** 0.5)

        self.step_size.data = self.gradscale(self.step_size.data, grad_scale_factor)

        if self.per_channel:
            v = v / self.step_size.repeat(1, v[0].numel()).reshape(v.shape)
            v = torch.clamp(v, qn, qp)
            v_bar = torch.round(v)
            # v_hat = v_bar * self.step_size.repeat(1, v[0].numel()).reshape(v.shape)
        else:
            v = v / self.step_size
            v = torch.clamp(v, qn, qp)
            v_bar = self.roundpass(v)
            # v_hat = v_bar * self.step_size
        if self.training:
            return v_bar
        
        if self.is_activation:
            return v_bar.byte()
        else:
            return v_bar.char()

    def dequantize(self, v_bar):
        # v_bar = v_bar.float()
        if self.per_channel:
            v_hat = v_bar.float() * self.step_size.repeat(1, v_bar[0].numel()).reshape(v_bar.shape)
        else:
            v_hat = v_bar.float() * self.step_size
        return v_hat


class LSQ_Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs
    ):
        super(LSQ_Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )
        self.activation_quantizer = LSQ_Quantizer(
            bit_width=kwargs['bit_width'][0],
            is_activation=True
        )
        self.weight_quantizer = LSQ_Quantizer(
            bit_width=kwargs['bit_width'][1],
            is_activation=False,
            per_channel=kwargs['per_channel'],
            out_channels=out_channels if kwargs['per_channel'] else None
        )
        self.register_buffer('initialized', torch.zeros(1))

    def forward(self, x):
        if self.initialized == 0:
            self.activation_quantizer.initialize_step_size(x)
            self.weight_quantizer.initialize_step_size(self.weight)
            self.initialized.fill_(1)
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_activation = self.activation_quantizer(x)
        return F.conv2d(quantized_activation, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LSQ_Linear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        **kwargs
    ):
        super(LSQ_Linear, self).__init__(
            in_features,
            out_features,
            bias
        )
        self.activation_quantizer = LSQ_Quantizer(
            bit_width=kwargs['bit_width'][0],
            is_activation=True
        )
        self.weight_quantizer = LSQ_Quantizer(
            bit_width=kwargs['bit_width'][1],
            is_activation=False
        )
        self.register_buffer('initialized', torch.zeros(1))

    def forward(self, x):
        if self.initialized == 0:
            self.activation_quantizer.initialize_step_size(x)
            self.weight_quantizer.initialize_step_size(self.weight)
            self.initialized.fill_(1)
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_activation = self.activation_quantizer(x)
        return F.linear(quantized_activation, quantized_weight, self.bias)
