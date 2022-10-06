import torch
from torch import nn
import numpy as np
from enum import Enum, auto

# conv block default configs
###############################################################

if 'ACTIVATION_FUNCTION_DEFAULT' not in globals(): 
    ACTIVATION_FUNCTION_DEFAULT = nn.ReLU

if 'APPLY_BATCH_NORM_DEFAULT' not in globals(): 
    APPLY_BATCH_NORM_DEFAULT = True

if 'DROP_OUT_PROBABILITY_DEFAULT' not in globals(): 
    DROP_OUT_PROBABILITY_DEFAULT = 0.0

def set_activation_function_default(activation=nn.ReLU):
    global ACTIVATION_FUNCTION_DEFAULT
    ACTIVATION_FUNCTION_DEFAULT = activation

def set_apply_batch_norm_default(apply=False):
    global APPLY_BATCH_NORM_DEFAULT
    APPLY_BATCH_NORM_DEFAULT = apply

def set_dropout_probabaility_default(probability=0.0):
    global DROP_OUT_PROBABILITY_DEFAULT
    DROP_OUT_PROBABILITY_DEFAULT = probability

###############################################################


def conv_block(in_channels, out_channels, kernel_size=3, padding=1, stride=1, activation_func=None, apply_batch_norm=None, dropout_prob=None):
    # Populate configs with global defaults if not specified
    global ACTIVATION_FUNCTION_DEFAULT
    activation_func = activation_func if activation_func is not None else ACTIVATION_FUNCTION_DEFAULT
    global APPLY_BATCH_NORM_DEFAULT
    apply_batch_norm = apply_batch_norm if apply_batch_norm is not None else APPLY_BATCH_NORM_DEFAULT
    global DROP_OUT_PROBABILITY_DEFAULT
    dropout_prob = dropout_prob if dropout_prob is not None else DROP_OUT_PROBABILITY_DEFAULT

    # build conv block
    block = [nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=padding, stride=stride),
             activation_func(),]
    if apply_batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    if dropout_prob is not None and dropout_prob != 0.0:
        block.append(torch.nn.Dropout(p=dropout_prob))

    return tuple(block)

def conv_block_sequential(in_channels, out_channels, kernel_size=3, padding=1, stride=1 ,activation_func=None, apply_batch_norm=None, dropout_prob=None):
    return nn.Sequential(*conv_block(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, padding=padding, stride=stride,
                                                activation_func=activation_func,
                                                apply_batch_norm=apply_batch_norm,
                                                dropout_prob=dropout_prob)
                                )

def interpolate_block(sampling_OS, intergration_OS, in_channels, out_channels, kernel_size=3):
    """
    This function creates one interpolation block
    """
    div = intergration_OS // sampling_OS

    # concatenate all blocks on a single tuple and pack them on a sequential module at the end
    blocks_tuple = ()
    module_nums = range(int(np.log2(div)) + 1)

    block_out = out_channels
    for mod_num in module_nums:
        stride = 2
        if len(module_nums) == 1:
            block_in = block_out = in_channels
            stride = 1
        elif mod_num == module_nums[-1]:
            block_in = block_out = out_channels
            stride = 1
        elif mod_num == module_nums[0]:
            block_in = in_channels
            block_out //= div//2
        else:
            block_in = block_out
            block_out *= 2
        # add modules to tuple
        blocks_tuple += conv_block(in_channels=block_in, out_channels=block_out,
                                    kernel_size=kernel_size, padding=1, stride=stride)
    # unpack modules as sequential arguments
    return nn.Sequential(*blocks_tuple)

def contracting_block(in_channels, out_channels, kernel_size=3):
    """
    This function creates one contracting block
    """
    return nn.Sequential(
                *conv_block(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                *conv_block(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                )

def expansive_block(in_channels, mid_channel, out_channels, kernel_size=3):
    """
    This function creates one expansive block
    """
    return nn.Sequential(
                    *conv_block(in_channels, mid_channel, kernel_size=kernel_size),
                    *conv_block(mid_channel, mid_channel, kernel_size=kernel_size),
                    nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    )

def decode_block(in_channels, out_channels, kernel_size=3):
    """
    This function creates a decode block
    """
    return nn.Sequential(
                    *conv_block(in_channels, out_channels, kernel_size=kernel_size),
                    *conv_block(out_channels, out_channels, kernel_size=kernel_size),
                    )

def conv_upsample(in_channels, out_channels):
    """
    This function creates a conv upsample block
    """
    return nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    )

def final_block_seq1(in_channels, mid_channel, kernel_size=3):
    """
    This returns final block
    """
    return conv_block_sequential(in_channels, mid_channel, kernel_size=kernel_size)

def final_block_seq2(in_channels, out_channels, kernel_size=3):
    """
    This returns final block
    """
    return nn.Sequential(
                *conv_block(in_channels, in_channels, kernel_size=kernel_size),
                *conv_block(in_channels, out_channels, kernel_size=kernel_size)
                )

def frame_fusion_block(in_channels, mid_channels, out_channels, kernel_size=3):
    return nn.Sequential(
                *conv_block(in_channels, mid_channels, kernel_size=kernel_size),
                *conv_block(mid_channels, out_channels, kernel_size=kernel_size)
                )

def final_block_seq3(in_channels, out_channels, kernel_size=3):
    """
    This returns final block
    """
    return conv_block_sequential(in_channels, out_channels, kernel_size=kernel_size)

def final_block(in_channels, mid_channel, out_channels, kernel_size=3):
    """
    This returns final block
    """
    return nn.Sequential(
                *conv_block(in_channels, mid_channel, kernel_size=kernel_size),
                *conv_block(mid_channel, mid_channel, kernel_size=kernel_size),
                *conv_block(mid_channel, out_channels, kernel_size=kernel_size)
                )

def bottleneck_block(in_channels, mid_channel, out_channels, kernel_size=3):
    return nn.Sequential(
                *conv_block(in_channels, mid_channel, kernel_size=kernel_size),
                *conv_block(mid_channel, mid_channel, kernel_size=kernel_size),
                nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                )

# Attention mechanism from:
# Self-Supervised Model Adaptation for Multimodal Semantic Segmentation A. Valada, et al
# https://arxiv.org/abs/1808.03833 (Sec. 4.1)
class SSMAActivationFunction(Enum):
    SIGMOID = auto()
    TANH = auto()

act_func_map = {SSMAActivationFunction.SIGMOID: nn.Sigmoid,
                SSMAActivationFunction.TANH: nn.Tanh,}

def SSMA_attention_bottleneck(in_channels, mid_channels, kernel_size, activation_func, out_channels=None):
    if out_channels is None:
        out_channels = in_channels
    return nn.Sequential(
            *conv_block(in_channels, mid_channels, kernel_size=kernel_size, apply_batch_norm=False),
            *conv_block(mid_channels, out_channels, kernel_size=kernel_size, apply_batch_norm=False,activation_func=act_func_map[activation_func]),
            )

class SSMA_block(nn.Module):
    def __init__(self, in_channels: int, num_modes: int, activation_func: SSMAActivationFunction, compression_rate=2, kernel_size=3):
        super(SSMA_block, self).__init__()

        # num channels from all modalities combined
        cat_channels = int(in_channels * num_modes)
        mid_channels = int(cat_channels/compression_rate)
        # Attention weighting matrix without batch norm and with values in range [-1,1] 
        self.attention = SSMA_attention_bottleneck(cat_channels, mid_channels, kernel_size=kernel_size, activation_func=activation_func)
        # Final conv with batch norm to size of a single modality
        self.conv_norm = conv_block_sequential(cat_channels, in_channels, kernel_size=kernel_size,activation_func=nn.Identity)

    def forward(self, x):
        modes_cat = torch.cat(x,1)
        return self.conv_norm(self.attention(modes_cat) * modes_cat)

class SSMA_dense_block(nn.Module):
    def __init__(self, in_channels: int, num_modes: int, activation_func: SSMAActivationFunction, compression_rate=2, kernel_size=3):
        super(SSMA_dense_block, self).__init__()
        # num channels from all modalities combined
        cat_channels = int(in_channels * num_modes)
        mid_channels = int(cat_channels/compression_rate)
        # Attention weighting matrix without batch norm and with values in range [-1,1] 
        self.attention = SSMA_attention_bottleneck(in_channels=cat_channels,
                                                   mid_channels=mid_channels,
                                                   out_channels=in_channels,
                                                   kernel_size=kernel_size,
                                                   activation_func=activation_func)
        # Final conv with batch norm to size of a single modality
        self.conv_norm = conv_block_sequential(in_channels, in_channels, kernel_size=kernel_size,activation_func=nn.Identity)

    def forward(self, x):
        modes_cat = torch.cat(x,1)
        return self.conv_norm(self.attention(modes_cat) * x[0])