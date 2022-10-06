import torch
from torch import nn
import torch.nn.functional as F

import models.blocks as blocks
import models.utils as utils


class Unet(nn.Module):
  def __init__(self, cfg={}, in_channels=3, out_channels=None, base_channels=64):
    super(Unet, self).__init__()
    
    if cfg:
      self.in_channels = cfg['dataset']['img_size']['depth'] if 'img_size' in cfg['dataset'] else in_channels
      self.out_channels = out_channels or len(cfg['dataset']['class_labels'])
      if 'base_channels' in cfg['network'].keys() and cfg['network']['base_channels']:
        base_channels = cfg['network']['base_channels']
    else:
      self.in_channels = in_channels
      self.out_channels = out_channels
      
    # default out_channel_sizes: (64, 128, 256, 512, 1024)
    self.out_channel_sizes = [base_channels * 2**pow for pow in range(5)]


    self.conv_maxpool = torch.nn.MaxPool2d(kernel_size=2)
    self.layers_dict = torch.nn.ModuleDict()

    ## Encoder (default settings)
    ## Layer Names: encoder_[0,    1,   2,   3  ]
    ## Layer In channels    [im_d, 64,  128, 256]   
    ## Layer out channels   [64,   128, 256, 512]
    ############
    in_channels = self.in_channels
    for depth, out_channels in zip(range(4), self.out_channel_sizes[:-1]):
      layer_name = f'encoder_{depth}'
      self.layers_dict[layer_name] = blocks.contracting_block(in_channels, out_channels)
      in_channels = out_channels

    ## Bottleneck (default settings)
    ## Layer Names:        bottelneck
    ## Layer In channels   512
    ## Layer out channels  1024
    ###############
    layer_name = 'bottelneck'
    self.layers_dict[layer_name] = blocks.bottleneck_block(in_channels, self.out_channel_sizes[-1], out_channels)
    # From now on, cat(encoder_n, decoder_n)
    in_channels = 2 * out_channels # cat(out_encoder_n, out_decoder_n)

    ## Decoder (default settings)
    ## Layer Names: decoder_[3,    2,   1  ]
    ## Layer In channels    [1024, 512, 256]
    ## mid channels         [512,  264, 128]
    ## Layer out channels   [256,  128, 64 ]
    ## Cat out channels     [512,  264, 128]
    ############
    for depth, mid_channels in zip(reversed(range(0,3)), reversed(self.out_channel_sizes[:-1])):
      layer_name = f'decoder_{depth + 1}'
      out_channels = mid_channels // 2
      self.layers_dict[layer_name] = blocks.expansive_block(in_channels, mid_channels, out_channels)
      in_channels = 2 * out_channels # cat(out_encoder_n, out_decoder_n)

    ## Decoder Final layers (default settings)
    ## Layer Names: decoder_[0_B, 0_A]
    ## Layer in channels    [128, 64]
    ## Layer out channels   [64,  C ]
    ############
    layer_name = 'decoder_0_B'
    self.layers_dict[layer_name] = blocks.final_block_seq1(in_channels, in_channels // 2)
    layer_name = 'decoder_0_A'
    self.layers_dict[layer_name] = blocks.final_block_seq2(in_channels // 2, self.out_channels)

  def forward(self, x):
    encode_blocks = []
    for layer_name, layer in self.layers_dict.items():
      ## Encode
      ###########
      if 'encoder' in layer_name:
        # compute feature map
        encode_blocks.append(layer(x))
        x = self.conv_maxpool(encode_blocks[-1])
        
      ## Bottleneck
      ###############
      if 'bottelneck' in layer_name:
        bottleneck = layer(x)
        decode_block = utils.crop_and_concat(bottleneck, encode_blocks[-1])

      ## Decode
      ###############
      if 'decoder' in layer_name:
        decode_block = layer(decode_block)

        dec_depth = int(layer_name.split('_')[1])
        if dec_depth != 0:
          decode_block = utils.crop_and_concat(decode_block, encode_blocks[dec_depth-1])

    return {"out": decode_block}

class UnetDeepInput(Unet):
  def __init__(self,  cfg={},
                      in_channels=None, out_channels=None, base_channels=64,
                      extra_input_layer=False,
                      bigger_input_kernels=False,
                      unet_pretrained_path=''):
    super(UnetDeepInput, self).__init__(cfg=cfg,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        base_channels=base_channels)
    
    if unet_pretrained_path:
        pretrain = torch.load(unet_pretrained_path, map_location='cpu')
        unet_state_dict = {k.replace('net.',''):v for k,v in pretrain['state_dict'].items() if 'net.' in k}
        self.load_state_dict(unet_state_dict, strict=True)

    ## Modified input encoder layer for deeper input tensors
    ##########################################################
    encoder_list = []
    if extra_input_layer:
      # add base_size/2 extra layer with depth*depth frames
      kernel_size = 5
      if bigger_input_kernels:
        # in layer with 7x7 filters
        kernel_size = 7
      encoder_list.extend(list(blocks.conv_block(in_channels,
                                                 base_channels//2,
                                                 kernel_size=kernel_size,
                                                 padding=kernel_size//2)))
    in_channels = base_channels//2

    # change original in layer in channel size
    kernel_size = 3
    if bigger_input_kernels:
      # in layer with 5x5 filters
      kernel_size = 5
    encoder_list.extend(list(blocks.contracting_block(in_channels,
                                                      base_channels,
                                                      kernel_size=kernel_size)))

    # Replace first encoder layer sequential with modified one
    self.layers_dict[list(self.layers_dict.keys())[0]] = nn.Sequential(*encoder_list)