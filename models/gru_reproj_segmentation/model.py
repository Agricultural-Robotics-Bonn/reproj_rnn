import torch
from torch import nn
import torch.nn.functional as F

import models.blocks as blocks
import models.utils as utils

from modules.reprojectorch import ReprojMaskToShiftMatrix
from modules.reprojectorch import PxCoordOrder

from models.convgru import ConvGRUCell
from models.unet.model import Unet

class GRUReprojCell(nn.Module):
  def __init__(self, cfg={}, intrinsics=None, extrinsics=None):
    super(GRUReprojCell, self).__init__()
    
    # Instanciate base unet and load pretrained if required
    self.unet_base = Unet(cfg)
    if cfg['network']['unet_base_pretrained']:
        pretrain = torch.load(cfg['network']['unet_pretrained_path'], map_location='cpu')
        unet_state_dict = {k.replace('net.',''):v for k,v in pretrain['state_dict'].items() if 'net.' in k}
        self.unet_base.load_state_dict(unet_state_dict, strict=True)
    # Remap unet labels to this model
    self.layers_dict = self.unet_base.layers_dict
    self.conv_maxpool = self.unet_base.conv_maxpool
    
    ## Attention layers
    #####################

    # Load attention parameters 
    self.fb_layers = cfg['network']['prior']['layers']
    self.GRU_layers = torch.nn.ModuleDict()
    for layer_name, seq_block in self.layers_dict.items():
      if layer_name in self.fb_layers:
        out_channels = [op.out_channels for op in reversed(seq_block) if hasattr(op, 'out_channels')][-1]
        self.GRU_layers[layer_name] = ConvGRUCell(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

  def forward(self, x, priors={}):
    # in_buffer = x.clone()

    encode_blocks = []
    for layer_name, layer in self.layers_dict.items():
      ## Encode
      ###########
      if 'encoder' in layer_name:
        # compute feature map
        encode_blocks.append(layer(x))
        
        # Update GRU if layer is in the temporal list
        if layer_name in self.fb_layers:
          # to initize GRU's hidden state  
          if layer_name not in priors.keys():
            priors[layer_name] = None
          encode_blocks[-1] = self.GRU_layers[layer_name](encode_blocks[-1], priors[layer_name])
          # buffer output/hidden state for reprojection
          priors[layer_name] = encode_blocks[-1]

        x = self.conv_maxpool(encode_blocks[-1])
        
      ## Bottleneck
      ###############
      if 'bottelneck' in layer_name:
        bottleneck = layer(x)
        decode_block = utils.crop_and_concat(bottleneck, encode_blocks[-1])
        
        # Update GRU if layer is in the temporal list
        if layer_name in self.fb_layers:
          # to initize GRU's hidden state  
          if layer_name not in priors.keys():
            priors[layer_name] = None
          decode_block = self.GRU_layers[layer_name](decode_block, priors[layer_name])
          # buffer output/hidden state for reprojection
          priors[layer_name] = decode_block

      ## Decode
      ###############
      if 'decoder' in layer_name:
        decode_block = layer(decode_block)

        dec_depth = int(layer_name.split('_')[1])
        if dec_depth != 0:
          decode_block = utils.crop_and_concat(decode_block, encode_blocks[dec_depth-1])
        

        # Update GRU if layer is in the temporal list
        if layer_name in self.fb_layers:
          # to initize GRU's hidden state  
          if layer_name not in priors.keys():
            priors[layer_name] = None
          decode_block = self.GRU_layers[layer_name](decode_block, priors[layer_name])
          # buffer output/hidden state for reprojection
          priors[layer_name] = decode_block

    output = decode_block

    # save reprojection masks with the higher scores for reproj-feedback
    mask = torch.unsqueeze(torch.argmax(output, dim=1),1).type(output.dtype)

    return {"out": output, "mask": mask, "priors": priors}

class ReprojRNN(nn.Module):
  def __init__(self, cfg={}, intrinsics=None, extrinsics=None):
    super(ReprojRNN, self).__init__()

    self.reprojection_enabled = cfg['network']['prior']['spatial_enable']
    self.reproj_default_value = cfg['network']['prior']["default_value"]
    
    # Reprojector
    if intrinsics and extrinsics:
      self.reprojector = ReprojMaskToShiftMatrix(cam_model=intrinsics,
                                                 cam_extrinsics=extrinsics,
                                                 coord_order=PxCoordOrder.vu)
    else:
      self.reprojector = ReprojMaskToShiftMatrix(coord_order=PxCoordOrder.vu)

    self.segmentation_cell = GRUReprojCell(cfg, intrinsics, extrinsics)


  def forward(self, x):
    # Feedback tensor initialized by the segmentation model
    priors = {}
    preds = []
    for i, frame in enumerate(x):
      # Segmentation cell forward pass
      out_dict = self.segmentation_cell(frame['rgb'], priors)
      preds.append({'out':out_dict['out'].clone()})

      if not self.reprojection_enabled:
        priors = out_dict['priors']
        continue
      
      reproj_mask = torch.ones_like(out_dict['mask'])
      if 'robot_mask' in frame:
        reproj_mask = torch.where(frame['robot_mask']>0, reproj_mask, torch.zeros_like(reproj_mask))

      # reproject feedback to the next frame
      feat_shifts = utils.reprojectTensorFeedback(
                      reproj_mask,
                      frame['depth'] if 'depth' in frame.keys() else None,
                      frame['odom'] if 'odom' in frame.keys() else None,
                      frame['intrinsics'] if 'intrinsics' in frame.keys() else None,
                      frame['extrinsics'] if 'extrinsics' in frame.keys() else None,
                      tensor_reproj_model=self.reprojector,
                      )

      for layer, tensor in out_dict['priors'].items():
        priors[layer] = utils.shiftTensor(features=tensor,
                                      shiftMat=feat_shifts,
                                      mask=reproj_mask,
                                      default_value=self.reproj_default_value)
    
    return preds