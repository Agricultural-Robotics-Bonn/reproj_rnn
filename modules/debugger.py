# -*- coding: utf-8 -*-

import numpy as np

import torch
import torchvision

from pytorch_lightning.callbacks import Callback

class NetDebugger(Callback):
  def __init__(self, cfg={}):
    self.log_out_every_n_epochs = cfg['debugger']['log_output_every_n_epochs']
    self.num_log_outputs = cfg['debugger']['num_log_outputs']
  
  def setup(self, trainer, pl_module, stage=None):
    self.log_flag = False
    self.images = None
  
  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    if self.log_flag and batch_idx < self.num_log_outputs:
      mask = torch.argmax(outputs['preds']['out'][0], dim=0).detach().to('cpu')
      blend = batch[-1]['rgb'][0] * 0.5 + torch.unsqueeze(mask, 0).repeat(3,1,1) * 0.5
      label = batch[-1]['labels'][0].repeat(3,1,1)
      blend_label = torch.cat((blend,label), dim=1)
      
      if self.images == None:
        self.images = blend_label.unsqueeze(0)
      else:
        self.images = torch.cat((self.images, blend_label.unsqueeze(0)))
    
    else:
      if self.images != None:
        pl_module.logger.experiment.add_images(f'dbg_val_out',self.images, pl_module.current_epoch)
      self.log_flag = False
      self.images = None

  def on_train_epoch_end(self, trainer, pl_module, outputs):
    if pl_module.current_epoch % self.log_out_every_n_epochs == 0:
      self.log_flag = True
