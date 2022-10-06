# -*- coding: utf-8 -*-

import torch
import numpy as np
import torchcontrib
import torch.nn as nn


class OptimizerConfigurator:
  def __init__(self, parameters, cfg={}):
    
    self.optimCfg = cfg['optimizer']
    self.base_lr = self.optimCfg["base_lr"]
    self.max_lr = self.optimCfg["max_lr"]
    self.alpha = self.optimCfg["alpha"]
    self.gamma = self.optimCfg["gamma"]
    self.momentum = self.optimCfg["momentum"]
    self.optimizer_type = self.optimCfg["type"]
    self.lr_scheduler =  self.optimCfg['lr_scheduler']
    self.step_size= self.optimCfg["step_size"]
    self.weight_decay = self.optimCfg['weight_decay'] if 'weight_decay' in self.optimCfg.keys() else 0

    self.class_labels = cfg['dataset']['class_labels']
    self.class_weights = torch.from_numpy(np.array(cfg['dataset']['class_weights'])).float()
    
    self.parameters = parameters
    
  def get_optimizer(self):
    opt = None
    print("#[INFO] Optimizer:", self.optimizer_type)
    # Choose between the optimisers...
    if self.optimizer_type == "SGD":
      opt = torch.optim.SGD(self.parameters(), lr=self.base_lr, momentum=self.momentum, weight_decay=self.weight_decay)
    elif self.optimizer_type == "Adam":
      opt = torch.optim.Adam(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
    elif self.optimizer_type == "RMSProp":
      opt = torch.optim.RMSprop(self.parameters(), lr=self.base_lr, alpha=self.alpha, weight_decay=self.weight_decay)

    self.optimizer = opt
    return self.optimizer
      
  def get_lr_scheduler(self):
    ### Set the decay function for the learning rate
    if self.lr_scheduler == 'StepLR' :
      lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                     step_size=self.step_size, 
                                                     gamma=self.gamma)
    elif self.lr_scheduler == 'CyclicLR':
      lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                       base_lr=self.base_lr, 
                                                       max_lr=self.max_lr, 
                                                       gamma=self.gamma, 
                                                       step_size_up=self.step_size, 
                                                       cycle_momentum=True)
    else:
      lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                     step_size=self.step_size, 
                                                     gamma=self.gamma)
    
    return lr_scheduler
      
  def get_criterion_base(self, loss_type, class_weights):
    if loss_type == "xentropy":
      return torch.nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "dice":
      print("*[TODO] dice loss needs to be added!!!")
    elif loss_type == "focal":
      print("*[TODO] focal loss needs to be added!!!")
    elif loss_type == "BEC":
      return torch.nn.BCELoss(weight=self.class_weights)       
    print("*[Warning] No valid loss retrieved")
    return None

  def get_criterion(self):
    return self.get_criterion_base(self.optimCfg["loss_type"], self.class_weights)

  def get_reprojected_labels_criterion(self):
    reproj_class_weights = self.class_weights.clone()
    for i, label in enumerate(self.class_labels):
      if any([label == bg for bg in ['bg', 'background', 'BG', 'BACKGROUND']]):
        reproj_class_weights[i] = 0.0
        
    return self.get_criterion_base(self.optimCfg["loss_type"], reproj_class_weights)