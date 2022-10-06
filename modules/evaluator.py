# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import torchmetrics as metrics
from pytorch_lightning.callbacks import Callback

def tb_confusion_matrix_img(cm, class_names, epsilon=1e-6):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    if isinstance(cm, torch.Tensor):
      cm = cm.cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+epsilon), decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Normalize into 0-1 range for TensorBoard(X)
    img = img / 255.0
    # Swap axes (Newer API>=1.8 expects colors in first dim)
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.swapaxes(img, 1, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    plt.close(fig)

    return img


class EvaluatorStageWrapper:
  def __init__(self, evaluator=None, stage=None):
    if not evaluator or not isinstance(evaluator, nn.Module):
      raise TypeError('Invalid evaluator passed to evaluator callback class')
    self.eval = evaluator
    if any([stage == s for s in ['train', 'val', 'test']]):
      self.stage = stage
    else:
      raise ValueError('Unable to instanciate evaluator callback. Stage parameter must be one of: [train, val, test]')

  def setup(self, stage=None):
    #reset all metrics
    if any([stage == s for s in ['train', 'val', 'test']]):
      self.stage = stage
    self.eval.reset()

  def on_batch_end(self, pl_module, outputs, batch_idx):
    self.eval.log_metrics_batch(pl_module, outputs, batch_idx, self.stage)
  
  def on_epoch_end(self, pl_module):
    self.eval.log_metrics_epoch(pl_module, self.stage)
    self.eval.reset()


class EvaluatorTrainCallbacks(Callback):
  def __init__(self, evaluator=None):
    self.eval = EvaluatorStageWrapper(evaluator, 'train')
  # Train Callbacks
  def setup(self, trainer, pl_module, stage=None):
    self.eval.setup()
  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.eval.on_batch_end(pl_module, outputs[0][0], batch_idx)
  def on_train_epoch_end(self, trainer, pl_module, outputs):
    self.eval.on_epoch_end(pl_module)


class EvaluatorValCallbacks(Callback):
  def __init__(self, evaluator=None):
    self.eval = EvaluatorStageWrapper(evaluator, 'val')
  # Val Callbacks
  def setup(self, trainer, pl_module, stage=None):
    self.eval.setup()
  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.eval.on_batch_end(pl_module, outputs, batch_idx)
  def on_validation_epoch_end(self, trainer, pl_module):
    self.eval.on_epoch_end(pl_module)


class EvaluatorTestCallbacks(Callback):
  def __init__(self, evaluator=None):
    self.eval = EvaluatorStageWrapper(evaluator, 'test')
  # Test Callbacks
  def setup(self, trainer, pl_module, stage=None):
    self.eval.setup()
  def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.eval.on_batch_end(pl_module, outputs, batch_idx)
  def on_test_epoch_end(self, trainer, pl_module):
    self.eval.on_epoch_end(pl_module)


class EvaluatorBase(nn.Module):
  '''
  Evaluator base class 
  '''
  def __init__(self, class_num, class_labels, class_weights=[], log_batch=True, log_epoch=True, log_best_metrics=True):
    super(EvaluatorBase, self).__init__()
    self.num_classes = class_num
    self.class_labels = class_labels

    self.log_best_metrics = log_best_metrics
    self.best_metrics={}
    self.best_criterions={'iou':max,
                          'F1':max,
                          'precision':max,
                          'recall':max,
                          'loss':min}

    self.log_periods = []
    self.log_batch = log_batch
    if self.log_batch: self.log_periods.append('batch')
    self.log_epoch = log_epoch
    if self.log_batch: self.log_periods.append('epoch')


    # progress bar metrics interface
    self.pbar_metrics = []
    # metric_dict interface
    self.metric_dict = nn.ModuleDict()

    if class_weights:
      self.register_buffer('class_weights', torch.Tensor(class_weights))
    else:
      self.register_buffer('class_weights', torch.ones(self.class_num))
    
    self.register_buffer('running_loss', torch.Tensor([]))

  def get_metric_names(self):
    metric_names = []
    for period in self.log_periods:
      # loss name
      metric_names.append(f'{period}_loss')
      for stat_name, stat in self.metric_dict.items():
        # all other metric names
        if stat_name == 'conf_matrix':
          metric_names.append(f'{period}_{stat_name}')
          continue
        # average metric names
        avg_str = 'wavg'
        if torch.all(self.class_weights == 1):
          avg_str = 'avg'
        metric_names.append(f'{period}_{stat_name}_{avg_str}')
        # Class-wise metric names
        for label in self.class_labels:
          metric_names.append(f'{period}_{stat_name}_{label}')

    return metric_names

  def reset(self):
    self.running_loss = torch.Tensor([]).type_as(self.running_loss)
    for m in self.metric_dict.values():
      m.reset()

  def update_loss(self, loss=None):
    if loss:
      self.running_loss = torch.cat((self.running_loss, torch.unsqueeze(loss,0)))

  def update(self, outputs, loss=None):
    '''
    Incrementally update metric with predictions and labels

    This function should be overriden if additional evaluation inputs/capabilities
    are required by concrete evaluation child clases
    ''' 
    preds, labels = (outputs['preds']['out'], outputs['labels'])
    class_preds = preds.argmax(dim=1)
    
    return {name:m(class_preds, labels) for name, m in self.metric_dict.items()}
      
  def weightedAVG(self, class_scores):
    '''
    Compute class-wise weighted average score using weights specified in config file.
    If no weights are specified, the arithmetic mean is computed.
    '''
    return  ((class_scores * self.class_weights).sum() / self.class_weights.sum())

  def compute(self):
    '''
    Compute all class-weighted metrics and loss from incremental gathered batch results
    '''
    ret = {}
    ret['loss'] = torch.mean(self.running_loss) if self.running_loss.nelement() != 0 else 0
    for name, m in self.metric_dict.items():
      stat_classes = m.compute()
      # {metric_name : [Weighted average metric, class metric]
      ret[name] = [self.weightedAVG(stat_classes), stat_classes]
    return ret
  
  def log_metrics_batch(self, pl_module, outputs, batch_idx, stage='', loss=None):
    if loss is None:
      if 'loss' in outputs.keys():
        loss = outputs['loss']
      elif 'minimize' in outputs.keys():
        loss = outputs['minimize']
    if stage == 'train' and 'extra' in outputs.keys():
      outputs = outputs['extra']

    if any([isinstance(v, (list, tuple)) for k,v in outputs.items()]):
      out_keys = [k for k in outputs.keys() if k != 'loss']
      seq_length = len(outputs[out_keys[0]])

      for i in range(seq_length):
        out = {k:outputs[k][i] for k in out_keys}
        self.log_metrics_single_batch(pl_module, out, batch_idx, stage, loss)

    else:
      self.log_metrics_single_batch(pl_module, outputs, batch_idx, stage, loss)

  def log_metrics_single_batch(self, pl_module, outputs, batch_idx, stage='', loss=None):
    # unpack loss and log if exsists
    if 'loss' in outputs.keys():
      loss = outputs['loss']
    elif 'minimize' in outputs.keys():
      loss = outputs['minimize']
    if loss:
      self.update_loss(loss)
      
      if self.log_batch:
        self.log_metric(pl_module, f'{stage}_batch_loss', loss)
    
    # retrieve extra outputs to compute training metrics
    if stage == 'train' and 'extra' in outputs.keys():
      outputs = outputs['extra']
    # update metrics with current batch results
    stats_dict = self.update(outputs, loss)
    
    if not self.log_batch:
      return  
    
    # log all batch computed metrics
    for stat_name, stat_classes in stats_dict.items():
      if stat_name == 'conf_matrix':
        cm_img = tb_confusion_matrix_img(stat_classes, self.class_labels)
        if pl_module.logger is not None:
          pl_module.logger.experiment.add_image(f'{stage}_batch_{stat_name}',cm_img, batch_idx)  
        continue

      # Weighed average stat
      avg_str = 'wavg'
      if torch.all(self.class_weights == 1):
        avg_str = 'avg'
      self.log_metric(pl_module, f'{stage}_batch_{stat_name}_{avg_str}', self.weightedAVG(stat_classes))

      # Class-wise stat
      for label, s in zip(self.class_labels, stat_classes):
        self.log_metric(pl_module, f'{stage}_batch_{stat_name}_{label}', s)

  def log_metrics_epoch(self, pl_module, stage=''):
    stats = self.compute()
    if not self.log_epoch:
      return
      
    for stat_name, stat in stats.items():
      # class related stats/scores
      if isinstance(stat,list):
        stat_avg, stat_classes = stat
        if stat_name == 'conf_matrix':
          cm_img = tb_confusion_matrix_img(stat_classes, self.class_labels)
          if pl_module.logger is not None:
            pl_module.logger.experiment.add_image(f'{stage}_epoch_{stat_name}',cm_img, pl_module.current_epoch)
          continue
        # Weighed average stat
        avg_str = 'wavg'
        if torch.all(self.class_weights == 1):
          avg_str = 'avg'
        self.log_metric(pl_module, f'{stage}_epoch_{stat_name}_{avg_str}', stat_avg)
        # Class-wise stat
        for label, stat_class in zip(self.class_labels, stat_classes):
          self.log_metric(pl_module, f'{stage}_epoch_{stat_name}_{label}', stat_class)
      else:
        # General stats (e.g.: epoch loss)
        self.log_metric(pl_module, f'{stage}_epoch_{stat_name}', stat)
  
  def log_metric(self, pl_module, metric_name, metric):
    pl_module.log(metric_name, metric)

    if any([metric_name in pbm for pbm in self.pbar_metrics]):
      metric_name = '_'.join(metric_name.split('_')[-2:])
      pl_module.log(metric_name, metric, prog_bar=True)

    if self.log_best_metrics:
      criterion = [c for name,c in self.best_criterions.items() if name in metric_name]
      if not criterion:
        return
      # Initialize best metric if not already
      if metric_name not in self.best_metrics.keys():
        self.best_metrics[metric_name] = metric
      # Replace metric if better than pervious one 
      else:
        self.best_metrics[metric_name] = criterion[0](self.best_metrics[metric_name], metric)
      # log best metric
      pl_module.log(f'{metric_name}_best', self.best_metrics[metric_name])
    


#semantic evaluators
class SemanticEvaluator(EvaluatorBase):
  def __init__(self, *args, **kwargs):
    super(SemanticEvaluator, self).__init__(*args, **kwargs)

    self.metric_dict['F1'] = metrics.F1(self.num_classes, average='none')
    self.metric_dict['iou'] = metrics.IoU(self.num_classes, reduction='none')
    self.metric_dict['precision'] = metrics.Precision(self.num_classes, average='none', mdmc_average='global')
    self.metric_dict['recall'] = metrics.Recall(self.num_classes, average='none', mdmc_average='global')

class SemanticBUPEvaluator(SemanticEvaluator):
  def __init__(self, *args, **kwargs):
    super(SemanticBUPEvaluator, self).__init__(*args, **kwargs)
    self.pbar_metrics = ['batch_val_iou_pepper']
class SemanticSBEvaluator(SemanticEvaluator):
  def __init__(self, *args, **kwargs):
    super(SemanticBUPEvaluator, self).__init__(*args, **kwargs)
    self.pbar_metrics = ['batch_val_iou_wavg']