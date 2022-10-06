from pathlib import Path
from PIL import Image
import numpy

import torch

from pytorch_lightning.callbacks import Callback

class TestOutputLogger(Callback):
  def __init__(self, log_path=''):
    self.log_path = Path(log_path)
    if log_path:
      self.log_path.mkdir(parents=True, exist_ok=True)

  def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    if not self.log_path:
      return
    if isinstance(outputs['preds'], (list,tuple)):
      preds = [p['out'] for p in outputs['preds']]
    else:
      preds = outputs['preds']['out']
    # save reprojection masks with the higher scores for reproj-feedback
    sequence =  preds[-1] if isinstance(preds,list) else preds
    for i, pred in enumerate(sequence):
      # import ipdb; ipdb.set_trace()
      mask = torch.argmax(pred, dim=0).detach().to('cpu').numpy()
      mask = Image.fromarray(mask.astype(numpy.uint8))
      mask.save(self.log_path / batch[-1]['file_names'][i])
