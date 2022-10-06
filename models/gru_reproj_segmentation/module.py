import pytorch_lightning as pl

from modules.optimizer import OptimizerConfigurator
from modules.evaluator import SemanticEvaluator
from modules.evaluator import EvaluatorTrainCallbacks
from modules.evaluator import EvaluatorValCallbacks
from modules.evaluator import EvaluatorTestCallbacks

from models.gru_reproj_segmentation.model import ReprojRNN

class PLModule(pl.LightningModule):

  def __init__(self, cfg={}, intrinsics=None, extrinsics=None, evaluator=None):
    super(PLModule, self).__init__()
    self.cfg = cfg
    self.hparams = cfg
    self.save_hyperparameters()
    self.net = ReprojRNN(cfg, intrinsics, extrinsics)

  #############################################
  ### Infrastructure Config
  #############################################
  def configure_callbacks(self):
    # Create independant evaluators for train, test & val to allow independet
    # stage-wise metrics bookkeeping
    class_labels = self.cfg['dataset']['class_labels']
    class_num = len(class_labels)
    class_weights = self.cfg['dataset']['class_weights'] if 'class_weights' in self.cfg['dataset'] else []

    self.train_evaluator = SemanticEvaluator(class_num, class_labels, class_weights)
    self.val_evaluator = SemanticEvaluator(class_num, class_labels, class_weights)
    self.test_evaluator = SemanticEvaluator(class_num, class_labels, class_weights)


    # log hyperparameters and validation metrics for hparam search and comparison
    init_val_metrics = {f'val_{m}_best':0 for m in self.val_evaluator.get_metric_names() if 'epoch' in m}
    tb_loggers = [l for l in self.logger if isinstance(l, pl.loggers.TensorBoardLogger)]
    [tb_logger.log_hyperparams(self.hparams, init_val_metrics) for tb_logger in tb_loggers]

    # Append evaluator callbacks for all stages
    return [EvaluatorTrainCallbacks(self.train_evaluator),
            EvaluatorValCallbacks(self.val_evaluator),
            EvaluatorTestCallbacks(self.test_evaluator),
            ]
    

  def configure_optimizers(self):
    opt_cfg = OptimizerConfigurator(self.parameters, self.cfg)
    
    self.criterion = opt_cfg.get_criterion().to(self.device)
    optimizer = opt_cfg.get_optimizer()
    lr_scheduler = opt_cfg.get_lr_scheduler()
    return [optimizer], [lr_scheduler]

  #############################################
  ### loop steps
  #############################################
  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    labels = batch[-1]['labels']
    # Run RNN segmentation on image sequence
    preds = self(batch)
    # compute loss for all outputs with labels
    loss = self.criterion(preds[-1]['out'], labels)

    return {'loss':loss, 'preds':preds[-1], 'labels':labels}
    
  def validation_step(self, batch, batch_idx):
    labels = batch[-1]['labels']
    # Run RNN segmentation on image sequence
    preds = self(batch)
    # compute loss for all outputs with labels
    loss = self.criterion(preds[-1]['out'], labels)
    
    return {'preds':preds[-1], 'labels':labels, 'loss':loss}
  
  def test_step(self, batch, batch_idx):

    labels = batch[-1]['labels']
    depth = batch[-1]['depth']
    # Run RNN segmentation on image sequence
    preds = self(batch)
    
    return {'preds':preds[-1], 'labels':labels, 'depth': depth}