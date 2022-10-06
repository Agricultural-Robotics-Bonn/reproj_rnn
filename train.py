import importlib
import yaml
import click
from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from modules.debugger import NetDebugger

root_path = Path(__file__).parent.absolute()
models_path = Path(__file__).parent.absolute()


@click.command()
@click.argument('config')
@click.option('--gpus',
              '-g',
              type=int,
              default=0)
@click.option('--out_path',
              '-o',
              type=str,
              default='')
@click.option('--log_path',
              '-l',
              type=str,
              default='')
@click.option('--ckpt_path',
              '-c',
              type=str,
              default='')
@click.option('--data_path',
              '-d',
              type=str,
              default='')
@click.option('--short_dev_test_only',
              '-t',
              is_flag=True)
def main(config, gpus, out_path, log_path, ckpt_path, data_path, short_dev_test_only):
    # load config file
    with open(config, 'r') as fid:
        cfg = yaml.safe_load(fid)#, Loader=yaml.FullLoader)

    cfg['logger']['log_path'] = log_path if log_path else out_path
    cfg['trainer']['save_checkpoints']['path'] = ckpt_path if ckpt_path else out_path
    
    if data_path: cfg['dataset']['yaml_path'] = data_path

    if short_dev_test_only:
        cfg['debugger']['fast_dev_run'] = True

    train_model(cfg, gpus)

def train_model(cfg, gpus):
    callbacks = []

    timestamp = datetime.now().strftime('%H%M%S_%d%m%Y')
    
    experiment_id = '_'+cfg['experiment_id'] if 'experiment_id' in cfg.keys() else ''
    experiment_name = f'{cfg["network"]["model"]}{experiment_id}_train_{timestamp}'

    print(f'\nRunning experiment:\n{experiment_name}\n')

    #################################
    ### Load Model
    #################################
    #  import selected network module
    networkModule = importlib.import_module('models.'+
                                        cfg["network"]["model"]+
                                        '.module')

    # Instantiate network Pytorch Lightning Module
    #Load pretrained weights if required
    if cfg['network']['pretrained']:
        if(Path(cfg['network']['pretrained_path']).suffix == '.ckpt'):
            model = networkModule.PLModule.load_from_checkpoint(cfg['network']['pretrained_path'],strict=False, cfg=cfg)
        else:
            model = networkModule.PLModule(cfg)
            pretrain = torch.load(cfg['network']['pretrained_path'], map_location='cpu')
            model.load_state_dict(pretrain['state_dict'],strict=True)
    else:
        model = networkModule.PLModule(cfg)
    #################################
    ### Load Data
    #################################
    # Instanciate dataset Lightning Datamodule
    dataModule = importlib.import_module('dataloaders.' + cfg["dataloader"]["name"])
    data = dataModule.Parser(cfg)

    #################################
    ### Logger
    #################################
    loggers = []
    if cfg['logger']['tb_enable']:
        if 'tb_log_path' in cfg['logger'].keys():
            tb_log_path = cfg['logger']['tb_log_path'] 
            tb_log_name = cfg["network"]["model"]+experiment_id
            tb_log_version = timestamp
        else:
            tb_log_path = str(Path(cfg['logger']['log_path']) / experiment_name)
            tb_log_name = None
            tb_log_version = 'logs'
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_path,
                                                 name=tb_log_name,
                                                 version=tb_log_version,
                                                 default_hp_metric=False)
        loggers.append(tb_logger)
   

    if cfg['logger']['csv_enable']:
        if 'csv_log_path' in cfg['logger'].keys():
            csv_log_path = cfg['logger']['csv_log_path'] 
            csv_log_name = cfg["network"]["model"]+experiment_id
            csv_log_version = timestamp
        else:
            csv_log_path = str(Path(cfg['logger']['log_path']) / experiment_name)
            csv_log_name = None
            csv_log_version = 'logs'
        csv_logger = pl_loggers.CSVLogger(csv_log_path, 
                                          name=csv_log_name, 
                                          version=csv_log_version)
        loggers.append(csv_logger)

    if cfg['logger']['log_lr']:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    #################################
    ### debugger
    #################################
    if cfg['debugger']['enable']:
        callbacks.append(NetDebugger(cfg))     

    #################################
    ### Save checkpoints
    #################################
    # Save model with best val_loss
    if cfg['trainer']['save_checkpoints']['enable']:
        checkpoints_path = str(Path(cfg['trainer']['save_checkpoints']['path']) / experiment_name / 'checkpoints')
        
        if 'best_metric' in cfg['trainer']['save_checkpoints'].keys():
            callbacks.append(ModelCheckpoint(monitor=cfg['trainer']['save_checkpoints']['best_metric'],
                                            mode=cfg['trainer']['save_checkpoints']['best_metric_mode'],
                                            dirpath=checkpoints_path,
                                            filename="best_%s_{epoch:03d}" % cfg['trainer']['save_checkpoints']['best_metric'] ))

        if 'every_n_val_epochs' in cfg['trainer']['save_checkpoints'].keys():
            callbacks.append(ModelCheckpoint(period=cfg['trainer']['save_checkpoints']['every_n_val_epochs'] * cfg['trainer']['val_every_n_epochs'],
                                            mode=cfg['trainer']['save_checkpoints']['best_metric_mode'],
                                            dirpath=checkpoints_path,
                                            save_top_k=-1,
                                            filename='checkpoint_epoch_{epoch:03d}'))

    #################################
    ### Early stopping
    #################################
    if cfg['trainer']['early_stopping']['enable']:
        callbacks.append(EarlyStopping(monitor=cfg['trainer']['early_stopping']['monitor_metric'],
                                       mode=cfg['trainer']['early_stopping']['mode'],
                                       min_delta=cfg['trainer']['early_stopping']['min_delta'],
                                       patience=cfg['trainer']['early_stopping']['patience'],
                                       strict=cfg['trainer']['early_stopping']['strict'],
                                       verbose=cfg['trainer']['early_stopping']['verbose']
                                       ))

    # log number of GPUs in config dict
    num_gpus = len(gpus) if isinstance(gpus,(list,tuple)) else gpus
    accelerator = "ddp" if num_gpus >= 2 else None
    
    if num_gpus >= 2 and 'step_size' in cfg['optimizer'].keys():
        cfg['optimizer']['step_size'] = cfg['optimizer']['step_size'] // 2

    # log number of GPUs in config dict
    cfg["trainer"]["num_gpus"] = num_gpus

    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      plugins=DDPPlugin(find_unused_parameters=False) if gpus > 1 else None,
                      
                      accumulate_grad_batches=cfg['dataloader']['accumulate_grad_batches'],

                      max_epochs= cfg['trainer']['max_epochs'],

                      check_val_every_n_epoch=cfg['trainer']['val_every_n_epochs'],
                      precision= cfg['trainer']['precision'] if 'precision' in cfg['trainer'].keys() else 32,
                      resume_from_checkpoint=cfg['trainer']['resume_from_ckpt'] if 'resume_from_ckpt' in cfg['trainer'].keys() else None,
                      
                      logger=loggers,
                      callbacks=callbacks,
                      
                      fast_dev_run=cfg['debugger']['fast_dev_run'],
                      )

    # save config file in log directory now that the trainer is configured and ready to run
    if trainer.local_rank == 0 and cfg['logger']['log_cfg_file']:
        cfg_log_path = Path(cfg['logger']['log_path']) / experiment_name / 'config.yaml'
        cfg_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_log_path, 'w') as fid:
            yaml.dump( cfg, fid, sort_keys=False)

    ###### Learning rate finder
    if 'mode' in cfg['trainer'] and cfg['trainer']['mode'] == 'find_lr':
        lr_finder = trainer.tuner.lr_find(model,data)
        fig = lr_finder.plot(suggest=True)
        print(f'suggested initial lr{lr_finder.suggestion()}')
        fig.savefig('lr_finder.png')
        return

    ###### Training
    trainer.fit(model, data)
    # trainer.test()

if __name__ == "__main__":
    main()
