import importlib
import yaml
import click
from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from modules.debugger import NetDebugger
from modules.img_logger import  TestOutputLogger

root_path = Path(__file__).parent.absolute()
models_path = Path(__file__).parent.absolute()


@click.command()
@click.argument('config')
@click.option('--gpus',
              '-g',
              type=int,
              default=0)
@click.option('--data_path',
              '-d',
              type=str,
              default='')
@click.option('--short_dev_test_only',
              '-t',
              is_flag=True)
def main(config, gpus, data_path, short_dev_test_only):
    # load config file
    with open(config, 'r') as fid:
        cfg = yaml.safe_load(fid)#, Loader=yaml.FullLoader)
    
    if data_path: cfg['dataset']['yaml_path'] = data_path

    if short_dev_test_only: cfg['debugger']['fast_dev_run'] = True

    test_model(cfg, gpus)

def test_model(cfg, gpus):
    callbacks = []

    timestamp = datetime.now().strftime('%H%M_%d%m%Y')
    experiment_id = '_'+cfg['experiment_id'] if 'experiment_id' in cfg.keys() else ''
    experiment_name = f'{cfg["network"]["model"]}{experiment_id}_test_{timestamp}'

    print(f'\n####\nRunning experiment:\n{experiment_name}\n####\n')


    if not cfg['network']['pretrained'] and not 'resume_from_ckpt' in cfg['trainer'].keys():
        raise ValueError('Neither pretrained path nor resume_from_ckpt specified in config file. Unable to test model')

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

    if cfg['logger']['test_output_enable']:
        test_out_path = Path(cfg['logger']['test_output_log_path']) if 'test_output_log_path' in cfg['logger'].keys() else Path(cfg['logger']['log_path']) / experiment_name / 'output'
        callbacks.append(TestOutputLogger(str(test_out_path)))

    
    
    #################################
    ### debugger
    #################################
    if cfg['debugger']['enable']:
        callbacks.append(NetDebugger(cfg))     

    trainer = Trainer(gpus=gpus,
                      accelerator="ddp" if gpus > 1 else None,
                      
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

    ###### Testing
    trainer.test(model, test_dataloaders=data.test_dataloader())

if __name__ == "__main__":
    main()
