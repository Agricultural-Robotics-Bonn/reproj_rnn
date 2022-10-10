# Spatial-Temporal DNN for Agriculture (ReprojRNN)


### [Paper](https://arxiv.org/pdf/2206.13406.pdf) | [Data](http://agrobotics.uni-bonn.de/data/)

[Explicitly incorporating spatial information to recurrent networks for agriculture](https://arxiv.org/pdf/2206.13406.pdf)  
 [Claus Smitt](http://agrobotics.uni-bonn.de/claus-g-smitt/),
 [Michael Halstead](http://agrobotics.uni-bonn.de/michael-halstead/),
 [Alireza Ahmadi](http://agrobotics.uni-bonn.de/alireza-ahmadi/),
 [Chris McCool](https://sites.google.com/site/christophersmccool/)
 
 [Agricultural Robotics & Engineering](http://agrobotics.uni-bonn.de/),
 Institute of Agriculture, University of Bonn

To be presented at IROS 2022 (Best AgRobotics Paper Nomination)

<img src='imgs/stNetworks._simplified.png'/>

## TL;DR - Test Pre-Trained model

```
python3 -m venv ".venv"
. .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

bash get_bup20.sh                 # ~70GB
bash get_st_atte_model_bup20.sh   # ~500MB

python test.py \
  trained_models/st_atte_bup20/config.yaml \
  -g 1 \
  --data_path ~/datasets/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.yaml

```

If all went well, you should get the list of all metrics when the model finishes testing.

**Note**: Change the torch version in the `requirements.txt` file according to your CUDA version in case you get version errors

## How does it work?

<img src='imgs/reprojLayer.png'/>

## Setup

Tested on Ubuntu 18.06; CUDA 11.3

### Pre-requisites
```
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
sudo apt-get install -y python3-venv
```

Install [CUDA and cuDDN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

### Python virtual environment setup

```
python3 -m venv ".venv"
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets used
## Train your own model

Folder `./config` has several `yaml` config examples used in the paper that can be used as examples
```
python train.py \  
  -g [num_gpus] \
  --out_path /net/outputs/save/path
  --log_path /train/logs/save/path
  --ckpt_path /model/checkpoints/save/path
  --data_path /dataset/yaml/file/location
```
The training spript uses [`PytorchLightning` DDP](https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html#distributed-data-parallel) pluging for multi GPU training.

**Note**: The training process is quiet memory intensive due to the recurrent nature of the models (trained on Nvidia RTX A6000). I case you get an out of memory error, try making the following changes to your `yaml` config file:
  - Reduce `dataloader/batch_size`
  - Reduce `dataloader/sequencing/num_frames` to use shorter frame sequences
  - Set `/trainet/precision` to 16 

This would likely give you different results from the ones reported in the paper but, you'll be able to train the model on you own.

### Prepare your own dataset
One easy way to 
## Citation
```
@article{smitt2022explicitly,
  title={Explicitly incorporating spatial information to recurrent networks for agriculture},
  author={Smitt, Claus and Halstead, Michael and Ahmadi, Alireza and McCool, Chris},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
}
```