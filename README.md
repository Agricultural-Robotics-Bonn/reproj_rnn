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

## Setup
## Train your own

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