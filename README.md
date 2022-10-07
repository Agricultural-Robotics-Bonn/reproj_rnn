# reproj_rnn
Explicitly incorporating spatial information to recurrent networks for agriculture

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