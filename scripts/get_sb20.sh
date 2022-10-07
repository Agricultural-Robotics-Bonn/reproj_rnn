echo Creating datasets path...

mkdir -p datasets
cd datasets

echo Downloading SB_20 dataset...
wget -O SB20.tar.gz -c https://uni-bonn.sciebo.de/s/Eq0WVMa3y1uxB0h/download

echo Extracting dataset...
tar -xf SB20.tar.gz --checkpoint=.10000

cd ../..