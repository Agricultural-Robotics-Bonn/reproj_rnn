echo Creating trained models path...

mkdir -p trained_models
cd trained_models

echo Spatial Temporal Attention model BUP_20
wget -O st_atte_bup20.zip -c https://uni-bonn.sciebo.de/s/BelFXtc4jgycDTP/download

echo Extracting model...
unzip st_atte_bup20.zip

cd ..