echo Creating trained models path...

mkdir -p trained_models
cd trained_models

echo Spatial Temporal Attention model SB_20
wget -O st_atte_sb20.zip -c https://uni-bonn.sciebo.de/s/UMo4xJUu04Ek59U/download

echo Extracting model...
unzip st_atte_sb20.zip

cd ..