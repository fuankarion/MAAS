# Envirrorment
conda create -n maas_env pytorch=1.8.1 torchvision torchaudio cudatoolkit -c pytorch -c nvidia -y

source activate maas_env

pip uninstall Pillow -y
pip install Pillow-SIMD

pip install python_speech_features
pip install natsort
pip install scipy
pip install sklearn

conda install pytorch-geometric -c rusty1s -c conda-forge
