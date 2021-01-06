apt update
apt install -y ffmpeg libsm6 libxext6
pip install pyyaml numpy pandas matplotlib visdom tensorflow-gpu tensorboardX opencv-python
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
