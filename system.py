sudo useradd -m user-name 

sudo passwd user-name


安装anaconda

wget -O - https://www.anaconda.com/distribution/ 2>/dev/null | sed -ne 's@.*\(https:\/\/repo\.anaconda\.com\/archive\/Anaconda3-.*-Linux-x86_64\.sh\)\">64-Bit (x86) Installer.*@\1@p' | xargs wget


zip -r <output_file> <folder_1> <folder_2> ... <folder_n>


指定显卡来进行训练：
1. 直接在终端指定：
CUDA_VISIBLE_DEVICES=1 python my_script.py

2. python代码中指定：
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

3. 使用函数set_device:
import torch
torch.cuda.set_device(id)
该函数见 pytorch-master\torch\cuda\__init__.py