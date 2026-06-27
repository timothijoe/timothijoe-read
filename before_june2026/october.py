spc run dev zt-dev2 -N xlab -i registry.sensetime.com/sensephoenix/cuda9.0-cudnn7.6.5-py3.6-devel-centos7-ofed5.1  --cpus 24     --mems 48Gi     --gpus 1 -v /mnt/lustre/zhoutong:/mnt/lustre/zhoutong:rw

 registry.sensetime.com/xlab/ding:cuda-nightly-dev

export http_proxy=http://172.16.1.135:3128/ ; export https_proxy=http://172.16.1.135:3128/ ; export HTTP_PROXY=http://172.16.1.135:3128/ ; export HTTPS_PROXY=http://172.16.1.135:3128/

开发机配置教程：
1. 代理配置： export http_proxy=http://172.16.1.135:3128/ ; export https_proxy=http://172.16.1.135:3128/ ; export HTTP_PROXY=http://172.16.1.135:3128/ ; export HTTPS_PROXY=http://172.16.1.135:3128/
2. spc list dev-machine用来查看dev machine
3. spc list dev -R 查看配置
4. spc cancel dev-machine zt-dev删除某个dev machine
5.创建spc run dev zt-dev2 -N xlab -i registry.sensetime.com/sensephoenix/cuda9.0-cudnn7.6.5-py3.6-devel-centos7-ofed5.1  --cpus 24     --mems 48Gi     --gpus 1 -v /mnt/lustre/zhoutong:/mnt/lustre/zhoutong:rw
6. 链接教程 https://confluence.sensetime.com/pages/viewpage.action?pageId=359907567


kubectl get dev查看开发机所在的机器


pip install gym
pip install ale-py
pip install autorom==0.7.0
AutoROM --accept-license


spc update dev-machine zt-dev2 --gpus 1


wget https://repo.anaconda.com/archive/Anaconda2-2019.07-Linux-x86_64.sh

wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
sys.path.append('/home/zhoutong/di_base_folder/LightZero')