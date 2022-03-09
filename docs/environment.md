# Reproduce mv-model-building-gui environment
In order to reproduce the environment used for 
all experiments and simulations of the repo 
(mv-model-building-gui.yml), you just need to open 
terminal and run the following commands:
```
conda create -n mv-model-building-gui python=3.9
conda activate mv-model-building-gui
conda install pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyyaml=6.0
pip install timm==0.5.4
pip install pandas==1.4.1
pip install matplotlib==3.5.1
conda install -c conda-forge tensorflow=2.7
pip install keras_vggface==0.6
pip install keras_applications==1.0.8 --no-deps
pip install keras_preprocessing==1.1.0 --no-deps
pip install keras==2.4.3
pip install mtcnn==0.1.1
```