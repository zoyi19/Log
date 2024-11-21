


## Get Started

### Install
#### 1. Clone (or download) the source code 
```
git clone https://github.com/ylwhxht/V2X-R.git
cd V2X-R/V2X-R
```
 
#### 2. Create conda environment and set up the base dependencies
```
conda create --name v2xr python=3.7 cmake=3.22.1
conda activate v2xr
conda install cudnn -c conda-forge
conda install boost
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

##### *(Option) If there is error or speed issues in install cudatoolkit
```
# could instead specify the PATH, CUDA_HOME, and LD_LIBRARY_PATH, using current cuda write it to ~/.bashrc, for example use Vim
vim ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda/bin:$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# add head file search directories 
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/Anaconda3/envs/bm2cp/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/Anaconda3/envs/bm2cp/include
# add shared library searching directories
export LIBRARY_PATH=$LIBRARY_PATH:/Anaconda3/envs/bm2cp/lib
# add runtime library searching directories
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Anaconda3/envs/bm2cp/lib

# go out of Vim and activate it in current shell
source ~/.bashrc

conda activate bm2cp
```

### 3. Install spconv (Support both 1.2.1 and 2.x)

##### *(Notice): Make sure *libboost-all-dev* is installed in your linux system before installing *spconv*. If not:
```
sudo apt-get install libboost-all-dev
```

##### Install 2.x
```
pip install spconv-cu113
```

### 4. Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```

### 5. Install V2XR
```
# install requirements
pip install -r requirements.txt
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace

# FPVRCNN's iou_loss dependency (optional)
python opencood/pcdet_utils/setup.py build_ext --inplace
```

### 6. *(Option) for training and testing SCOPE&How2comm
```
# install basic library of deformable attention
git clone https://github.com/TuSimple/centerformer.git
cd centerformer

# install requirements
pip install -r requirements.txt
sh setup.sh
```

##### if there is a problem about cv2:
```
# module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
pip install opencv-python install "opencv-python-headless<4.3"
```


#### Train your model
First of all, modify the dataset path in the setting file, i.e. `xxx.yaml`.
```
data_dir: "{YOUR PATH}/DAIR-V2X-C/cooperative-vehicle-infrastructure"
root_dir: "{YOUR PATH}/DAIR-V2X-C/cooperative-vehicle-infrastructure/train.json"
validate_dir: "{YOUR PATH}/DAIR-V2X-C/cooperative-vehicle-infrastructure/val.json"
test_dir: "{YOUR PATH}/DAIR-V2X-C/cooperative-vehicle-infrastructure/val.json"
```

The setting is same as OpenCOOD, which uses yaml file to configure all the parameters for training. To train your own model from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

For example, to train BM2CP from scratch:
```
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_bm2cp.yaml
```

To train BM2CP from a checkpoint:
```
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/dair-v2x/dair_bm2cp.yaml --model_dir opencood/logs/dair_bm2cp_2023_11_28_08_52_46
```

#### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --eval_epoch ${epoch_number} --save_vis ${default False}
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', 'intermediate', 'no'(indicate no fusion, single agent), 'intermediate_with_comm'(adopt intermediate fusion and output the communication cost).
- `eval_epoch`: int. Choose to inferece which epoch.
- `save_vis`: bool. Wether to save the visualization result.

The evaluation results  will be dumped in the model directory.

## Citation
If you are using our project for your research, please cite the following paper:

```

@InProceedings{zhao2023bm,
  title = {BM2CP: Efficient Collaborative Perception with LiDAR-Camera Modalities},
  author = {Zhao, Binyu and ZHANG, Wei and Zou, Zhaonian},
  booktitle = {Proceedings of The 7th Conference on Robot Learning},
  pages = {1022--1035},
  year = {2023},
  series = {Proceedings of Machine Learning Research},
}
```

## Acknowledgements
Thank for the excellent cooperative perception codebases [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [CoPerception](https://github.com/coperception/coperception) and [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm).

Thank for the excellent cooperative perception datasets [DAIR-V2X](https://thudair.baai.ac.cn/index), [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) and [V2X-SIM](https://ai4ce.github.io/V2X-Sim/).

Thank for the dataset and code support by [DerrickXu](https://github.com/DerrickXuNu), [Yue Hu](https://github.com/MediaBrain-SJTU) and [YiFan Lu](https://github.com/yifanlu0227).

## Relevant Projects

Thanks for the insightful previous works in cooperative perception field.

### Methods

**V2VNet: Vehicle-to-vehicle communication for joint perception and prediction** 
*ECCV20* [[Paper]](https://arxiv.org/abs/2008.07519) 

**When2com: Multi-agent perception via communication graph grouping** 
*CVPR20* [[Paper]](https://arxiv.org/abs/2006.00176) [[Code]](https://arxiv.org/abs/2006.00176)

**Learning Distilled Collaboration Graph for Multi-Agent Perception** 
*NeurIPS21* [[Paper]](https://arxiv.org/abs/2111.00643) [[Code]](https://github.com/DerrickXuNu/OpenCOOD)

**V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer** *ECCV2022* [[Paper]](https://arxiv.org/abs/2203.10638) [[Code]](https://github.com/DerrickXuNu/v2x-vit) [[Talk]](https://course.zhidx.com/c/MmQ1YWUyMzM1M2I3YzVlZjE1NzM=)

**Self-Supervised Collaborative Scene Completion: Towards Task-Agnostic Multi-Robot Perception** 
*CoRL2022* [[Paper]](https://openreview.net/forum?id=hW0tcXOJas2)

**CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers** *CoRL2022* [[Paper]](https://arxiv.org/abs/2207.02202) [[Code]](https://github.com/DerrickXuNu/CoBEVT)

**Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps** *NeurIPS2022* [[Paper]](https://arxiv.org/abs/2209.12836) [[Code]](https://github.com/MediaBrain-SJTU/Where2comm)

**Spatio-Temporal Domain Awareness for Multi-Agent Collaborative Perception** *ICCV2023* [[Paper]](https://arxiv.org/abs/2307.13929)[[Code]](https://github.com/starfdu1418/SCOPE)

**How2comm: Communication-Efficient and Collaboration-Pragmatic Multi-Agent Perceptio** *NeurIPS2023* [[Paper]](https://openreview.net/pdf?id=Dbaxm9ujq6) [[Code]](https://github.com/ydk122024/How2comm)


### Datasets

**OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication** 
*ICRA2022* [[Paper]](https://arxiv.org/abs/2109.07644) [[Website]](https://mobility-lab.seas.ucla.edu/opv2v/) [[Code]](https://github.com/DerrickXuNu/OpenCOOD)

**V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving** 
*RAL21* [[Paper]](https://arxiv.org/abs/2111.00643) [[Website]](https://ai4ce.github.io/V2X-Sim/)[[Code]](https://github.com/ai4ce/V2X-Sim)

**DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection** *CVPR2022* [[Paper]](https://arxiv.org/abs/2204.05575) [[Website]](https://thudair.baai.ac.cn/index) [[Code]](https://github.com/AIR-THU/DAIR-V2X)

