<div align="center">
  <img src="images/logo.png" width="200"/>
</div>



# V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion 
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.08402)
## :balloon: Introduction
:wave: This is the official repository for the V2X-R, including the V2X-R dataset and the implementation of the benchmark model, and MDD module. 


## :balloon: V2X-R Dataset Manual 
The first V2X dataset incorporating LiDAR, camera, and **4D radar**. V2X-R contains **12,079 scenarios** with **37,727 frames of LiDAR and 4D radar point clouds**, **150,908 images**, and **170,859 annotated 3D vehicle bounding boxes**.
<div align="center">
  <img src="images/radar_sup.png" width="600"/>
</div>



### Dataset Collection
Thanks to the [CARLA](https://github.com/carla-simulator/carla) simulator and the [OpenCDA](https://github.com/ucla-mobility/OpenCDA) framework, our V2X-R simulation dataset was implemented on top of them. In addition, our dataset route acquisition process partly references [V2XViT](https://github.com/DerrickXuNu/v2x-vit), which researchers can reproduce according to the data_protocol in the dataset.

### Download and Decompression
:ledger: The data can be found from this URL. 

Since the data is large (including 3xLiDAR{normal, fog, snow}, 1xradar, 4ximages for each agent). We have compressed the sequence data of each agent, you can refer to this code for batch decompression after downloading.


### Structure
:open_file_folder: After download and decompression are finished, the dataset is structured as following:

```sh
V2X-R # root path of v2x-r
├── train
│   ├──Sequence name (time of data collection, e.g. 2024_06_24_20_24_02)
│   │   ├──Agent Number ("-1" denotes infrastructure, otherwise is CAVs)
│   │   │   ├──Data (including the following types of data)
│   │   │   │ Timestamp.Type, eg.
│   │   │   │ - 000060_camerai.png (i-th Camera),
│   │   │   │ - 000060.pcd (LiDAR),
│   │   │   │ - 000060_radar.pcd (4D radar),
│   │   │   │ - 000060_fog.pcd (LiDAR with fog simulation),
│   │   │   │ - 000060_snow.pcd (LiDAR with snow simulation),
│   │   │   │ - 000060.yaml (LiDAR with fog simulation)
├── validate
│   ├──...
├── test
│   ├──...

```

### Calibration
We provide calibration information for each sensor (LiDAR, 4D radar, camera) of each agent for inter-sensor fusion. In particular, the exported 4D radar point cloud has been converted to the LiDAR coordinate system of the corresponding agent in advance to facilitate fusion, so the 4D radar point cloud is referenced to the LiDAR coordinate system.


## :balloon: Benchmark and Models Zoo

## :balloon: Quickly Get Started
For installation , model training/testing, and use of the MDD module refer to [document](V2X-R/README.md)
