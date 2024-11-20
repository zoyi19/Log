# V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion 
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.08402)

<div align="center">
  <img src="images/logo.png" width="200"/>
</div>



## :balloon: Introduction
:wave: This is the official repository for the V2X-R, including the V2X-R dataset and the implementation of the benchmark model, and MDD module. 


## :balloon: V2X-R Dataset Manual 
The first V2X dataset incorporates LiDAR, camera, and **4D radar**. V2X-R contains **12,079 scenarios** with **37,727 frames of LiDAR and 4D radar point clouds**, **150,908 images**, and **170,859 annotated 3D vehicle bounding boxes**.
<div align="center">
  <img src="images/radar_sup.png" width="600"/>
</div>



### Dataset Collection
Thanks to the [CARLA](https://github.com/carla-simulator/carla) simulator and the [OpenCDA](https://github.com/ucla-mobility/OpenCDA) framework, our V2X-R simulation dataset was implemented on top of them. In addition, our dataset route acquisition process partly references [V2XViT](https://github.com/DerrickXuNu/v2x-vit), which researchers can reproduce according to the data_protocol in the dataset.

### Download and Decompression
:ledger: The data can be found from this URL. 

Since the data is large (including 3xLiDAR{normal, fog, snow}, 1xradar, 4ximages for each agent). We have compressed the sequence data of each agent, you can refer to this code for batch decompression after downloading.
```python
import os
import subprocess
def decompress_v2x_r(root_dir, save_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.7z'):
                file_path = os.path.join(root, file)
                path = root.split('/')
                extract_path = os.path.join(save_dir, path[-2], path[-1], file[:-3])
                os.makedirs(extract_path, exist_ok=True)
                subprocess.run(['7z', 'x', '-o' + extract_path + '/', file_path])

data_directory = #downloaded dataset path e.g: '/mnt/16THDD-2/hx/V2X-R_Dataset(compressed)'
output_directory =  #output dataset path  e.g: '/mnt/16THDD-2/hx/t'
decompress_v2x_r(data_directory, output_directory)
```

### Structure
:open_file_folder: After download and decompression are finished, the dataset is structured as following:

```sh
V2X-R # root path of v2x-r output_directory 
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
We provide calibration information for each sensor (LiDAR, 4D radar, camera) of each agent for inter-sensor fusion. In particular, the exported 4D radar point cloud has been converted to the LiDAR coordinate system of the corresponding agent in advance of fusion, so the 4D radar point cloud is referenced to the LiDAR coordinate system.


## :balloon: Benchmark and Models Zoo
### 4DRadar-based Cooperative 3D Detector (no-compression)
| **Method** | **Validation (IoU=0.3/0.5/0.7)** | **Testing (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:--------------------------------:|:--------------------------------:|:-----------------------------:|:----------:|:-----------:|
|         ITSC2021:PFA-Net         |         76.90/68.00/39.30        |       85.10/79.90/52.50       |      √     | coming |
|           NIPS2022:RTNH          |         71.70/62.20/34.40        |       73.70/67.70/41.90       |      √     | coming |
|          ECCV2022:V2XViT         |         71.14/64.28/31.12        |       80.94/73.82/42.73       |      √     | coming |
|         ICRA2022:AttFuse         |         75.30/66.50/36.10        |       81.80/75.40/48.20       |      √     | coming |
|        NIPS2023:Where2comm       |         71.60/67.20/42.90        |       80.40/77.30/56.70       |      √     | coming |
|          ICCV2023:SCOPE          |         61.90/59.30/47.90        |       73.00/71.60/51.60       |      √     | coming |
|          CoRL2023:CoBEVT         |         80.20/73.40/41.10        |       85.80/80.60/52.90       |      √     | coming |
|         ICRA2023:CoAlign         |         65.80/59.20/34.70        |       76.90/70.20/46.20       |      √     | coming |
|        WACV2023:AdaFusion        |         77.84/72.48/42.85        |       82.20/78.08/55.51       |      √     | coming |
|           IROS2024:SICP          |         70.08/60.62/32.43        |       71.45/63.47/33.39       |      √     | coming |

### LiDAR-based Cooperative 3D Detector (no-compression)
| **Method** | **Validation (IoU=0.3/0.5/0.7)** | **Testing (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:------------------------------:|:--------------------------------:|:-----------------------------:|:----------:|:-----------:|
| ECCV2022:V2XViT                | 84.99/82.22/64.92                | 90.14/89.01/77.71             |      √     | coming |
| ICRA2022:AttFuse               | 86.00/82.20/66.90                | 91.40/89.60/80.20             |      √     | coming |
| NeurIPS2023:Where2comm         | 85.20/83.10/65.90                | 91.60/88.50/80.50             |      √     | coming |
| ICCV2023:SCOPE                 | 76.00/74.70/60.90                | 81.40/72.90/67.00             |      √     | coming |
| CoRL2023:CoBEVT                | 87.64/84.79/71.01                | 92.29/91.44/82.45             |      √     | coming |
| ICRA2023:CoAlign               | 89.08/87.57/80.05                | 89.59/88.89/83.29             |      √     | coming |
| ICCV:AdaFusion                 | 88.11/86.91/75.61                | 92.70/90.60/84.80             |      √     | coming |
| IROS2024:SICP                  | 81.14/77.62/58.14                | 84.64/82.17/66.71             |      √     | coming |
| WACV2024:MACP                  | 72.80/70.90/60.00                | 83.70/83.10/75.50             |      √     | coming |


### LiDAR-4D Radar based Cooperative 3D Detector (no-compression)
|       **Method**       | **Validation (IoU=0.3/0.5/0.7)** | **Testing (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:----------------------:|:--------------------------------:|:-----------------------------:|:----------:|:-----------:|
|  IROS2023:InterFusion  |         81.23/77.33/52.93        |       87.91/86.51/69.63       |      √     | coming |
|     Arxiv2024:L4DR     |         84.58/82.75/70.29        |       90.78/89.62/82.91       |      √     | coming |
|    ICRA2022:AttFuse    |         86.14/84.30/70.72        |       92.20/90.70/84.60       |      √     | coming |
|     ECCV2022:V2XViT    |         85.23/83.90/69.77        |       91.99/91.22/83.04       |      √     | coming |
|     ICCV2023:Scope     |         78.79/77.96/62.57        |       83.38/82.89/70.00       |      √     | coming |
| NeurIPS2023:Where2comm |         87.62/85.58/69.61        |       92.20/91.00/82.04       |      √     | coming |
|     CoRL2023:CoBEVT    |         87.72/87.02/78.56        |       94.00/93.21/86.68       |      √     | coming |
|    ICRA2023:CoAlign    |         87.16/85.54/73.79        |       92.14/91.21/83.77       |      √     | coming |
|   WACV2023:AdaFusion   |         89.24/87.31/74.86        |       93.10/92.15/85.48       |      √     | coming |
|      IROS2024:SICP     |         82.46/79.44/61.51        |       86.19/84.20/68.15       |      √     | coming |

## :balloon: Quickly Get Started
For installation, model training/testing, and use of the MDD module refer to [document](V2X-R/README.md)
