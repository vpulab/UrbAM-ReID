# [UrbAM-ReID Dataset](https://drive.google.com/drive/folders/1yvZY_EKlkSvUy-By5KvSUB5nUxXamM1D?usp=sharing)


UrbAM-ReID is a long-term geo-positioned urban ReID dataset. It is composed by four subdatasets recording the same trajectory at the UAM Campus, each one recorded in different seasons and including an inverse direction recording.  This work specifically addresses urban objects re-identification, currently, waste containers, rubbish bins, and crosswalks. The dataset provides different attributes of the annotated objects, like their classes, their foreground or background status and the geo-position. Several evaluation configurations can be defined to simulate realistic scenarios that may arise in actual situations within the management of urban elements, considering the utilization of just visual data, or incorporating additional attributes, providing different complexity levels. Finally, the dataset is used for defining a benchmark where two open source state-of-the-art systems are evaluated.

## link of UrbAM-ReID dataset
 [Link to the dataset](https://doi.org/10.5281/zenodo.10628486)
## Supplementary material
 It is available in [Supplemetary_material.pdf](./Supplementary_material.pdf)
## Code

This porposal utilizes two open-source state-of-the-art works:

1. [Part-Aware Transformer](https://github.com/liyuke65535/Part-Aware-Transformer)
2. [ReID Strong Baseline](https://github.com/michuanhaohao/reid-strong-baseline)
Enivronments are the described in previous links. We have indicate in environmentBOT.yml and environmentPAT.yml the environments for a TITANRTX with CUDA version 11.0, using Python versions 3.8.18 and 3.10.0, respectively.
### Modified Codes
The folder includes modified codes to integrate the UrbAM-ReID dataset into each system. In case of BoT code, indicate the data root in configs/softmax_triplet_XXX.yml in DATASETS.ROOT_DIR.
### Download BOT resnet50 model
resnet50: https://download.pytorch.org/models/resnet50-19c8e357.pth

### Evaluation and train per each split
To evaluate the system, follow the instructions in each GitHub repository.
In case of BOT code, indicate the data path in softmax_triplet_UAM.yml softmax_triplet_UAM_test.yml in DATASETS.ROOT_DIR.

After training, use the `update.py` (or `update_GPS.py` for GPS post-processing) script to infer the data. The results are saved in the output file track.txt. Evaluate the obtained `track.txt` results using `Evaluate_UrbAM-ReID.py`.
#### Example of BOT `train.py`
```bash
python tools/train.py --config_file="configs/softmax_triplet_UAM.yml" MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('UAM')" OUTPUT_DIR "('local rute to store BOT results...../Results/')"
```
#### Example of BOT `update.py`
```bash
python tools/update.py --config_file=./configs/SOA/softmax_triplet_UAM_test.yml --track="local rute to store BOT results...../Results/track.txt" MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('UAM_test')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('local rute to store BOT results...../Results/resnet50_model_100.pth')"
```
#### Example of BOT `update_GPS.py`
```bash
python tools/update_GPS.py --config_file=./configs/SOA/softmax_triplet_UAM_test.yml --track=./your path to save checkpoints and logs for each split/track.txt --xml_dir_gallery=test_label_all.xml --xml_dir_query=query_label_all.xml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('UAM_test')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('./your path to save checkpoints and logs for each split/resnet50_model_100.pth')"
```
#### Evaluate BOT `Evaluate_UrbAM-ReID.py`
```bash
python  Evaluate_UrbAM-ReID.py --track "local rute to store BOT results...../Results/track.txt" --path "local rute to dataset...../ICIP_UrbAM-ReID/splits/Containers/containers/"
```
#### Example of PAT `train.py`
```bash
python train.py --config_file "config/UAM_containers.yml"
```
#### Example of PAT `update.py`
```bash
python update.py --config_file "config/UAM_containers_test.yml" --track="local rute to store PAT results....../Results/track.txt"
```
#### Example of PAT `update_GPS.py`
```bash
python  update_GPS.py --config_file "config/SoA/UAM_containers_test.yml" --track ./your path to save checkpoints and logs for each split/trackGPS --xml_dir_gallery test_label_all.xml --xml_dir_query query_label_all.xml
```
#### Evaluate PAT `Evaluate_UrbAM-ReID.py`
```bash
python Evaluate_UrbAM-ReID.py --track="local rute to store PAT results...../Results/track.txt" --path="local rute to dataset...../ICIP_UrbAM-ReID/splits/Containers/containers/"
```
## GPS Annotations

The `./GPSannotations` directory is divided into four subfolders, each related to a sequence (March, May, July, and July inverse). Each subfolder contains a CSV file with GPS annotations.

- The first row is a header indicating the columns: `frame`, `lat`, `long`, `azi`.
- The subsequent rows of the file provide information on GPS coordinates per frame.
- GPS annotation is included in this proposal as a post-processing step, but the possibilities of use are extensive.

## ReID Sequences

The `./ReIDSequences` folders contain annotated files obtained from CVAT `annotations.xml` with all the attributes, and the `images_output` with the format `XXXXXX_Y_Z.jpg`, where XXXXX is the original frame name, Z is the annotated ID, and Y is the number of this identity in this location. In the case of containers, the same identity is considered per the same container class in the same location, and it enumerates the number of containers of the same class in this location.

## Splits

The `./splits` directory collects five different splits per urban object.

- For each split, it has the baseline data (`./image_train`, `./image_test`, `image_query`).
- In `./FG`, there are just the foreground annotated objects for the baseline.
- In `./inv`, it contains the baseline data adding the inverse as a query (the baseline query now is in `image_test`).
- In `./inv/FG`, there are just the foreground annotated objects for the data including the inverse.

Each data scenario includes the following files with annotations:

- `train_label.xml`, `test_label.xml`, `query_label.xml`: Includes cameraID (each ID=1 is May sequence, ID=2 is March, ID=3 is July, and ID=4 is July inverse), image name related to this split, and object ID.
- `train_label_all.xml`, `test_label_all.xml`, `query_label_all.xml`: Includes the same cameraID; in this case, the imageName is the frame of the original sequence (the same as "frame" in GPS .csv, and in XXXXXX in image output, and the same as in `ReIDSequences/../annotations.xml` attribute image name). The rename is the image name in this folder (in the case of the `FG` folder, it also appears as `rename2` due to it is related to the baseline folder).


<p align="center">
  <img src=./map.png width=50% height=50%>
 
## Acknowledgment

This work has been supported by the Ministerio de Ciencia, Innovación y Universidades of the Spanish Government under projects SEGA-CV (TED2021-131643A-I00) and HVD (PID2021-125051OB-I00).

## Citation
To cite this work, please use the following reference.
```bash
@INPROCEEDINGS{10647759,
  author={Moral, Paula and García-Martín, Álvaro and Martínez, José M.},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Long-Term Geo-Positioned Re-Identification Dataset of Urban Elements}, 
  year={2024},
  volume={},
  number={},
  pages={124-130},
  keywords={Waste management;Visualization;Image processing;Vegetation mapping;Benchmark testing;Complexity theory;Trajectory;Recording;Maintenance;Monitoring;Re-Identification;Deep Learning;Computer Vision;Urban Environment},
  doi={10.1109/ICIP51287.2024.10647759}}
```
