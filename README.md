# UrbAM-ReID Dataset
![UrbAM-ReID Dataset](./SamevsDifferentID.pdf)
UrbAM-ReID is a long-term geo-positioned urban ReID dataset. It is composed by four subdatasets recording the same trajectory at the UAM Campus, each one recorded in different seasons and including an inverse direction recording.  This work specifically addresses urban objects re-identification, currently, waste containers, rubbish bins, and crosswalks. The dataset provides different attributes of the annotated objects, like their classes, their foreground or background status and the geo-position. Several evaluation configurations can be defined to simulate realistic scenarios that may arise in actual situations within the management of urban elements, considering the utilization of just visual data, or incorporating additional attributes, providing different complexity levels. Finally, the dataset is used for defining a benchmark where two open source state-of-the-art systems are evaluated.  

## Code

This porposal utilizes two open-source state-of-the-art works:

1. [Part-Aware Transformer](https://github.com/liyuke65535/Part-Aware-Transformer)
2. [ReID Strong Baseline](https://github.com/michuanhaohao/reid-strong-baseline)

### Modified Codes
The folder includes modified codes to integrate the UrbAM-ReID dataset into each system.

### Evaluation
To evaluate the system, follow the instructions in each GitHub repository. After training, use the `update.py` (or `update_GPS.py` for GPS post-processing) script to infer the data. Evaluate the obtained `track.txt` results using `Evaluate_UrbAM-ReID.py`.

#### Example of BOT `update.py`
```bash
python update.py --config_file=./configs/SOA/softmax_triplet_UAM_test_cross1.yml --track=./checkpoints/crosswalk/crosswalk13new/track2_model1UAM_tradicional.txt MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('UAM_test')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('./checkpoints/crosswalk/crosswalk13new/resnet50_model_100.pth')"
```
#### Example of BOT `update_GPS.py`
```bash
python update_GPS.py --config_file=./configs/SOA/softmax_triplet_UAM_test_cross1.yml --track=./checkpoints/crosswalk/crosswalk13new/track2_model1UAM_tradicionalGPS.txt --xml_dir_gallery=test_label_all.xml --xml_dir_query=query_label_all.xml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('UAM_test')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('./checkpoints/crosswalk/crosswalk13new/resnet50_model_100.pth')"
```
#### Example of PAT `update.py`
```bash
python update.py --config_file "config/UAM_cross1_test.yml" --track ./logs/UAM/UAM_cross1/track2_model1UAM_tradicional

```
#### Example of PAT `update_GPS.py`
```bash
python  update_GPS.py --config_file "config/SoA/UAM_cross1_test.yml" --track ./logs/UAM/UAM_cross1/track2_model1UAM_tradicionalGPS --xml_dir_gallery test_label_all.xml --xml_dir_query query_label_all.xml
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
