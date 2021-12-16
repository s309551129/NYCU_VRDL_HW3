# NYCU_VRDL_HW3
This repository is about HW3 in Visual Recognition using Deep Learning class, NYCU. The main target is to recognize nuclei and localize them. Futhermore, we also need to segment for each nuclei in pixel-wise level. It is a task about instance segmentation.
# Environment
Please install the mmdetection package from this website:
```
https://github.com/open-mmlab/mmdetection
```
Follow the steps to install mmdetection and run requirements.txt to set the environment:
```
pip3 install -r ./mmdetection/requirements.txt
```
# Prepare Dataset
You can download the dataset from codalab in-class competition:
```
https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5#participate-get_data
```
After downloading, the data directory is structured as:
```
${dataset}
  +- dataset
    +- train
    |  +- TCGA-18...
    |  +- TCGA-21...
    |  +- ...
    +- test
    |  +- TCGA-50....png
    |  +- TCGA-A7....png
    |  +- ...
    +- test_img_ids.json
${mmdetection}
  +- config.py
test_pre.py
train_pre.py
```
You can run this command to split the training data into training set and validation set, also tranfer binary mask into COCO format:
```
python3 train_pre.py
```
You can run this command to get the testing image's ids json file in COCO format:
```
python3 test_pre.py
```
# Training
To train the model, run this command:
```
python3 ./mmdetection tools/train config.py
```
You can download the pretrained weight from model zoo:
```
https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn
```
# Evaluation
Download the pretrained weight from the link:
```
https://drive.google.com/drive/folders/1tPnZFVozy8B1RHGlYduaWGgP8Ds9TUwo
```
You can run this command to generate answer.segm.txt:
```
python3 ./mmdetection/tools/test.py config.py nuclei.pth --format-only --options jsonfile_prefix=../answer.json
```
# Results
mAP 0.5, 0.95 => 0.243091 <br>
# Reference
Mask RCNN
```
https://arxiv.org/pdf/1703.06870.pdf
```
mmdetection
```
https://github.com/open-mmlab/mmdetection
```
