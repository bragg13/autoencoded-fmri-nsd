# Neuroscience meets ML
Building models to encode human brain.

Originally, final project for Artificial Intelligence for Games and Simulations.

## Setting up the environment
1. Install python version 3.11.10
```bash
# macosx
brew install pyenv

# windows powershell - not tested
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"


# then
pyenv install 3.11.10 # or 3.11.9 if .10 is not available
```

2. Create and activate virtual environment
```bash
pyenv exec python3 -m venv .venv
source .venv/bin/activate # macosx

.venv-aigs/Scripts/activate # windows powershell - not tested
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download datasets
- create dataset structure
```bash
mkdir dataset
mkdir dataset/coco/ dataset/nsd_data/
```

- download coco annotations
```bash
cd dataset/coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
rm annotations_trainval2017.zip annotations_trainval2017.zip
```

- download algonauts dataset
visit [Algonauts Challenge form](https://www.google.com/url?q=https%3A%2F%2Fdocs.google.com%2Fforms%2Fd%2Fe%2F1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA%2Fviewform%3Fusp%3Dsf_link) and fill in the form to get access to the Google Drive folder containing the unzipped dataset for each subject.

The resulting structure should be the following
```
dataset/
  nsd_coco.csv
  coco/
    annotations/
    panoptic_annotations/
  nsd_data/
    subj01/
    ...
    subj08/
```

## Run the training
...

## Run inference
...
