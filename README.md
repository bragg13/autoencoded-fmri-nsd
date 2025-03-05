# Neuroscience meets ML
Building models to encode human brain.

![](https://github.com/bragg13/autoencoded-fmri-nsd/blob/main/img10_15_subj3.png)

*On the left, an image from COCO, specifically #10 and #15.*

*On the right, 2D visualisation of fMRI BOLD signal in left and right hemisphere while looking at the image, plotted on a flat FreeSurfer's fsaverage.*

Originally, final project for Artificial Intelligence for Games and Simulations.

## TL;DR
We trained models to encode the most relevant features of some fMRI data (training set) into a latent vector. The fMRI data is obtained while subjects look at images, which can contain a human or not.
Our model was not explicitly trained to separate the two categories; it just encodes highly complex fMRI data into a small vector. 
By performing inference on known-category data (test set), we observe encoded vectors to be distinguishable based on the category.

## Research Idea, Methods, and Results

The [Natural Scenes Dataset](https://naturalscenesdataset.org/) is a large-scale fMRI dataset conducted at ultra-high-field (7T) strength. The dataset consists of whole-brain, high-resolution (1.8-mm isotropic, 1.6-s sampling rate) fMRI measurements of 8 healthy adult subjects while they viewed thousands of color natural scenes. (adapt. from their website)

The images are part of the [COCO dataset](https://cocodataset.org/#home) and are labeled with category. In particular, we were interested in those which contained a person/human body, categorised as _person_, and those which did not.
Additionally, every subject viewed a total of ~10000 images; a small part was _shared_ by all the participants (~1000 images), while the others were _subject-specific_ , ie. only seen by that subject.
We selected a specific Region of Interest (ROI) in the Visual Cortex called [floc-bodies](http://vpnl.stanford.edu/fLoc/), which specialises in recognising human bodies.

For each subject, we trained a Sparse Autoencoder (SAE) on BOLD signals from the subject-specific images (our training set). This way, it learned to extract key features of the brain state of subjects while looking at images. 

The result was a set of 8 trained models, one for each subject, which we used for inference with the _shared_ images (test set). We performed inference with ~600 images, equally separated between _person_ and _non-person_ categories. 
We plotted the resulting vectors with t-SNE, observing different distributions based on the category, which the model was completely unaware of during training.

<img width="890" alt="image" src="https://github.com/user-attachments/assets/b89bf725-b0a1-4b2c-b02d-2fec443c267c" />
_Note: greyed-out squares are non-significant subjects_

Nevertheless, this does not allow us to state that it specifically encodes information about this categorization. Natural scene images present a large spectrum of different features, which in turn result in complex activation patterns in the brain. Those features most likely overlap for person and non-person stimuli, making it difficult for the model to distinguish categories.

Moreover, this is a very small-scale experiment, and we specifically selected _floc-bodies_ because it is known in the literature to be sensitive to stimuli containing bodies. Our results suggest that our sparse autoencoder, trained on activations within the _floc-bodies_ region, might be able to capture information about whether the participant was looking at an image with or without a human body in it.

More detailed information can be found on our [Research Report](https://github.com/bragg13/autoencoded-fmri-nsd/blob/dev/NeuroscienceMeetsML.pdf).


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
