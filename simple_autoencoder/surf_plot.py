# %% imports in use
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import nibabel as nib
from roi import ROI_TO_CLASS

# %% download the dataset, this is also in nsd_data
data_dir = '../data'
subject = 3

# the class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

args = argObj(data_dir, subject)

# load the dataset
fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

print(f'LH training fMRI data shape:\n{lh_fmri.shape} (Training stimulus images × LH vertices)')
print(f'\nRH training fMRI data shape:\n{rh_fmri.shape} (Training stimulus images × RH vertices)')

# %% load image lists and define some params
img = 0

# train image directory
train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

# train and test i
train_img_list = os.listdir(train_img_dir).sort()
test_img_list = os.listdir(test_img_dir).sort()

# %% img plot
def plotImg(img: int):
    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    plt.figure()
    plt.axis('off')
    plt.imshow(train_img)
    plt.title('Training image: ' + str(img+1))
    plt.show()

# %% print data for paper figure
def printData(roi_map, challenge_roi_class, fsaverage_roi_class, start=250, end=300):
    print(f'\nfloc-bodies: {roi_map}')
    print(f'fsaverage space: {len(fsaverage_roi_class)}')
    print(f'\nlh.floc-bodies_fsaverage_space[{start}:{end}]')
    print(fsaverage_roi_class[start:end])
    print(f'challenge space: {len(challenge_roi_class)}')
    print(f'\nlh.floc-bodies_challenge_space[{start}:{end}]')
    print(challenge_roi_class[start:end])
    print(f'\nlh_fmri[0][{start}:{end}]')
    print(f'{lh_fmri[0][start:end]}')

# %% show roi on brain surface map

# this should now be sourced from roi.py
def get_roi_data(roi_class, hemi):
    challenge_roi_class_dir = os.path.join(args.data_dir, 'roi_masks', hemi[0]+'h.'+roi_class+'_challenge_space.npy')
    fsaverage_roi_class_dir = os.path.join(args.data_dir, 'roi_masks', hemi[0]+'h.'+roi_class+'_fsaverage_space.npy')
    roi_map_dir = os.path.join(args.data_dir, 'roi_masks', 'mapping_'+roi_class+'.npy')
    
    # Load the ROI brain surface maps
    challenge_roi_class = np.load(challenge_roi_class_dir) # roi indices mapped to challange space
    fsaverage_roi_class = np.load(fsaverage_roi_class_dir) # roi indices mapped to fsaverage
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()
    return challenge_roi_class, fsaverage_roi_class, roi_map

# map challenge space to fsaverage for roi of interest
def map_fsaverage_resp(img, roi, hemisphere:str, full_class=False):
    challenge_roi_class, fsaverage_roi_class, roi_map = get_roi_data(ROI_TO_CLASS[roi], hemisphere)
    
    if full_class: 
        fsvg_roi, ch_roi = fsaverage_roi_class, challenge_roi_class
    else:
        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)] # the id of the roi
        challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int) # set all other roi ids to 0 (False)
        fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)
        fsvg_roi, ch_roi = fsaverage_roi, challenge_roi
    
    fsaverage_response = np.zeros(len(fsvg_roi))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsvg_roi)[0]] = lh_fmri[img,np.where(ch_roi)[0]]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsvg_roi)[0]] = rh_fmri[img,np.where(ch_roi)[0]]
    return fsaverage_response

def plotViewSurf(fsaverage_map, hemi, title: str, cmap, vmax=None, vmin=None, sym_cmap=True):
    """title should be ROI or ROI class"""
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage') # this should be global
    return plotting.view_surf(
        surf_mesh=fsaverage['infl' + '_' + hemi], # flat_ , pial_ , sphere_
        surf_map=fsaverage_map,
        bg_map=fsaverage['sulc_' + hemi],
        threshold=1e-14,
        cmap=cmap,
        colorbar=True,
        title=title + ', ' + hemi + ' hemisphere',
        vmax=vmax,
        vmin=vmin,
        symmetric_cmap=sym_cmap
        )

# %% plot functions
def plotRoiClass(roi_class,hemi,cmap):
    _, fsaverage_roi_class, _ = get_roi_data(roi_class, hemi)
    return plotViewSurf(fsaverage_roi_class, hemi, roi_class, cmap, vmax=3.,vmin=1.,sym_cmap=False)

def plotRoiClassValues(img, roi,hemi,cmap):
    fsaverage_response = map_fsaverage_resp(img, roi, hemi, full_class=True)
    return plotViewSurf(fsaverage_response, hemi, ROI_TO_CLASS[roi], cmap)

def plotRoiValues(img, roi,hemi,cmap):
    fsaverage_response = map_fsaverage_resp(img, roi, hemi)
    return plotViewSurf(fsaverage_response, hemi, roi, cmap)

# %% view in browser
roiclass = plotRoiClass('floc-bodies', hemi='left', cmap='brg')
roiclass.open_in_browser()
roiclassvalues = plotRoiClassValues(img, 'EBA', hemi='left', cmap='cold_hot')
roiclassvalues.open_in_browser()
roivalues = plotRoiValues(img, 'EBA', 'left', cmap='blue_transparent_full_alpha_range')
roivalues.open_in_browser()