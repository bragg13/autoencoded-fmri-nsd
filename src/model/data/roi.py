#%%
import os
import numpy as np

# %% Dict mapping ROI class names to lists of ROI names
CLASS_TO_ROI = {"prf-visualrois":  ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
                    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
                    "floc-faces":  ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
                    "floc-places": ["OPA", "PPA", "RSC"],
                    "floc-words":  ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
                    "streams":     ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]}

# Dict mapping ROI names to their class name
ROI_TO_CLASS = {roi: roi_class for roi_class, rois in CLASS_TO_ROI.items() for roi in rois}

# List of all ROI names
ROIS = [roi for roi_class in CLASS_TO_ROI.values() for roi in roi_class]


def get_fmri_data_roi(subj: int = 3, roi_class: str = 'floc-bodies', hemisphere: str = 'lh', base_dir: str = './dataset/nsd_data'):
    """
    Load ROI masks for a specific subject, ROI, and hemisphere.

    Returns:
        Tuple of (challenge_roi, fsaverage_roi) arrays.
    """
    subj_str = f"subj0{subj}"
    data_dir = os.path.join(base_dir, subj_str)
    # For file naming, append a period (e.g., 'lh.' for left)
    hemi_prefix = hemisphere + '.'

    challenge_roi_path = os.path.join(data_dir, 'roi_masks', hemi_prefix + roi_class + '_challenge_space.npy')
    fsaverage_roi_path = os.path.join(data_dir, 'roi_masks', hemisphere[0] + 'h.' + roi_class + '_fsaverage_space.npy')
    # roi_map_path = os.path.join(data_dir, 'roi_masks', 'mapping_' + roi_class + '.npy')

    challenge_roi_class = np.load(challenge_roi_path)
    fsaverage_roi_class = np.load(fsaverage_roi_path)
    # print how many vertices are 0
    print(f"Number of 0 vertices in challenge space: {np.sum(challenge_roi_class == 0)}")
    print(f"Number of 0 vertices in fsaverage space: {np.sum(fsaverage_roi_class == 0)}")

    # roi_map = np.load(roi_map_path, allow_pickle=True).item()

    # Get the mapping for the ROI name
    # roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    # print(f"ROI mapping for : {roi_mapping}")
    # challenge_roi = (challenge_roi_class == roi_mapping).astype(int)
    # fsaverage_roi = (fsaverage_roi_class == roi_mapping).astype(int)

    return challenge_roi_class, fsaverage_roi_class

def load_roi_data(dataDir, subject=3):
    """
    Loads ROI data structures for specified subject
        Args:
            subject (int): Subject number (default: 3)
        Returns:
            dict: Mapping, challenge and fsaverage space ROI data
    """
    roi_dir = os.path.join(dataDir, "subj0"+str(subject), "roi_masks")

    data = {'mapping' : {},
            'challenge' : {'lh' : {}, 'rh' : {}},
            'fsaverage' : {'lh' : {}, 'rh' : {}}}

    for roi_class in CLASS_TO_ROI.keys():
        data['mapping'][roi_class] = {'id_to_roi' : {}, 'roi_to_id' : {}}
        data['mapping'][roi_class]['id_to_roi'] = np.load(os.path.join(roi_dir, f'mapping_{roi_class}.npy'), allow_pickle=True).item()
        # do we need the below?
        data['mapping'][roi_class]['roi_to_id'] = {v: k for k, v in data['mapping'][roi_class]['id_to_roi'].items()}

    for hem in ['lh', 'rh']:
        data['fsaverage'][hem]['all-vertices'] = np.load(os.path.join(roi_dir, f'{hem}.all-vertices_fsaverage_space.npy'))
        for roi_class in CLASS_TO_ROI.keys():
            data['challenge'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_challenge_space.npy'))
            data['fsaverage'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_fsaverage_space.npy'))
    return data
