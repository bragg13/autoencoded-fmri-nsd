# import libraries
from logger import log
import os
import numpy as np
from pathlib import Path
import jax.numpy as jnp
from tqdm.gui import tqdm
import pandas as pd
import coco_load as cl
import nsd_data
import matplotlib.pyplot as plt

# from visualisations import plot_data_distribution
from sklearn.model_selection import train_test_split
from roi import load_roi_data
# jnp.fft.fft2(matirx)

def images_to_nsd_df(subject=3):
    # training and test images list, sorted
    images_path = os.path.join("../data", "subj0"+str(subject), "training_split", "training_images")
    images = sorted(os.listdir(images_path))

    # make a dataframe with mapping image-nsd_index
    images_to_nsd= {}
    for i, filename in enumerate(images):
        start_i = filename.find("nsd-") + len("nds-")
        nsd_index = int(filename[start_i : start_i + 5])
        images_to_nsd[i] = [nsd_index] # are you sure this works andrea?
    images_to_nsd = pd.DataFrame.from_dict(
        images_to_nsd, orient="index", columns=["nsdId"] # are you sure this works andrea?
    )
    log(f"total images for subject {subject}: {len(images_to_nsd)}", 'DATA')
    return images_to_nsd

# andrea ill leave this here you can replace as you want, images_to_nsd is used in get_shared_inidces and get_train_test_indices
images_to_nsd = images_to_nsd_df(subject=3)

def get_shared_indices(category: str):
    coco_loaded = cl.nsd_coco
    shared_df = cl.getSharedDf(coco_loaded).merge(images_to_nsd, on='nsdId')
    shared_pers, shared_not_pers = cl.splitByCategory(shared_df, category)
    # category and not indices
    category_idxs = shared_pers["listIdx"].values
    not_category_idxs = shared_not_pers["listIdx"].values
    return category_idxs, not_category_idxs

def get_train_test_indices(subject=3):
    """
    Get the image indices for training and testing sets for a given subject.

    Args:
        subject (int, optional): Subject ID number (1-8). Defaults to 3.

    Returns:
        tuple: Two numpy arrays containing train and test indices respectively:
            - train_idxs (np.ndarray): Indices for training set (90% of data)
            - test_idxs (np.ndarray): Indices for test set (10% of data)
    """ 

    # map coco categories to the pics in the dataset
    coco_loaded = cl.nsd_coco
    subject_coco_df = cl.getSubjDf(coco_loaded, subject)
    subject_images = pd.merge(images_to_nsd, subject_coco_df, left_on="nsdId", right_on="nsdId", how="inner")

    train_idxs, test_idxs = train_test_split(np.arange(len(subject_images)), test_size=0.1, random_state=42)
    return train_idxs, test_idxs

def get_train_test_datasets(subject=3, roi_class='floc-bodies', hem='all') -> tuple:
    """Get training and test fMRI datasets for a specified subject and ROI class.

    Args:
        subject (int, optional): The subject ID number (1-8). Defaults to 3.
        roi_class (str, optional): Region of interest class name. Defaults to 'floc-bodies'.
        hem (str, optional): Hemisphere selection ('all', 'lh', or 'rh'). Defaults to 'all'.

    Returns:
        tuple: Two arrays containing train and test fMRI data respectively:
            - train_fmri: Training fMRI data array
            - test_fmri: Test fMRI data array
            For hem='all', arrays contain concatenated data from both hemispheres.
            For hem='lh'/'rh', arrays contain data from specified hemisphere only.
    """
    log('creating datasets...', 'DATA')
    # get the paths to the fmri data
    fmri_base_path = os.path.join("../data", "subj0"+str(subject), "training_split", "training_fmri")
    lh_fmri_path = os.path.join(fmri_base_path, "lh_training_fmri.npy")
    rh_fmri_path = os.path.join(fmri_base_path, "rh_training_fmri.npy")

    # get the indices for training and testing
    train_idxs, test_idxs = get_train_test_indices(subject)

    # get the ROI mask
    roi_data = load_roi_data(subject=3)

    # load the fmri data, sliced by indexes
    # ndr: for one image, there is both the left and right hemisphere (I mean not necessarily, but yea)
    train_lh_fmri = jnp.load(lh_fmri_path)[train_idxs]
    train_rh_fmri = jnp.load(rh_fmri_path)[train_idxs]

    test_lh_fmri = jnp.load(lh_fmri_path)[test_idxs]
    test_rh_fmri = jnp.load(rh_fmri_path)[test_idxs]

    # maske the data by ROI
    roi_lh, roi_rh = roi_data['challenge']['lh'][roi_class] > 0, roi_data['challenge']['rh'][roi_class] > 0

    train_lh_fmri = train_lh_fmri[:, roi_lh]
    train_rh_fmri = train_rh_fmri[:, roi_rh]
    test_lh_fmri = test_lh_fmri[:, roi_lh]
    test_rh_fmri = test_rh_fmri[:, roi_rh]

    # print(f"train_lh_fmri min: {train_lh_fmri.min()}, max: {train_lh_fmri.max()}")
    # print(f"train_rh_fmri min: {train_rh_fmri.min()}, max: {train_rh_fmri.max()}")
    # print(f"train_lh_fmri shape: {train_lh_fmri.shape}")
    # print(f"train_rh_fmri shape: {train_rh_fmri.shape}")

    # print(f"test_lh_fmri min: {test_lh_fmri.min()}, max: {test_lh_fmri.max()}")
    # print(f"test_rh_fmri min: {test_rh_fmri.min()}, max: {test_rh_fmri.max()}")
    # print(f"test_lh_fmri shape: {test_lh_fmri.shape}")
    # print(f"test_rh_fmri shape: {test_rh_fmri.shape}")

    if hem == 'all':
        train_all_fmri = np.concatenate([train_lh_fmri, train_rh_fmri], axis=1)
        test_all_fmri = np.concatenate([test_lh_fmri, test_rh_fmri], axis=1)
        return train_all_fmri, test_all_fmri
    elif hem == 'lh':
        return train_lh_fmri, test_lh_fmri
    elif hem == 'rh':
        return train_rh_fmri, test_rh_fmri
    else:
        raise ValueError(f"Invalid hemisphere selection: {hem}. Must be 'all', 'lh', or 'rh'.")

def get_batches(fmri, batch_size: int):
    """Create batches of fMRI data with the specified batch size.

    Args:
        fmri: Array containing fMRI data to be batched
        batch_size (int): Size of each batch

    Yields:
        ndarray: Batch of fMRI data with shape (batch_size, voxels)
    """

    num_samples = fmri.shape[0]
    while True:
        permutation = np.random.permutation(num_samples // batch_size * batch_size)
        for i in range(0, len(permutation), batch_size):
            batch_perm = permutation[i:i + batch_size]
            batch = fmri[batch_perm]
            # batch_volume = fmri[batch_perm] ... TODO
            yield batch
