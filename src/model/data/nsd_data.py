import os
import numpy as np
import jax.numpy as jnp
import pandas as pd
from jax import random
from sklearn.model_selection import train_test_split
from data.roi import load_roi_data
import logging
from data.coco_load import CocoLoader
from nilearn import datasets, plotting
from nilearn.surface import load_surf_mesh
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSDDataLoader:
    def __init__(self, data_dir, subject=3, roi_class='floc-bodies', hem='lh'):
        self.subject = subject
        self.data_dir = f"{ data_dir }/nsd_data"
        self.roi_class = roi_class
        self.hem = hem

        # not all images listed in the df are actually in the folder, so we need to adjust the df to only include the images that are actually in the folder
        self.images_to_nsd = self.get_images_to_nsd_df()

        # this is the dataframe with the images and their information for each subject
        if not os.path.exists(os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "subject_specific.csv")):
            self.coco_loader = CocoLoader(data_dir, self.subject)
            self.subject_specific_df = pd.merge(self.images_to_nsd, self.coco_loader.subject_coco_df, left_on="nsdId", right_on="nsdId", how="inner")
            self.subject_shared_df = pd.merge(self.images_to_nsd, self.coco_loader.shared_coco_df, left_on="nsdId", right_on="nsdId", how="inner")

            # now merge with the categories
            self.subject_specific_df = pd.merge(self.subject_specific_df, self.coco_loader.categories_df, left_on="cocoId", right_on="cocoId", how="inner")
            self.subject_shared_df = pd.merge(self.subject_shared_df, self.coco_loader.categories_df, left_on="cocoId", right_on="cocoId", how="inner")

            # export to csv
            self.subject_specific_df.to_csv(os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "subject_specific.csv"), index=False)
            self.subject_shared_df.to_csv(os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "subject_shared.csv"), index=False)
            del self.coco_loader

        else:
            self.subject_specific_df = pd.read_csv(os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "subject_specific.csv"))
            self.subject_shared_df = pd.read_csv(os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "subject_shared.csv"))

        self.get_df_info()


    def get_df_info(self):
        # number of shared images for this subject
        num_specific = len(self.subject_specific_df)
        num_shared = len(self.subject_shared_df)

        # how many images in the specific have 'person' category
        # num_person_specific = len(self.subject_specific_df[self.subject_specific_df['categories'].str.contains('person')])
        # num_person_shared = len(self.subject_shared_df[self.subject_shared_df['categories'].str.contains('person')])

        logger.info(f"Subject {self.subject} has {num_shared} shared images and {num_specific} specific images")
        # logger.info(f"Subject {self.subject} has {num_person_shared} shared images and {num_person_specific} specific images with 'person' category")



    def get_images_to_nsd_df(self):
        """
            Load the image list indices for a given subject.

            Returns:
                pd.DataFrame: DataFrame containing image list indices and corresponding nsd indices
        """
        path_img2nsd = os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", 'images_to_nsd.csv')
        if os.path.exists(path_img2nsd):
            return pd.read_csv(path_img2nsd)
        else:
            logger.info('File **images_to_csd.csv* not foudn for subject. Building image list indices...')
            images_path = os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "training_images")
            images = sorted(os.listdir(images_path))
            images_to_nsd= {}
            for i, filename in enumerate(images):
                start_i = filename.find("nsd-") + len("nds-")
                nsd_index = int(filename[start_i : start_i + 5])
                images_to_nsd[i] = [i, nsd_index]
            logger.info(f'Loaded {len(images_to_nsd)} images for subject {self.subject}')

            df = pd.DataFrame.from_dict(images_to_nsd, orient="index", columns=["listIdx", "nsdId"])
            df.to_csv(path_img2nsd, index=False)
            return df

    def get_shared_indices(self, category='person'):
        """
            Get the image indices for shared images with a specified category.

            Args:
                category (str, optional): The coco category. Defaults to 'person'.

            Returns:
                tuple: Two numpy arrays containing indices for shared images with and without the specified category respectively
        """
        shared_df = self.subject_shared_df
        shared_category, shared_not_category = self.coco_loader.split_by_category(shared_df, category)
        logger.info(f'Found {len(shared_category)} shared images with category {category}')
        return shared_category["listIdx"].values, shared_not_category["listIdx"].values

    def get_train_eval_indices(self):
        """
            Returns:
                tuple: Two numpy arrays containing train and eval indices respectively:
                    - train_idxs (np.ndarray): Indices for training set (90% of data)
                    - eval_idxs (np.ndarray): Indices for eval set (10% of data)
        """
        subject_df = self.subject_specific_df
        logger.info(f'Loaded {len(subject_df)} images for subject {self.subject}')

        # on training, we don't want to use shared images
        subject_df = subject_df[subject_df['shared1000'] == False]
        logger.info(f'Excluded shared images from training set. {len(subject_df)} images remaining')
        train_idxs, eval_idxs = train_test_split(np.arange(len(subject_df)), test_size=0.1, random_state=42)
        logger.info(f'Split data into {len(train_idxs)} training and {len(eval_idxs)} validation indices')
        return train_idxs, eval_idxs

    def z_score(self, data):
        return jnp.array((data - data.mean()) / data.std())

    def get_split_masked_datasets(self, indices):
        """
            Get training and eval fMRI datasets for a specified subject and ROI class; mask the data by ROI.

            Returns:
                tuple: Two arrays containing train and eval fMRI data respectively.
                For hem='all', arrays contain concatenated data from both hemispheres.
                For hem='lh'/'rh', arrays contain data from specified hemisphere only.
        """
        # get the paths to the fmri data
        fmri_base_path = os.path.join(self.data_dir, f"subj0{self.subject}", "training_split", "training_fmri")
        lh_fmri_path = os.path.join(fmri_base_path, "lh_training_fmri.npy")
        rh_fmri_path = os.path.join(fmri_base_path, "rh_training_fmri.npy")

        # get the indices of the training and evaling sets, or shared images
        first_idxs, sec_idxs = indices
        first_lh_fmri = jnp.load(lh_fmri_path)[first_idxs]
        first_rh_fmri = jnp.load(rh_fmri_path)[first_idxs]
        sec_lh_fmri = jnp.load(lh_fmri_path)[sec_idxs]
        sec_rh_fmri = jnp.load(rh_fmri_path)[sec_idxs]

        # get the ROI mask
        roi_data = load_roi_data(self.data_dir, self.subject)
        roi_lh, roi_rh = roi_data['challenge']['lh'][self.roi_class] > 0, roi_data['challenge']['rh'][self.roi_class] > 0

        # mask the data by ROI
        first_lh_fmri = first_lh_fmri[:, roi_lh]
        first_rh_fmri = first_rh_fmri[:, roi_rh]
        sec_lh_fmri = sec_lh_fmri[:, roi_lh]
        sec_rh_fmri = sec_rh_fmri[:, roi_rh]

        if self.hem == 'all':
            train_all_fmri = np.concatenate([first_lh_fmri, first_rh_fmri], axis=1)
            eval_all_fmri = np.concatenate([sec_lh_fmri, sec_rh_fmri], axis=1)
            logger.info(f'Loaded all hemisphere data with shape {train_all_fmri.shape}')
            return self.z_score(train_all_fmri), self.z_score(eval_all_fmri)
        elif self.hem == 'lh':
            logger.info(f'Loaded left hemisphere data with shape {first_lh_fmri.shape}')
            return self.z_score(first_lh_fmri), self.z_score(sec_lh_fmri)
        elif self.hem == 'rh':
            logger.info(f'Loaded right hemisphere data with shape {first_rh_fmri.shape}')
            return self.z_score(first_rh_fmri), self.z_score(sec_rh_fmri)
        else:
            raise ValueError(f"Invalid hemisphere selection: {self.hem}. Must be 'all', 'lh', or 'rh'.")

    def get_train_eval_datasets(self):
        indices = self.get_train_eval_indices()
        return self.get_split_masked_datasets(indices)

    def get_analysis_datasets(self, category):
        indices = self.get_shared_indices(category)
        return self.get_split_masked_datasets(indices)

    def get_batches(self, fmri, key, batch_size: int):
        """Create batches of fMRI data with the specified batch size.

        Args:
            fmri: Array containing fMRI data to be batched
            batch_size (int): Size of each batch

        Yields:
            ndarray: Batch of fMRI data with shape (batch_size, voxels)
        """

        num_samples = fmri.shape[0]
        permutation = random.permutation(key, num_samples // batch_size * batch_size)
        return fmri[permutation]

    def fsa_fn(data, cfg, idx):
        class_name, roi_id

    def get_mesh_data(self, hem: Literal["all", "lh", "rh"]):
        ATLAS = datasets.fetch_surf_fsaverage("fsaverage")
        coords, faces = load_surf_mesh(ATLAS["flat_lh"])
        fsa = fsa_fn(data, cfg, idx)


    #  unmask data to original challenge space
    def unmask_from_roi_class(self, masked, roi_class, hem: Literal["all", "lh", "rh"], lh_chal_space_size=19004, rh_chal_space_size=20544):
        """
        Args:
            lh_chal_space_size (int, optional): length of the subject's lh challenge space. Defaults to 19004 (true for subjects 1,2,3,4,5,7).
            rh_chal_space_size (int, optional): length of the subject's rh challenge space. Defaults to 20544 (true for subjects 1,2,3,4,5,7).
        """
        roi_data = load_roi_data(self.data_dir, self.subject)
        if hem == 'all':
            roi_lh = roi_data['challenge']['lh'][roi_class] > 0
            roi_rh = roi_data['challenge']['rh'][roi_class] > 0
            roi_mask = np.concatenate([roi_lh, roi_rh], axis=0)
            unmasked = np.zeros((masked.shape[0], lh_chal_space_size+rh_chal_space_size))
        elif hem == 'lh':
            roi_mask = roi_data['challenge']['lh'][roi_class] > 0
            unmasked = np.zeros((masked.shape[0], lh_chal_space_size))
        elif hem == 'rh':
            roi_mask = roi_data['challenge']['rh'][roi_class] > 0
            unmasked = np.zeros((masked.shape[0], rh_chal_space_size))
        unmasked[:, roi_mask] = masked
        return unmasked
