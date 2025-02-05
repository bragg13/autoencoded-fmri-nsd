import os
import json
import pandas as pd
import numpy as np
import skimage.io
from skimage.transform import resize
from collections import defaultdict
import ast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CocoLoader:
    def __init__(self, data_dir):
        self.data_dir = f"{ data_dir }/coco"
        self.nsd_coco: pd.DataFrame = self._read_and_preprocess()
        self.img_id_to_anns = defaultdict(list)
        self.cat_id_to_cat = defaultdict(list)
        self._load_annotations()
        self.subject_cols = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8']

    """
        Load the nsd-coco csv file and preprocess it.
    """
    def _read_and_preprocess(self) -> pd.DataFrame:
        useless_cols = ['Unnamed: 0', 'loss', 'flagged', 'BOLD5000',
                        'subject1_rep0', 'subject1_rep1', 'subject1_rep2',
                        'subject2_rep0', 'subject2_rep1', 'subject2_rep2',
                        'subject3_rep0', 'subject3_rep1', 'subject3_rep2',
                        'subject4_rep0', 'subject4_rep1', 'subject4_rep2',
                        'subject5_rep0', 'subject5_rep1', 'subject5_rep2',
                        'subject6_rep0', 'subject6_rep1', 'subject6_rep2',
                        'subject7_rep0', 'subject7_rep1', 'subject7_rep2',
                        'subject8_rep0', 'subject8_rep1', 'subject8_rep2']

        # association between nsd and coco ids
        nsd_coco = pd.read_csv(f'{self.data_dir}/../nsd_coco.csv')
        nsd_coco.drop(columns=useless_cols, inplace=True)
        logger.info(f'nsd-coco loaded: {len(nsd_coco)} images')
        return nsd_coco

    """
        Load the panoptic annotations for the coco dataset.
    """
    def _load_annotations(self):
        ann_files = [f'{self.data_dir}/panoptic_annotations/panoptic_train2017.json',
                     f'{self.data_dir}/panoptic_annotations/panoptic_val2017.json']
        for ann_file in ann_files:
            dataset = json.load(open(ann_file, 'r'))
            if 'annotations' in dataset:
                for ann in dataset['annotations']:
                    self.img_id_to_anns[ann['image_id']].append(ann)
                for cat in dataset['categories']:
                    self.cat_id_to_cat[cat['id']].append(cat)
        logger.info('Annotations loaded')
        logger.info(f'{type(self.img_id_to_anns)}')
        # logger.info(f'{self.cat_id_to_cat}')


    """
        Get the shared images between all subjects.

        Returns:
            shared: pandas dataframe with shared images
    """
    def get_shared_images_df(self):
            shared = self.nsd_coco[self.nsd_coco['shared1000'] == True]
            shared = shared.drop(columns=['shared1000'] + self.subject_cols)
            return shared

    """
        Get the images for a specific subject.

        Args:
            subject_id: subject number

        Returns:
            img_df: pandas dataframe with images for the subject
    """
    def get_subject_images_df(self, subject_id):
        img_df = self.nsd_coco[(self.nsd_coco[f'subject{subject_id}'] == True) & (self.nsd_coco['shared1000'] == False)]
        img_df = img_df.drop(columns=self.subject_cols)
        return img_df

    """
        Get the categories for the shared images.

        Returns:
            categories: pandas dataframe with categories
    """
    def get_categories_df(self, df):
        return pd.DataFrame.from_dict(self._extract_categories(df), orient='index', columns=['cocoId', 'categories'])

    """
        Filter the dataframe by category.

        Args:
            df: dataframe to filter
            category: category to filter by
            contain: whether to include or exclude the category

        Returns:
            df: filtered dataframe
    """
    def filter_by_category(self, df, category, contain=True):
            if contain:
                df = df[df['categories'].apply(lambda x: category in x)]
            else:
                df = df[df['categories'].apply(lambda x: category not in x)]
            return df

    """
        Split the dataframe by category.

        Args:
            df: dataframe to split
            category: category to split by

        Returns:
            df1: dataframe with category
            df2: dataframe without category
    """
    def split_by_category(self, df, category):
        df1 = df[df['categories'].apply(lambda x: category in x)]
        df2 = df[df['categories'].apply(lambda x: category not in x)]
        return df1, df2

    """"
        Get the category names for a coco id.

        Args:
            coco_id: coco id

        Returns:
            category: category names
    """
    def get_category_from_coco_id(self, coco_id):
        return self.nsd_coco[self.nsd_coco['cocoId'] == coco_id]['categories']

    """
        Get the category names for a nsd id.

        Args:
            nsd_id: nsd id

        Returns:
            category: category names
    """
    def get_category_from_nsd_id(self, nsd_id):
        return self.nsd_coco[self.nsd_coco['nsdId'] == nsd_id]['categories']

    """
        Get the shared dataframe with categories.

        Returns:
            shared: shared dataframe with categories
    """
    def get_shared_df(self):
        shared = self.get_shared_images_df()
        categories = self.get_categories_df(shared)
        return shared.merge(categories, on='cocoId')

    """
        Get the subject dataframe with categories.

        Args:
            subject_id: subject number

        Returns:
            subj: subject dataframe with categories
    """
    def get_subject_df(self, subject_id):
        subj = self.get_subject_images_df(subject_id)
        categories = self.get_categories_df(subj)
        return subj.merge(categories, on='cocoId')

    """
        Get the shared dataframe with categories filtered by category.

        Args:
            category: category to filter by
            contain: whether to include or exclude the category

        Returns:
            shared: shared dataframe with categories filtered by category
    """
    def _extract_categories(self, shared_df):
        min_size = 227
        categories = dict()
        coco_id_arr = np.copy(shared_df['cocoId'].values)
        nsd_crop_arr = shared_df['cropBox'].values
        nsd_crop_arr = [ast.literal_eval(item) for item in nsd_crop_arr]

        for i in range(len(coco_id_arr)):
            crop = nsd_crop_arr[i]
            coco_id = coco_id_arr[i]
            png_name = os.path.join(self.data_dir, 'panoptic_joint', f'{coco_id:012d}.png')

            img = skimage.io.imread(png_name)
            cropped_img = resize(self._apply_crop_to_img(img, crop), (min_size, min_size), order=0)

            img_seg_ids = self._mask_to_unique_indices(cropped_img.astype('uint32'))
            cat_ids = self._get_category_ids(self.img_id_to_anns[coco_id], img_seg_ids)
            cat_names = self._get_category_names(cat_ids)
            categories[coco_id] = [coco_id, cat_names]

        return categories

    """
        Apply the crop to the image.

        Args:
            img: image
            box: crop box

        Returns:
            img: cropped image
    """
    def _apply_crop_to_img(self, img, box):
        shape = img.shape
        top_crop = np.round(shape[0] * box[0]).astype(int)
        bottom_crop = np.round(shape[0] * box[1]).astype(int)
        left_crop = np.round(shape[1] * box[2]).astype(int)
        right_crop = np.round(shape[1] * box[3]).astype(int)
        return img[top_crop:(shape[0] - bottom_crop), left_crop:(shape[1] - right_crop)]

    """
        Convert the mask to indices.

        Args:
            img: image

        Returns:
            indices: indices
    """
    def _mask_to_indices(self, img):
        return img[:, :, 0] + img[:, :, 1] * 256 + img[:, :, 2] * (256 ** 2)

    """
        Convert the mask to unique indices.

        Args:
            img: image

        Returns:
            img_seg_ids: unique indices
    """
    def _mask_to_unique_indices(self, img):
        img_seg_ids = np.unique(self._mask_to_indices(img))
        return img_seg_ids[img_seg_ids != 0]

    """
        Get the category ids.

        Args:
            annotations: annotations
            img_seg_ids: image segment ids

        Returns:
            category ids
    """
    def _get_category_ids(self, annotations, img_seg_ids):
        seg_to_cat_id = defaultdict(list)
        for ann in annotations:
            for seg in ann['segments_info']:
                seg_to_cat_id[seg['id']] = seg['category_id']
        return [seg_to_cat_id[s] for s in img_seg_ids if s in seg_to_cat_id]

    """
        Get the category names.

        Args:
            cat_ids: category ids

        Returns:
            category names
    """
    def _get_category_names(self, cat_ids):
        return np.unique([self.cat_id_to_cat[c][0]['name'] for c in cat_ids])

    """
        Get the supercategory names.

        Args:
            cat_ids: category ids

        Returns:
            supercategory names
    """
    def _get_supercategory_names(self, cat_ids):
        return np.unique([self.cat_id_to_cat[c][0]['supercategory'] for c in cat_ids])
