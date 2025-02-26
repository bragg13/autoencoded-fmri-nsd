import os
import json
import pandas as pd
import numpy as np
import skimage.io
from skimage.transform import resize
from collections import defaultdict
import ast
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CocoLoader:
    def __init__(self, data_dir, subject):
        self.data_dir = f"{ data_dir }/coco"
        self.subject = subject
        self.nsd_coco: pd.DataFrame = self._read_and_preprocess()
        self.img_id_to_anns = defaultdict(list)
        self.cat_id_to_cat = defaultdict(list)
        self._load_annotations()
        self.subject_cols = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8']
        self.subject_coco_df = self.nsd_coco[( self.nsd_coco[f'subject{self.subject}'] == 1) & (self.nsd_coco['shared1000'] == False)]
        self.shared_coco_df = self.nsd_coco[( self.nsd_coco[f'subject{self.subject}'] == 1) & (self.nsd_coco['shared1000'] == True)]
        self.categories_df = self.get_categories_df()



    """
        Load the nsd-coco csv file and preprocess it.
    """
    def _read_and_preprocess(self) -> pd.DataFrame:
        if os.path.exists(f'{self.data_dir}/../nsd_coco_cleaned.csv'):
            nsd_coco = pd.read_csv(f'{self.data_dir}/../nsd_coco_cleaned.csv')
            logger.info(f'nsd-coco loaded: {len(nsd_coco)} images')
            return nsd_coco

        logger.info('nsd-coco not found, loading nsd-coco')
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
        nsd_coco = pd.read_csv(f'{self.data_dir}/../nsd_coco_full.csv')
        nsd_coco.drop(columns=useless_cols, inplace=True)
        logger.info(f'nsd-coco loaded: {len(nsd_coco)} images')
        nsd_coco.to_csv(f'{self.data_dir}/../nsd_coco_cleaned.csv')
        return nsd_coco

    """
        Load the panoptic annotations for the coco dataset.
    """
    def _load_annotations(self):
        if os.path.exists(f'{self.data_dir}/img_id_to_anns.json') and os.path.exists(f'{self.data_dir}/cat_id_to_cat.json'):
            with open(f'{self.data_dir}/img_id_to_anns.json', 'rb') as f:
                self.img_id_to_anns = pickle.load(f)
                logger.info('dict *img_id_to_anns* loaded from file')

            with open(f'{self.data_dir}/cat_id_to_cat.json', 'rb') as f:
                self.cat_id_to_cat = pickle.load(f)
                logger.info('dict *cat_id_to_cat* loaded from file')

            return

        logger.info('*img2anns* and *cat2cat* not found, loading annotations')
        ann_files = [f'{self.data_dir}/panoptic_annotations/panoptic_train2017.json',
                     f'{self.data_dir}/panoptic_annotations/panoptic_val2017.json']
        for ann_file in ann_files:
            dataset = json.load(open(ann_file, 'r'))
            if 'annotations' in dataset:
                for ann in dataset['annotations']:
                    self.img_id_to_anns[ann['image_id']].append(ann)
                for cat in dataset['categories']:
                    self.cat_id_to_cat[cat['id']].append(cat)
        with open(f'{self.data_dir}/img_id_to_anns.json', 'wb') as f:
            pickle.dump(self.img_id_to_anns, f)
            logger.info('dict *img_id_to_anns* saved to file')
        with open(f'{self.data_dir}/cat_id_to_cat.json', 'wb') as f:
            pickle.dump(self.cat_id_to_cat, f)
            logger.info('dict *cat_id_to_cat* saved to file')





    """
        Get the categories for the shared images.

        Returns:
            categories: pandas dataframe with categories
    """
    def get_categories_df(self):
        return pd.DataFrame.from_dict(self.extract_categories(self.subject_coco_df), orient='index', columns=['cocoId', 'categories'])

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


    def get_category_from_nsd_id(self, nsd_id):
        return self.nsd_coco[self.nsd_coco['nsdId'] == nsd_id]['categories']


    # def get_shared_df(self):
    #     shared = self.get_shared_images_df()
    #     categories = self.get_categories_df(shared)
    #     return shared.merge(categories, on='cocoId')

    # def get_subject_df(self, subject_id, with_shared=False):
    #     subj = self.get_subject_images_df(subject_id)
    #     categories = self.get_categories_df(subj)
    #     print(categories)

    #     return subj.merge(categories, on='cocoId')

    def extract_categories(self, shared_df):
        minSize = 227 # from og code
        categories = dict()

        cocoId_arr = np.copy(shared_df['cocoId'].values)
        nsdcrop_arr = shared_df['cropBox'].values # values are the cropBoxes but as strings
        nsdcrop_arr = [ast.literal_eval(item) for item in nsdcrop_arr] # ast converts strings to tuples of floats
        imgDir = os.path.join(self.data_dir, 'panoptic_annotations', 'panoptic_joint/')

        for i in range(len(cocoId_arr)):
            crop = nsdcrop_arr[i]
            cocoId = cocoId_arr[i]
            try:
                png_name = imgDir + '%012d.png' % cocoId

                img = skimage.io.imread(png_name)
                croppedImg = resize(self.applyCropToImg(img, crop), (minSize,minSize), order=0)

                imgSegIds = self.maskToUniqueIndices(croppedImg.astype('uint32'))
                catIds = self.getCategoryIDs(self.img_id_to_anns[cocoId], imgSegIds)
                catNames = self.getCategoryNames(self.cat_id_to_cat, catIds)
                categories[cocoId] = [cocoId, catNames]
            except:
                print(f'Error with image {cocoId}')
                continue

        return categories



    def applyCropToImg(self, img, box):
        '''
        applyCropToImg(img, cropBox)
        img ~ any h x w x n image
        cropBox ~ (top, bottom, left, right) in fractions of image size
        '''
        if box[0]+box[1] >= 1:
            raise ValueError('top and bottom crop must sum to less than 1')
        if box[2]+box[3] >= 1:
            raise ValueError('left and right crop must sum to less than 1')
        shape = img.shape
        topCrop = np.round(shape[0]*box[0]).astype(int)
        bottomCrop = np.round(shape[0]*box[1]).astype(int)
        leftCrop = np.round(shape[1]*box[2]).astype(int)
        rightCrop = np.round(shape[1]*box[3]).astype(int)
        croppedImage = img[topCrop:(shape[0]-bottomCrop),leftCrop:(shape[1]-rightCrop)]
        return croppedImage

    # base-256 representation, commonly used for image segmentation tasks
    # The resulting 2D array has a shape of h × w, where each element represents a unique integer ID corresponding to the color at that pixel
    def maskToIndices(self, img):
        return img[:,:,0]+img[:,:,1]*256+img[:,:,2]*(256**2)

    def maskToUniqueIndices(self, img):
        imgSegIds = np.unique(self.maskToIndices(img)) # returns a sorted array of unique segment IDs found in the image
        imgSegIds = imgSegIds[imgSegIds != 0]
        return np.unique(imgSegIds)

    def getCategoryIDs(self, annotations, imgSegIds): # get imgSegIds with maskToUniqueIndices()
        segToCatId = defaultdict(list)
        for ann in annotations:
            for seg in ann['segments_info']:
                segToCatId[seg['id']] = seg['category_id']
        return [segToCatId[s] for s in imgSegIds if s in segToCatId]

    def getCategoryNames(self, catIdToCat, catIds):
        return np.unique([catIdToCat[c][0]['name'] for c in catIds])

    def getSupercategoryNames(self, catIdToCat, catIds):
        return np.unique([catIdToCat[c][0]['supercategory'] for c in catIds])
