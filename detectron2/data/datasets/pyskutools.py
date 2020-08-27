import csv
import re
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import copy
import datetime
import itertools
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve
from detectron2.structures import BoxMode


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class SKU:
    def __init__(self, csv_file=None):

        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.ann_keys = ['bbox', 'image_id', 'id','category_id']
        self.imgToAnns, self.catToImgs=defaultdict(list), defaultdict(list)
        self.imgs, self.cats, self.anns = {}, {}, {}
        self.csv_file = csv_file
        self.box_mode = BoxMode.XYXY_ABS
        if not csv_file == None:
            print('loading annotations to memory ... ')
            tic = time.time()
            csv_reader = csv.reader(open(csv_file,'r'))
            self.cats[0] = {'name':'object', 'id':0}
            ann_id = 0
            for row in csv_reader:
                file_name = row[0]
                img_id = re.findall(r'\d+', file_name)[0]
                bbox = [float(i) for i in row[1:5]]
                category = row[5]
                width = float(row[6])
                height = float(row[7])

                if img_id not in self.imgs:
                    self.imgs[img_id] = {
                        'file_name' : file_name,
                        'height' : height,
                        'width' : width,
                        'image_id' : img_id,
                        'id' : img_id
                    }
    
                self.imgToAnns[img_id].append({
                    'bbox' : bbox,
                    'image_id' : img_id,
                    'id' : ann_id,
                    'category_id' : 0,
                    'area' : (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'iscrowd' : 0
                })

                self.anns[ann_id] = {
                    'bbox' : bbox,
                    'image_id' : img_id,
                    'id' : ann_id,
                    'category_id' : 0,
                    'area' : (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'iscrowd' : 0
                }

                ann_id += 1

                self.catToImgs[0].append(img_id)
            self.dataset['annotations'] = list(self.anns.values())
            self.dataset['images'] = list(self.imgs.values())
            self.dataset['categories'] = list(self.cats.values())
            print('Done (t={:0.2f}s)'.format(time.time()- tic))


        """
        self.csv_file = csv_file
        self.imgs = {}
        self.imgToAnns = {}
        csv_file = open(csv_file)
        csv_reader = csv.reader(csv_file)
        self.dataset = csv_reader
        for row in csv_reader:
            file_name = row[0]
            _id = re.findall(r'\d+', file_name)[0]
            bbox = [float(i) for i in row[1:5]]
            category = row[5]
            width = float(row[6])
            height = float(row[7])

            if _id not in self.imgs:
                self.imgs[_id] = {
                    'file_name' : file_name,
                    'height' : height,
                    'width' : width,
                    'image_id' : _id,
                    'id' : _id
                }
                self.imgToAnns[_id] = []

            self.imgToAnns[_id].append({
                'bbox' : bbox,
                'image_id' : _id,
                'id' : _id,
                'category_id' : 0
            })
        """
        
        #self.imgs = dictionary for imgs
        # key : image_id
        # val : {'file_name': 'test_0.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'image_id' : 0 (same with file name)
        # }

        #self.imgToAnns = dictionary to convert img_id to annotations
        #return sorted list of list

        #self.ann_keys = ["bbox", "category_id"]

    def eval(self):
        """
        change bbox mode to XYHW_ABS for evaluation
        """
        if self.box_mode == BoxMode.XYXY_ABS:
            for k in self.anns.keys():
                self.anns[k]['bbox'] = BoxMode.convert(self.anns[k]['bbox'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                bbox = self.anns[k]['bbox']
                #self.anns['area'] = bbox[2] * bbox[3]
            self.box_mode = BoxMode.XYWH_ABS
            self.createIndex()
            
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        
        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

        ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[ann_id] for ann_id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[cat_id] for cat_id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = SKU()
        res.dataset['images'] = [img for img in self.dataset['images']]
        res.box_mode = BoxMode.XYWH_ABS

        print('Loading and preparing results...')
        tic = time.time()
        """
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
        """
        anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'

        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for ann_id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = ann_id+1
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res