import csv
import re

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class SKU:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.imgs = {}
        self.imgToAnns = {}
        self.ann_keys = ['bbox', 'image_id', 'id','category_id']
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
    def loadImgs(self, img_ids):
        #self.loadImgs = function to load img with dictionary
        #return sorted list of dict
        return [self.imgs[i] for i in img_ids]

    def loadRes(self, results):
        pass

    def getImgIds(self, ImgIds=[], catIds=[]):
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

class SKUeval:
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def evaluate(self):
        pass
    def accumulate(self):
        pass
    def summarize(self):
        pass