import csv
import re

class SKU:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.imgs = {}
        self.imgToAnns = {}
        self.ann_keys = ['bbox', 'image_id', 'id','category_id']
        csv_file = open(csv_file)
        csv_reader = csv.reader(csv_file)
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