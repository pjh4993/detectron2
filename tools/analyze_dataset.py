import detectron2.data.datasets.builtin as builtin
from collections import defaultdict
from pycocotools.coco import COCO
import numpy as np
import json
import matplotlib.pyplot as plt
import torch


COCO_CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

SKU_CATEGORIES = [
    "object"
]

CATEGORIES = COCO_CATEGORIES
#CATEGORIES = SKU_CATEGORIES

def main():
    with open('data_dict_train.json') as fp:
        data_dict = json.load(fp)
        
    with open('result_dict_minival.json') as fp:
        result_dict = json.load(fp)

    result = np.zeros((3,len(CATEGORIES), 2))
    whole_count = 0

    for k in range(len(CATEGORIES)):
        key = CATEGORIES[k]
        sample_count = data_dict[key]
        print(key, sample_count)
        small_ap = result_dict[key + ", small"]
        medium_ap = result_dict[key + ", medium"]
        large_ap = result_dict[key + ", large"]

        result[0,k,0] = sample_count[1] #/ sample_count[0] * 100
        result[1,k,0] = sample_count[2] #/ sample_count[0] * 100
        result[2,k,0] = sample_count[3] #/ sample_count[0] * 100

        result[0,k,1] = small_ap
        result[1,k,1] = medium_ap
        result[2,k,1] = large_ap
        """
        whole_count += sample_count[0]
        fig, ax = plt.subplots(1,1)
        ax.plot(result[0,k,0],result[0,k,1],"ro")
        ax.plot(result[1,k,0],result[1,k,1],"yo")
        ax.plot(result[2,k,0],result[2,k,1],"go")
        ax.set_ylim((0,100))
        ax.set_xlabel("# of object / Whole #")
        ax.set_ylabel('mAP')
        plt.savefig("dist/"+key+".png")
        """
    
    fig, ax = plt.subplots(1,1)
    ax.plot(result[0,:,0] / np.sum(result[:,:,0]), result[0,:,1], "ro")
    ax.plot(result[1, :, 0] / np.sum(result[:,:,0]), result[1, :, 1], "yo")
    ax.plot(result[2, :, 0] / np.sum(result[:,:,0]), result[2, :, 1], "go")
    ax.set_ylim((0,100))
    ax.set_xlabel("# of object in size / # of object in whole dataset")
    ax.set_xscale("log")
    ax.set_ylabel('mAP')
    plt.savefig("fig_all.png")


"""
#train_json = "datasets/coco/annotations/instances_train2014.json"
train_json = "datasets/coco/annotations/coco_without_large.json"
def main():
    coco_api = COCO(train_json)
    large_anns = coco_api.getAnnIds(areaRng=[96 ** 2, 1e5 ** 2])
    #large_imgids = [ann.]
    large_image = torch.tensor([[coco_api.anns[li]['image_id'], coco_api.anns[li]['category_id']] for li in large_anns], dtype=torch.long)
    large_image_id = torch.unique(large_image[:,0]).tolist()
    large_image_cat, cat_count = torch.unique(large_image[:,1], return_counts=True)
    print(len(large_image_cat), cat_count)
    coco_api.dataset['images'] = [image for image in coco_api.dataset['images'] if image['id'] not in large_image_id]
    coco_api.dataset['annotations'] = [ann for ann in coco_api.dataset['annotations'] if ann['image_id'] not in large_image_id]
    coco_api.createIndex()
    large_anns = coco_api.getAnnIds(areaRng=[96 ** 2, 1e5 ** 2])

    with open('coco_without_large.json','w') as fp:
        json.dump(coco_api.dataset, fp)
"""

   
if __name__ == "__main__":
    main()


