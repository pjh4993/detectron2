import detectron2.data.datasets.builtin as builtin
from collections import defaultdict
from pycocotools.coco import COCO
import numpy as np
import json
import matplotlib.pyplot as plt


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
        whole_count += sample_count[0]
        fig, ax = plt.subplots(1,1)
        ax.plot(result[0,k,0],result[0,k,1],"ro")
        ax.plot(result[1,k,0],result[1,k,1],"yo")
        ax.plot(result[2,k,0],result[2,k,1],"go")
        ax.set_ylim((0,100))
        plt.savefig("dist/"+key+".png")
   
if __name__ == "__main__":
    main()


