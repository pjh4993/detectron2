import matplotlib.pyplot as plt
import numpy as np
import json 
import os

#inf_root = '/media/pjh3974/output/detectron2/COCO-Detection/Faster_rcnn_FCOS_FPN_1x_e2e/inference'
#box_result = json.read(os.path.join(inf_root, 'box_dict.json'))

x = np.arange(6)
mean = np.array([67.480, 63.264, 77.791, 74.594, 68.540, 35.941])
std = np.array([42.349, 46.736, 47.356, 48.992, 40.671, 12.806])


plt.errorbar(x, mean, yerr=std, fmt='-o', color='black',
             ecolor='lightgray', elinewidth=1, capsize=3, label='FCOS with ROI Head');

plt.plot(x, [3*3] * len(x), color='b', label='FCOS')
plt.title('Area of ROI on feature space per resolution')
plt.xticks(np.arange(6), ('all', 'lvl0', 'lvl1', 'lvl2', 'lvl3', 'lvl4'))
plt.legend(fontsize=9, loc='best')

plt.show()