import json
import sys
import matplotlib.pyplot as plt
import numpy as np

json_list = [json.load(open(file_path)) for file_path in sys.argv[1:]]
cat_key = json_list[0].keys()
base_val_org = np.array(list(json_list[0].values()))
color = ['r','g','b','y']
size = ['small', 'medium', 'large']
acc_diff = 2

fig, axes = plt.subplots(1,3,figsize=(50,15))

for sz_id in range(len(size)):
    sz = size[sz_id]
    ax = axes[sz_id]

    x_data = np.array(list(cat_key))
    y_data = np.array(list(json_list[0].values()))

    y_data = np.array([y_data[idx] for idx in range(len(x_data)) if x_data[idx].find(sz) != -1])
    x_data = np.array([x_data[idx] for idx in range(len(x_data)) if x_data[idx].find(sz) != -1])
    ax.plot(x_data, y_data, marker='*', color=color[0],label=sys.argv[1])


    for js_id in range(1, len(json_list)):
        x_data = np.array(list(cat_key))
        y_data = np.array(list(json_list[js_id].values()))

        y_data = np.array([y_data[idx] for idx in range(len(x_data)) if x_data[idx].find(sz) != -1])
        base_val = np.array([base_val_org[idx] for idx in range(len(x_data)) if x_data[idx].find(sz) != -1])
        x_data = np.array([x_data[idx] for idx in range(len(x_data)) if x_data[idx].find(sz) != -1])
        ax.plot(x_data, y_data, marker='*', color=color[js_id],label=sys.argv[1+js_id])

        small_x_data = x_data[y_data + acc_diff < base_val]
        small_y_data = y_data[y_data + acc_diff < base_val]
        small_base_val = base_val[y_data + acc_diff < base_val]
        ax.scatter(small_x_data, small_y_data, edgecolor='red', c=color[js_id], s=100)
        ax.scatter(small_x_data, small_base_val, edgecolor=color[js_id], c='red', s=100, linewidth=3)
        small_diff = np.abs(small_base_val - small_y_data)
        ax.text(0.5, 0.5 + js_id  * 4 - 4, "dropped - size : {} , mean : {:4.2}, std: {:4.2}".format(len(small_x_data),small_diff.mean(), small_diff.std()), bbox=dict(facecolor=color[js_id], alpha=0.5))

        large_x_data = x_data[y_data > base_val + acc_diff]
        large_y_data = y_data[y_data > base_val + acc_diff]
        large_base_val = base_val[y_data > base_val + acc_diff]
        large_diff = np.abs(large_base_val - large_y_data)
        ax.text(0.5, 0.5 + (js_id + 0.5) * 4 - 4, "improved - size : {} , mean : {:4.2}, std: {:4.2}".format(len(large_x_data),large_diff.mean(), large_diff.std()), bbox=dict(facecolor=color[js_id], alpha=0.5))


    ax.legend(fontsize=12, loc='best')
    ax.set_xticklabels(x_data,fontsize=6, rotation='vertical')
    ax.set_title('Line Graph w/ different markers and colors', fontsize=20) 
    ax.set_ylabel('mAP', fontsize=14)
    ax.set_xlabel('Category', fontsize=14)



plt.savefig("result.png")




