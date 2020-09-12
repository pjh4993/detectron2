from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from detectron2.layers import peak_stimulation_ori, peak_stimulation_ori_gt, pr_conv2d



class Addedmodule(torch.nn.Module):
    def __init__(self,dimension):
        super(Addedmodule, self).__init__()
        self.branch1=nn.Sequential(nn.BatchNorm2d(int(dimension/2)),nn.ReLU(inplace=True),nn.Conv2d(int(dimension/2), int(dimension/3), kernel_size=1, bias=True))
        self.branch2=nn.Sequential(nn.BatchNorm2d(int(dimension/2)),nn.ReLU(inplace=True),nn.Conv2d(int(dimension/2), int(dimension/3), kernel_size=1, bias=True))

    def forward(self, x):
        y = x.chunk(2, 1)
        y0=self.branch1(y[0].contiguous())
        y1=self.branch2(y[1].contiguous())
        return y0,y1

class Jointable(nn.Module):
    def __init__(self):
        super(Jointable, self).__init__()
        
    def forward(self, x):
        y = torch.cat(x, 1)
        return y

class PeakResponseMapping(object):
 
    def __init__(self, args):
        self.backbone = args[0]
        cfg = args[1]

        super(PeakResponseMapping, self).__init__()

        self.inferencing = True
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = cfg.MODEL.PRM.ENABLE_BACK_PROP
        # window size for peak finding
        self.win_size = cfg.MODEL.PRM.WIN_SIZE
        # sub-pixel peak finding
              
    """
    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)
    
    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)
    
    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)
    """

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    """
    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]
            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None
            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def refine_selected_instances(self, selected_instances, deleted_instances, count):
        count_cur=np.zeros((len(count),))
        len1=len(selected_instances)
        for ins in selected_instances:
            count_cur[ins[1]]+=1
        deleted_instances_sep={}
        for ins in deleted_instances:
            if ins[1] not in deleted_instances_sep.keys():
                deleted_instances_sep[ins[1]]=[]
                deleted_instances_sep[ins[1]].append(ins)
            else:
                deleted_instances_sep[ins[1]].append(ins)
        
        check_sum=0
        for val in deleted_instances_sep.values():
            check_sum+=len(val)
        if check_sum!=len(deleted_instances):
            print(check_sum,len(deleted_instances))
            print('error')
        
        for i in range(len(count)):
            if count_cur[i]<count[i]:
                count_need=count[i]-count_cur[i]
                # print(i,count_need)
                # print(deleted_instances_sep.keys())
                if i in deleted_instances_sep.keys():
                    if len(deleted_instances_sep[i])<=count_need:
                        selected_instances+=deleted_instances_sep[i]
                    else:
                        selected_instances+=deleted_instances_sep[i][:int(count_need)]
        len2=len(selected_instances)
        print(len2-len1)
        if len2>len1:
           print('added')
        return selected_instances
    """
    """
    import cv2
    from scipy.misc import imresize
    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):        
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]
        
        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)
        
        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)
            
            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT, np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                    (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                        peak_response_map[contour_mask].sum() - \
                        penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask
            
            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))
                
        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold, merge_peak_response)
        return [dict(score=v[0], category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    def instance_seg2(self, class_response_maps,class_response_maps1, peak_list, peak_response_maps, retrieval_cfg):        
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        class_response_maps1 = class_response_maps1.squeeze().cpu().numpy()
        class_response_maps1=class_response_maps1/196.0
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]
        
        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)
        
        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        # density_factor=1.0
        # print(len(peak_response_maps))
        proposals_reshape=[]
        try:
            for i in range(min(proposal_count, len(proposals))):
                if proposal_count>len(proposals):
                    print(proposal_count,len(proposals))
                proposals_reshape.append(imresize(proposals[i].astype(int), class_response_maps1[0].shape, interp='nearest'))
        except:
            print(class_response_maps1.shape)
            dd

        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor, density_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor, density_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)
            
            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                raw_mask2 = proposals_reshape[j]
                #raw_mask2 = imresize(proposals[j].astype(int), class_response_maps1[class_idx].shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT, np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                mask2 = raw_mask2.astype(bool)
                # mask2 = (raw_mask2>0.5)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                    (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                        peak_response_map[contour_mask].sum() - \
                        penalty_factor * bg_response[mask].sum()-\
                        density_factor * np.absolute(1-class_response_maps1[class_idx][mask2].sum())
                    # val=1/(0.01+np.absolute(1-class_response_maps1[class_idx][mask2].sum()))
                    if val > max_val:
                        # print(i,balance_factor * peak_response_map[mask].sum(),peak_response_map[contour_mask].sum(),penalty_factor * bg_response[mask].sum(),
                        #     density_factor * np.absolute(1-class_response_maps1[class_idx][mask2].sum()))                        
                        max_val = val
                        instance_mask = mask
            
            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))
        
        if instance_list is None:
            print("instance_list is none")
        if instance_list is not None:      
            instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
            if nms_threshold is not None:
                instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold, merge_peak_response)
            return [dict(score=v[0],category=v[1], mask=v[2], prm=v[3]) for v in instance_list], class_response_maps1
        else:
            return None, class_response_maps1
    """
    
    def __call__(self, class_threshold=0.5, peak_threshold=3, retrieval_cfg=None):  
        #Feature pyramid forward 처리 해야함
        features, input = self.backbone.class_response_maps()

        #input.requires_grad_()
        peak_list_whole = []
        aggregation_whole = []
        peak_map_whole = []
        for feature_per_level in features:
            class_response_maps = feature_per_level.sigmoid()
            peak_list_per_level, aggregation_per_level, peak_map_per_level = peak_stimulation_ori(class_response_maps,win_size=self.win_size)
            peak_list_whole.append(peak_list_per_level)
            aggregation_whole.append(aggregation_per_level)
            peak_map_whole.append(peak_map_per_level)

        valid_peak_whole = []
        peak_response_whole = []

        if self.inferencing:
            for l in range(len(peak_list_whole)):
                class_response_maps = features[l].sigmoid()
                peak_list = peak_list_whole[l]
                aggregation = aggregation_whole[l]

                peak_response_maps = []
                valid_peak_list = []
                grad_output = class_response_maps.new_empty(class_response_maps.size())
                peak_val_list=[]

                agg_class=aggregation>=class_threshold
                if torch.sum(agg_class)==0:
                    class_threshold=torch.max(aggregation)

                for idx in range(peak_list.size(0)):
                    if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
                        peak_val_list.append(class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]])

                peak_threshold=min([peak_threshold,max(peak_val_list)])

                for idx in range(peak_list.size(0)):
                    assert(aggregation.shape[0]>peak_list[idx,0])
                    assert(aggregation.shape[1]>peak_list[idx,1])
                    if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
                        peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                        # print(peak_val)
                        if peak_val >= peak_threshold:
                            grad_output.zero_()
                            # starting from the peak
                            grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                            if input.grad is not None:
                                input.grad.zero_()
                            class_response_maps.backward(grad_output, retain_graph=True)
                            prm = input.grad.detach().sum(1).clone().clamp(min=0)
                            peak_response_maps.append(prm / prm.sum())
                            valid_peak_list.append(peak_list[idx, :])
                class_response_maps = class_response_maps.detach()
                aggregation = aggregation.detach()

                if len(peak_response_maps) > 0:
                    valid_peak_list = torch.stack(valid_peak_list)
                    peak_response_maps = torch.cat(peak_response_maps, 0)
                    # classification confidence scores, class-aware and instance-aware visual cues
                
                aggregation_whole[l] = aggregation
                valid_peak_whole.append(valid_peak_list)
                peak_response_whole.append(peak_response_maps)
                
            return aggregation, features, valid_peak_whole, peak_response_whole
        else:
            return aggregation, features

    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self