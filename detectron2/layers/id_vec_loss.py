import torch
from torch import nn


class IdLoss(nn.Module):
    """
    """
    def __init__(self):
        super(IdLoss, self).__init__()
        self.push_loss_func = nn.TripletMarginLoss(margin=1.0, p=2, reduction="none")

    def forward(self, grouped_id_vec):
        """

        """
        pull_target = []
        push_target = []

        anchor_per_group = []

        for group in grouped_id_vec:
            anchor_per_group.append(group.mean())

        for gid in range(len(grouped_id_vec)):
            pos_group = grouped_id_vec[gid]
            neg_group = [elt for num, elt in enumerate(anchor_per_group) if not num == gid]
            pos_anchor = pos_group.mean().reshape(-1,1)

            pull_target.append({"anchor": pos_anchor, "pos_group": pos_group})
            push_target.append({"anchor": pos_anchor, "pos_group": pos_group, "neg_group": torch.stack(neg_group).reshape(-1,1)})
            
        push_loss = []
        pull_loss = []
        for pull in pull_target:
            pos_anchor = pull['anchor']
            pos_group = pull['pos_group']
            pull_loss.append(((pos_group - pos_anchor) **2).mean())

        for push in push_target:
            pos_anchor = push['anchor']
            pos_group = push['pos_group']
            neg_group = push['neg_group']

            len_pos_group = pos_group.shape[0]
            len_neg_group = neg_group.shape[0]

            loss = self.push_loss_func(
                pos_anchor.repeat(len_pos_group * len_neg_group,1),
                pos_group.repeat(1,len_neg_group).reshape(-1, 1),
                neg_group.repeat(len_pos_group,1)) 
            #weight = 1 - torch.clamp((pos_group.repeat(1,len_neg_group).reshape(-1, 1) - neg_group.repeat(len_pos_group,1)) ** 2, min=0, max=1)
            weight = 1

            push_loss.append(loss.mean() * weight)
        
        return torch.stack(pull_loss).mean(), torch.stack(push_loss).mean()

