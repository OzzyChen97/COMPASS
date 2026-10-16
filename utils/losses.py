import torch
import torch.nn as nn

from dataset.config import get_dataset_spec

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, dataset_name=None):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.spec = get_dataset_spec(dataset_name) if dataset_name else None
        self.motion_dim = (
            self.spec.dim_pose
            if self.spec is not None and self.spec.representation == "smplx322"
            else (nb_joints - 1) * 12 + 4 + 3 + 4
        )

    def _masked_loss(self, prediction, target, mask=None):
        if mask is None:
            return self.Loss(prediction, target)
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)
        while mask.ndim < prediction.ndim:
            mask = mask.unsqueeze(-1)
        if isinstance(self.Loss, nn.L1Loss):
            values = torch.abs(prediction - target)
        elif isinstance(self.Loss, nn.MSELoss):
            values = torch.square(prediction - target)
        else:
            values = torch.nn.functional.smooth_l1_loss(
                prediction, target, reduction="none")
        denominator = mask.sum() * prediction.shape[-1]
        return (values * mask).sum() / denominator.clamp_min(1.0)
        
    def forward(self, motion_pred, motion_gt, mask=None) :
        loss = self._masked_loss(
            motion_pred[..., : self.motion_dim],
            motion_gt[..., :self.motion_dim],
            mask,
        )
        return loss

    def forward_vel(self, motion_pred, motion_gt, mask=None) :
        if self.spec is not None and self.spec.velocity_slices:
            prediction = torch.cat([
                motion_pred[..., start:end]
                for start, end in self.spec.velocity_slices
            ], dim=-1)
            target = torch.cat([
                motion_gt[..., start:end]
                for start, end in self.spec.velocity_slices
            ], dim=-1)
            prediction = prediction[:, 1:] - prediction[:, :-1]
            target = target[:, 1:] - target[:, :-1]
            velocity_mask = None
            if mask is not None:
                velocity_mask = mask[:, 1:] & mask[:, :-1]
            return self._masked_loss(prediction, target, velocity_mask)
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss

