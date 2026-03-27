import torch

from dust3r.losses import Criterion, MultiLoss, Sum
from dust3r.utils.geometry import geotrf

from occany.model.must3r_blocks.geometry import normalize_pointcloud
from occany.utils.image_util import camera_to_pose_encoding


class Regr3D_multiview (Criterion, MultiLoss):
    """
    From must3r
    """
    def __init__(self, criterion, norm_mode='?avg_dis', 
                 pose_loss_value=0, 
                 loss_in_log=False, gt_scale=False):
        super().__init__(criterion)
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # use the same scale factor as ground-truth for predictions in metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.pose_loss_value = pose_loss_value
        print("pose_loss_value:", pose_loss_value)

    def get_all_pts3d(self, gt, pred, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        device = pred['pts3d'].device

        gt_c2w = [b['camera_pose'] for b in gt]
        gt_c2w = torch.stack(gt_c2w, dim=1).to(device)  # B, nimgs, 4, 4
        gt_w2c = torch.linalg.inv(gt_c2w)

        # in_camera0 = gt_w2c[:, 0]

        gt_pts3d = [b['pts3d'] for b in gt]
        gt_pts = torch.stack(gt_pts3d, dim=1).to(device)  # B, nimgs, H, W, 3
      
        gt_pts3d_local = geotrf(gt_w2c, gt_pts)  # B, nimgs, H, W, 3
        # gt_pts = geotrf(in_camera0, gt_pts3d)  # B, nimgs, H, W, 3

        valid = [b['valid_mask'] for b in gt]
        valid = torch.stack(valid, dim=1).to(device).clone()

        is_metric_scale = gt[0]['is_metric_scale'].to(device).clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis_g = gt_pts.norm(dim=-1)  # (B, nimgs, H, W)
            dis_l = gt_pts3d_local.norm(dim=-1)  # (B, nimgs, H, W)
            valid_g = valid & (dis_g <= dist_clip)
            valid_l = valid & (dis_l <= dist_clip)
        else:
            valid_g = valid
            valid_l = valid

        pr_pts = pred['pts3d'].clone()
        if 'pts3d_local' in pred:
            pr_pts_local = pred['pts3d_local'].clone()
        else:
            pr_pts_local = None

        if not self.norm_all:
            mask = ~is_metric_scale
        else:
            mask = torch.ones_like(is_metric_scale)
        
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts[mask], norm_factor_pred = normalize_pointcloud(pr_pts[mask], None, self.norm_mode, valid[mask], None,
                                                                  ret_factor=True)
            if pr_pts_local is not None:
                pr_pts_local[mask] = pr_pts_local[mask] / norm_factor_pred

        if self.norm_mode and not self.gt_scale:
            gt_pts, norm_factor = normalize_pointcloud(gt_pts, None, self.norm_mode, valid, None, ret_factor=True)
            gt_pts3d_local = gt_pts3d_local / norm_factor
            pr_pts[~mask] = pr_pts[~mask] / norm_factor[~mask]
            if pr_pts_local is not None:
                pr_pts_local[~mask] = pr_pts_local[~mask] / norm_factor[~mask]

        if self.pose_loss_value > 0:
            # gt_pose_in_camera0 = torch.einsum("bij, bhjw -> bhiw", in_camera0, gt_c2w)     
            gt_pose_in_camera0 = gt_c2w
            return gt_pts, gt_pts3d_local, pr_pts, pr_pts_local, valid_g, valid_l, gt_pose_in_camera0, {}
        else:
            return gt_pts, gt_pts3d_local, pr_pts, pr_pts_local, valid_g, valid_l, {}

    def compute_loss(self, gt, pred, **kw):
        if self.pose_loss_value > 0:
            gt_pts, gt_pts3d_local, pred_pts, pred_pts_local, mask_g, mask_l, gt_pose_in_camera0, monitoring = \
                self.get_all_pts3d(gt, pred, **kw)
            
            gt_pose_in_camera0_encoded = camera_to_pose_encoding(gt_pose_in_camera0)
            l_pose = self.criterion(gt_pose_in_camera0_encoded, pred['pose_absT_quaR'])
            l_pose = l_pose * self.pose_loss_value
        else:
            gt_pts, gt_pts3d_local, pred_pts, pred_pts_local, mask_g, mask_l, monitoring = \
                self.get_all_pts3d(gt, pred, **kw)
            l_pose = None

        bs, nimgs, h, w, _ = pred_pts.shape

        
        # loss on pts3d global
        mask_g = mask_g.reshape(bs * nimgs, h, w)
        mask_l = mask_l.reshape(bs * nimgs, h, w)

        gt_pts = gt_pts.reshape(bs * nimgs, h, w, 3)
        gt_pts = gt_pts[mask_g]
        

        pred_pts_m = pred_pts.reshape(bs * nimgs, h, w, 3)
        pred_pts_m = pred_pts_m[mask_g]

        l1 = self.criterion(pred_pts_m, gt_pts)

        # loss on pts3d local
        if pred_pts_local is not None:
            pred_pts_local = pred_pts_local.reshape(bs * nimgs, h, w, 3)
            pred_pts_local = pred_pts_local[mask_l]
            
            gt_pts3d_local = gt_pts3d_local.reshape(bs * nimgs, h, w, 3)
            gt_pts3d_local = gt_pts3d_local[mask_l]
            
            l2 = self.criterion(pred_pts_local, gt_pts3d_local)
        else:
            l2 = None

        self_name = type(self).__name__
        details = {self_name + '_pts3d': float(l1.mean())}
        if l2 is not None:
            details[self_name + '_pts3d_local'] = float(l2.mean())
        if l_pose is not None:
            details[self_name + '_pose'] = float(l_pose.mean())
        
            pred_rotmat = pred['pose_rotmat']
            pred_trans = pred['pose_trans']
            rot_error, trans_error = self._compute_pose_error(gt_pose_in_camera0, pred_rotmat, pred_trans)
            details[self_name + '_rot_error_deg'] = rot_error.item()
            details[self_name + '_trans_error'] = trans_error.item()

            pred_rot_mat_pose_regis = pred['c2w'][:, :, :3, :3]
            pred_trans_pose_regis = pred['c2w'][:, :, :3, 3]
            rot_error_pose_regis, trans_error_pose_regis = self._compute_pose_error(gt_pose_in_camera0, pred_rot_mat_pose_regis, pred_trans_pose_regis)
            details[self_name + '_rot_error_deg_pose_regis'] = rot_error_pose_regis.item()
            details[self_name + '_trans_error_pose_regis'] = trans_error_pose_regis.item()
        else:
            l_pose = 0.0

        return Sum((l1, mask_g), (l2, mask_l), (l_pose, None)), (details | monitoring)

    def _compute_pose_error(self, gt_pose_in_camera0, pred_rotmat, pred_trans):
        # Calculate rotation error in degrees between predicted and ground truth rotation matrices
        gt_rotmat = gt_pose_in_camera0[:, :, :3, :3]  # Extract rotation part from pose
        
        # Compute the relative rotation: R_rel = R_gt^T @ R_pred
        rel_rot = torch.bmm(gt_rotmat.reshape(-1, 3, 3).transpose(1, 2), pred_rotmat.reshape(-1, 3, 3))
        # Calculate rotation angle in degrees using arccos of (trace(R_rel) - 1) / 2
        rot_trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
        rot_angle = torch.acos(torch.clamp((rot_trace - 1) / 2, -1.0, 1.0)) * 180 / torch.pi
        rot_error = rot_angle.mean()
        
        gt_trans = gt_pose_in_camera0[:, :, :3, 3]
        trans_error = (gt_trans - pred_trans).norm(dim=-1).mean()
        
        return rot_error, trans_error


class ConfLoss_multiview (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: low impact parameter?
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_conf_loss(self, loss, pred_conf):
        conf, log_conf = self.get_conf_log(pred_conf)
        conf_loss = loss * conf - self.alpha * log_conf
        conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
        return conf_loss

    def compute_loss(self, gt, pred, **kw):
        # compute per-pixel loss
        ((loss_g, msk_g), (loss_l, msk_l), \
            (loss_pose, _)), details = self.pixel_loss(gt, pred, **kw)

  
        # compute conf loss for global point and local pointmap separately, then sum
        pred_conf = pred['conf']
        bs, nimgs, h, w = pred_conf.shape
        
        pred_conf = pred_conf.reshape(bs * nimgs, h, w)

        conf_loss_g = self.compute_conf_loss(loss_g, pred_conf[msk_g])
       
        if loss_l is not None:
            conf_loss_l = self.compute_conf_loss(loss_l, pred_conf[msk_l])
        else:
            conf_loss_l = 0
        details_conf = dict(conf_loss_g=float(conf_loss_g), conf_loss_l=float(conf_loss_l), **details)

        if torch.is_tensor(loss_pose):
            loss_pose = loss_pose.mean()

        return conf_loss_g + conf_loss_l + loss_pose, details_conf