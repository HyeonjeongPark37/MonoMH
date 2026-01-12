import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_merge_depth_n_conf(depth_input, conf_input, start_h, end_h, start_w, end_w):
    depth_input = depth_input.reshape(7, 7)
    conf_input = conf_input.reshape(7, 7)
    
    depth_val = depth_input[start_h:end_h, start_w:end_w].reshape(-1)
    depth_prob = conf_input[start_h:end_h, start_w:end_w].reshape(-1)
        
    merge_depth = (torch.sum((depth_val*depth_prob), dim=-1) / torch.sum(depth_prob, dim=-1))
    merge_conf = (torch.sum(depth_prob**2, dim=-1) / torch.sum(depth_prob, dim=-1))
    return merge_depth, merge_conf


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    
    regions = [
        (0, 4, 0, 4),  # top-left
        (0, 4, 2, 6),  # top-center
        (0, 4, 3, 7),  # top-right
        (2, 6, 0, 4),  # middle-left
        (2, 6, 2, 6),  # center
        (2, 6, 3, 7),  # middle-right
        (3, 7, 0, 4),  # bottom-left
        (3, 7, 2, 6),  # bottom-center
        (3, 7, 3, 7)   # bottom-right
    ]
    
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            
            score = dets[i, j, 1]
            if score < threshold: continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            multi_depth_values = dets[i, j, -147:-98]
            depth_values = dets[i, j, -98:-49]
            depth_uncers = dets[i, j, -49:]

            if isinstance(multi_depth_values, np.ndarray):    
                multi_depth_values = torch.tensor(multi_depth_values)
            if isinstance(depth_values, np.ndarray):
                depth_values = torch.tensor(depth_values)
            if isinstance(depth_uncers, np.ndarray):    
                depth_uncers = torch.tensor(depth_uncers)

            
            depth_prob = (-(0.5 * depth_uncers).exp()).exp()
            multi_depth_prob = torch.sigmoid(-0.01 * depth_uncers)

            depth = torch.sum((depth_values*depth_prob), dim=-1) / torch.sum(depth_prob, dim=-1)
            final_conf = torch.sum(depth_prob**2, dim=-1) / torch.sum(depth_prob, dim=-1)
            score *= final_conf

            multi_depth = torch.sum((multi_depth_values * depth_prob), dim=-1) / torch.sum(depth_prob, dim=-1)
            
            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 6:30])
            ry = calibs[i].alpha2ry(alpha, x)

            # dimensions decoding
            dimensions = dets[i, j, 30:33]
            dimensions += cls_mean_size[int(cls_id)]
            if True in (dimensions<0.0): continue

            # positions decoding
            x3d = dets[i, j, 33] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][1]

            locations = calibs[i].img_to_rect(x3d, y3d, multi_depth).reshape(-1)
            locations[1] += dimensions[0] / 2
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])

            if score < 0.75:
                grid_depth_values, grid_depth_prob = [], []
                for (start_h, end_h, start_w, end_w) in regions:
                    region_depth, region_conf = get_merge_depth_n_conf(multi_depth_values, multi_depth_prob, start_h, end_h, start_w, end_w)
                    grid_depth_values.append(region_depth)
                    grid_depth_prob.append(region_conf)

                multi_depth_values = torch.tensor(grid_depth_values)
                multi_depth_prob = torch.tensor(grid_depth_prob)
                for new_depth, new_conf in zip(multi_depth_values, multi_depth_prob):
                    depth_shift = multi_depth - new_depth
                    if abs(depth_shift) < 0.5 or abs(depth_shift) > 2.:
                        continue

                    new_score = score * new_conf
                    new_locations = calibs[i].img_to_rect(x3d, y3d, new_depth).reshape(-1)
                    new_locations[1] += dimensions[0] / 2

                    preds.append([cls_id, alpha] + bbox + dimensions.tolist() + new_locations.tolist() + [ry, new_score])

        results[info['img_id'][i]] = preds

    return results


def extract_dets_from_outputs(outputs, conf_mode='ada', K=50):

    roi_size = 7
    # get src outputs
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    batch, channel, height, width = heatmap.size() # get shape
    heading = outputs['heading'].view(batch,K,-1)
    size_3d = outputs['size_3d'].view(batch,K,-1)
    offset_3d = outputs['offset_3d'].view(batch,K,-1)

    heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    vis_depth = outputs['vis_depth'].view(batch,K,roi_size,roi_size)
    multi_vis_depth = outputs['multi_vis_depth'].view(batch,K,roi_size,roi_size)
    vis_depth_uncer = outputs['vis_depth_uncer'].view(batch,K,roi_size,roi_size)
    
    depth_values = vis_depth.view(batch, K, -1)
    multi_depth_values = multi_vis_depth.view(batch, K, -1)
    depth_uncers = vis_depth_uncer.view(batch, K, -1)
    
    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, heading, size_3d, xs3d, ys3d, multi_depth_values, depth_values, depth_uncers], dim=2)
    return detections


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def nms_3d_z(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []

    scores = detections[:, -1]  # Use the last column as scores
    indices = torch.argsort(scores, descending=True)
    detections = detections[indices]

    keep = []
    while detections.size(0) > 0:
        current_detection = detections[0]
        keep.append(current_detection)
        if detections.size(0) == 1:
            break

        ious = compute_iou_z(current_detection, detections[1:], kernel_size=1)
        keep_indices = (ious <= iou_threshold).nonzero(as_tuple=True)[0]
        detections = detections[keep_indices + 1]

    return torch.stack(keep)


def compute_iou_z(box, other_boxes, kernel_size):
    """
    Compute Intersection over Union (IoU) along the z-axis for 3D bounding boxes.

    Parameters:
    box (torch.Tensor): Tensor representing the (x, y, z) coordinates of the box.
    other_boxes (torch.Tensor): Tensor representing the (x, y, z) coordinates of the other boxes.
    kernel_size (int): Size of the kernel to apply max pooling along z-axis.

    Returns:
    torch.Tensor: IoU values.
    """
    z1 = box[2]
    z2 = other_boxes[:, 2]

    intersection = torch.maximum(torch.tensor(0.0), kernel_size - torch.abs(z1 - z2))
    union = kernel_size

    return intersection / union


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    if torch.__version__ == '1.6.0':
        topk_ys = (topk_inds // width).float()
    else:
        topk_ys = (topk_inds / width).floor().float()
    # topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    if torch.__version__ == '1.6.0':
        topk_cls_ids = (topk_ind // K).int()
    else:
        topk_cls_ids = (topk_ind / K).int()
    # topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti import KITTI
    from torch.utils.data import DataLoader

    dataset = KITTI('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
