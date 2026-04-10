import torch


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


def decode_bboxes(pred_bboxes, voxel_size, out_size_factor, pc_range):
    x = pred_bboxes[..., 0:1] * voxel_size[0] * out_size_factor + pc_range[0]
    y = pred_bboxes[..., 1:2] * voxel_size[1] * out_size_factor + pc_range[1]
    z = pred_bboxes[..., 2:3]
    w = pred_bboxes[..., 3:4].exp()
    l = pred_bboxes[..., 4:5].exp()
    h = pred_bboxes[..., 5:6].exp()
    rot_sine = pred_bboxes[..., 6:7]
    rot_cosine = pred_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    vel = pred_bboxes[..., 8:]
    return torch.cat((x, y, z, w, l, h, rot, vel), -1)


def encode_bboxes(gt_bboxes, voxel_size, out_size_factor, pc_range):
    x = (gt_bboxes[..., 0:1] - pc_range[0]) / (voxel_size[0] * out_size_factor)
    y = (gt_bboxes[..., 1:2] - pc_range[1]) / (voxel_size[1] * out_size_factor)

    # x = x - torch.floor(x)
    # y = y - torch.floor(y)

    z = gt_bboxes[..., 2:3]
    w = gt_bboxes[..., 3:4].log()
    l = gt_bboxes[..., 4:5].log()
    h = gt_bboxes[..., 5:6].log()
    rot = gt_bboxes[..., 6:7]
    vel = gt_bboxes[..., 7:]
    return torch.cat((x, y, z, w, l, h, rot.sin(), rot.cos(), vel), -1)
