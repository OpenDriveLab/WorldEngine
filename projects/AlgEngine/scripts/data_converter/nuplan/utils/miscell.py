import numpy as np
from numba import njit

def find_unique_common_from_lists(input_list1, input_list2, only_com=False):
	'''
	find common items from 2 lists, the returned elements are unique. repetitive items will be ignored
	if the common items in two elements are not in the same order, the outputs follows the order in the first list

	parameters:
		input_list1, input_list2:		two input lists
		only_com:		True if only need the common list, i.e., the first output, saving computational time

	outputs:
		list_common:	a list of elements existing both in list_src1 and list_src2	
		index_list1:	a list of index that list 1 has common items
		index_list2:	a list of index that list 2 has common items
	'''
	set1 = set(input_list1)
	set2 = set(input_list2)
	common_set = set1 & set2

	if only_com:
		return list(common_set)

	index_list1 = [i for i, item in enumerate(input_list1) if item in common_set]
	index_list2 = [i for i, item in enumerate(input_list2) if item in common_set]

	return list(common_set), index_list1, index_list2

@njit
def points_in_single_box_obb(
    points_xyz: np.ndarray,
    box_xyzlwhyaw: np.ndarray,
    eps: float = 1e-6,
):
    """
    Test whether points are inside a single oriented 3D box.

    Coordinate system:
        x: front, y: left, z: up
        yaw: rotation around z axis (radians)
        box z is center

    Args:
        points_xyz: (N, 3) array of point coordinates
        box_xyzlwhyaw: (7,) array -> [cx, cy, cz, l, w, h, yaw]
        eps: numerical tolerance

    Returns:
        count: number of points inside the box
    """
    cx, cy, cz, l, w, h, yaw = box_xyzlwhyaw

    # Translate points to box-centered coordinates
    dx = points_xyz[:, 0] - cx
    dy = points_xyz[:, 1] - cy
    dz = points_xyz[:, 2] - cz

    c = np.cos(yaw)
    s = np.sin(yaw)
    ac = abs(c)
    as_ = abs(s)

    # --- AABB prefilter ---
    hx = 0.5 * (ac * l + as_ * w)
    hy = 0.5 * (as_ * l + ac * w)
    hz = 0.5 * h

    aabb_mask = (
        (np.abs(dx) <= hx + eps) &
        (np.abs(dy) <= hy + eps) &
        (np.abs(dz) <= hz + eps)
    )

    if not np.any(aabb_mask):
        return 0

    # --- OBB exact test ---
    dxm = dx[aabb_mask]
    dym = dy[aabb_mask]
    dzm = dz[aabb_mask]

    # Rotate points into the box local frame
    x_local =  c * dxm + s * dym
    y_local = -s * dxm + c * dym
    z_local = dzm

    obb_mask = (
        (np.abs(x_local) <= 0.5 * l + eps) &
        (np.abs(y_local) <= 0.5 * w + eps) &
        (np.abs(z_local) <= 0.5 * h + eps)
    )

    inside = np.zeros(points_xyz.shape[0])
    inside[aabb_mask] = obb_mask

    return int(np.sum(inside))
