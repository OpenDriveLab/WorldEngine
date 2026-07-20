import torch

from mmdet3d_plugin.navformer.dense_heads.diffusion_planning_head import (
    DiffusionPlanningHead,
    GridSampleCrossBEVAttention,
)


def test_navformer_bev_sampling_maps_physical_x_to_width_and_y_to_height():
    attention = GridSampleCrossBEVAttention(
        embed_dims=1,
        num_heads=1,
        num_points=1,
        bev_range_x=4.0,
        bev_range_y=2.0,
        in_bev_dims=1,
    ).eval()

    with torch.no_grad():
        value_projection = attention.value_proj[0]
        value_projection.weight.zero_()
        value_projection.weight[0, 0, 1, 1] = 1.0
        value_projection.bias.zero_()
        attention.attention_weights.weight.zero_()
        attention.attention_weights.bias.zero_()
        attention.output_proj.weight.fill_(1.0)
        attention.output_proj.bias.zero_()

    # Physical x maps to W and physical y maps to H. With align_corners=False,
    # (x=3, y=-1) maps to the center of row 0, column 3 for these ranges.
    bev_feature = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
    trajectory = torch.tensor([[[[3.0, -1.0]]]])
    queries = torch.zeros(1, 1, 1)

    output = attention(
        queries=queries,
        traj_points=trajectory,
        bev_feature=bev_feature,
        spatial_shape=(2, 4),
    )

    torch.testing.assert_close(output, torch.tensor([[[4.0]]]))


def test_diffusiondrive_expands_half_second_poses_to_uniform_tenth_second_positions():
    head = DiffusionPlanningHead.__new__(DiffusionPlanningHead)
    head._num_poses = 8

    traj_8 = torch.zeros(1, 8, 3)
    traj_8[0, :, 0] = torch.arange(1, 9, dtype=torch.float32) * 0.5
    traj_8[0, :, 1] = torch.arange(1, 9, dtype=torch.float32)

    trajectory = head._expand_to_40(traj_8)

    assert trajectory.shape == (1, 40, 3)
    torch.testing.assert_close(
        trajectory[:, 4::5],
        traj_8,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        trajectory[0, :, :2],
        traj_8.new_tensor([[0.1 * step, 0.2 * step] for step in range(1, 41)]),
    )


def test_diffusiondrive_interpolates_heading_across_pi_on_the_short_arc():
    head = DiffusionPlanningHead.__new__(DiffusionPlanningHead)
    head._num_poses = 8

    traj_8 = torch.zeros(1, 8, 3)
    traj_8[0, 0, 2] = torch.pi - 0.1
    traj_8[0, 1:, 2] = -torch.pi + 0.1

    trajectory = head._expand_to_40(traj_8)

    torch.testing.assert_close(
        trajectory[:, 4::5],
        traj_8,
        rtol=0,
        atol=0,
    )
    boundary_headings = trajectory[0, 4:10, 2]
    wrapped_deltas = torch.atan2(
        torch.sin(boundary_headings[1:] - boundary_headings[:-1]),
        torch.cos(boundary_headings[1:] - boundary_headings[:-1]),
    )
    torch.testing.assert_close(
        wrapped_deltas,
        traj_8.new_full((5,), 0.04),
        rtol=0,
        atol=1e-6,
    )
