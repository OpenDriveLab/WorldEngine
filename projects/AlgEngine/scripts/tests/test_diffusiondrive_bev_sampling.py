import torch

from mmdet3d_plugin.navformer.dense_heads.diffusion_planning_head import (
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
