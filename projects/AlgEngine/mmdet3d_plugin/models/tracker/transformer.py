import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (
    TransformerLayerSequence,
    build_transformer_layer_sequence,
)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DCamTransformerPlus(BaseModule):
    """Implements the DeformableDETR transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        decoder=None,
        reference_points_aug=False,
        **kwargs
    ):
        super(Detr3DCamTransformerPlus, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.reference_points_aug = reference_points_aug
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))

        # self.cam_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))

        # move ref points to tracker
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # normal_(self.level_embeds)
        # normal_(self.cam_embeds)

    def forward(
        self, mlvl_feats, query_embed, reference_points, reg_branches=None, **kwargs
    ):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, 2*embed_dim], can be splitted into
                query_feat and query_positional_encoding.
            reference_points (Tensor): The corresponding 3d ref points
                for the query with shape (num_query, 3)
                value is in inverse sigmoid space
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder, has shape \
                      (num_dec_layers, num_query, bs, embed_dims)
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, 3)
                
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(dim=0).expand(bs, -1, -1)

        if self.training and self.reference_points_aug:
            reference_points = reference_points + torch.randn_like(reference_points)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs
        )

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER.register_module()
class Detr3DCamTrackTransformer(BaseModule):
    """Implements the DeformableDETR transformer.
        Specially designed for track: keep xyz trajectory, and
        kep bbox size(which should be consisten across frames)

    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        decoder=None,
        reference_points_aug=False,
        **kwargs
    ):
        super(Detr3DCamTrackTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.reference_points_aug = reference_points_aug
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))

        # self.cam_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))

        # move ref points to tracker
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        mlvl_feats,
        query_embed,
        reference_points,
        ref_size,
        reg_branches=None,
        **kwargs
    ):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, 2*embed_dim], can be splitted into
                query_feat and query_positional_encoding.
            reference_points (Tensor): The corresponding 3d ref points
                for the query with shape (num_query, 3)
                value is in inverse sigmoid space
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (num_query, 3)
                value in log space. 
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder, has shape \
                      (num_dec_layers, num_query, bs, embed_dims)
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, 3)
                
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(dim=0).expand(bs, -1, -1)
        ref_size = ref_size.unsqueeze(dim=0).expand(bs, -1, -1)

        # add augmentation to the reference points' location
        if self.training and self.reference_points_aug:
            reference_points = reference_points + torch.randn_like(reference_points)
        reference_points = reference_points.sigmoid()
        
        # decoder
        query = query.permute(1, 0, 2)
        # memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references, inter_box_sizes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            ref_size=ref_size,
            **kwargs
        )

        return inter_states, inter_references, inter_box_sizes


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DCamTrackPlusTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=True, **kwargs):

        super(Detr3DCamTrackPlusTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        ref_size=None,
        **kwargs
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The 3d reference points
                associated with each query. shape (num_query, 3).
                value is in inevrse sigmoid space
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (bs, num_query, 3)
                value in log space.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_box_sizes = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                ref_size=ref_size,
                **kwargs
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                ref_pts_update = torch.cat(
                    [
                        tmp[..., :2],
                        tmp[..., 4:5],
                    ],
                    dim=-1,
                )
                ref_size_update = torch.cat([tmp[..., 2:4], tmp[..., 5:6]], dim=-1)
                assert reference_points.shape[-1] == 3

                new_reference_points = ref_pts_update + inverse_sigmoid(
                    reference_points
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                # add in log space
                # ref_size = (ref_size.exp() + ref_size_update.exp()).log()
                ref_size = ref_size + ref_size_update
                if lid > 0:
                    ref_size = ref_size.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_box_sizes.append(ref_size)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_box_sizes),
            )

        return output, reference_points, ref_size
