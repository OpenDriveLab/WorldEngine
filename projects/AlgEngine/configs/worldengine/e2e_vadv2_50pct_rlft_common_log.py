"""
RLFT Common Log: RL fine-tuning on normal (non-failure) data only.
Key: normal_only=True, serves as ablation baseline for RLFT.
See docs/config_guide.md for full documentation.
"""
import os

_base_ = ["../_base_/default_runtime.py"]
custom_imports = dict(imports=['mmdet3d_plugin'])

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


# nuPlan/OpenScene/NAVSIM
class_names = ['vehicle', 'bicycle', 'pedestrian',
               'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
vehicle_id_list = [0, 1]
group_id_list = [[0], [1], [2], [3, 4, 5, 6]]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
queue_length = 4  # each sequence contains `queue_length` frames.

## tracking
past_steps = 3
fut_steps = 4

### planning ###
planning_steps = 8
use_col_optim = False

# Other settings
train_gt_iou_threshold=0.3

# data path
train_dataset_type = "NavSimOpenSceneE2EFineTune"
dataset_type = "NavSimOpenSceneE2E"
file_client_args = dict(backend="disk")

# Get WORLDENGINE_ROOT from environment variable
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))
data_root = os.path.join(WORLDENGINE_ROOT, "data/raw/openscene-v1.1/")
info_root = os.path.join(WORLDENGINE_ROOT, "data/alg_engine/merged_infos_navformer/")
img_root_train = data_root + "sensor_blobs/trainval"
img_root_test = data_root + "sensor_blobs/test"

ann_file_train = info_root + "nuplan_openscene_navtrain.pkl"
ann_file_val = info_root + "nuplan_openscene_navtest.pkl"
ann_file_test = info_root + "nuplan_openscene_navtest.pkl"
nav_filter_path_train = "configs/navsim_splits/navtrain_split/navtrain_50pct.yaml"
nav_filter_path_val = "configs/navsim_splits/navtest_split/navtest.yaml"
nav_filter_path_test = "configs/navsim_splits/navtest_split/navtest.yaml"

finetune_yaml = [
    "configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_collision.yaml",
    "configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_ep_1pct.yaml",
    "configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_off_road.yaml",
]

model = dict(
    type="NAVFormer",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=len(class_names),
    vehicle_id_list=vehicle_id_list,
    pc_range=point_cloud_range,
    lora_finetuning=True,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe',
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),  # hyper-param for query dropping mentioned in MOTR
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=len(class_names),
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
    ),  # loss cfg for tracking

    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=len(class_names),
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            num_cams=8,
            fix_temporal_shift=True,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            num_cams=8,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names),
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    planning_head=dict(
        type='TrajScoringHeadRL',
        reward_shaping=True,
        use_lora=True,
        trans_use_lora=True,
        rl_finetuning=True,
        importance_sampling=True,
        orig_IL=True,
        rl_loss_weight=dict(
            bce=0.0,
            rank=0.0,
            PG=0.01,
            entropy=1.0
        ),

        num_poses=40,
        d_ffn=256 * 4,
        d_model=256,
        vocab_path=os.path.join(WORLDENGINE_ROOT, "data/alg_engine/test_8192_kmeans.npy"),
        nhead=8,
        nlayers=1,
        num_commands=4,
        transformer_decoder=dict(
            type='BEVOnlyMotionTransformerDecoder',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,
            transformerlayers=dict(
                type='MotionTransformerAttentionLayer',
                batch_first=True,
                use_lora=True,
                lora_rank=16,
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttention',
                        num_steps=planning_steps,
                        embed_dims=_dim_,
                        num_levels=1,
                        num_heads=8,
                        num_points=4,
                        sample_index=-1,
                        use_lora=True,
                        lora_rank=16
                    ),
                ],

                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
        ),
        bev_h=bev_h_,
        bev_w=bev_w_,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesWithDownsample", to_float32=True, img_root=img_root_train, downsample_factor=2),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "sdc_planning",
            "sdc_planning_mask",
            "command",
            "sdc_planning_world",
            "sdc_planning_past",
            "sdc_planning_mask_past",
            "gt_pre_command_sdc",
            "sdc_status",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "score",
            "fail_mask",
        ],
    ),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True, file_client_args=file_client_args, img_root=img_root_test),
    dict(type="ScaleMultiViewImage3D", scale=0.5),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D", keys=[
                                            #############
                                            "img",
                                            "timestamp",
                                            "l2g_r_mat",
                                            "l2g_t",
                                            "sdc_planning",
                                            "sdc_planning_mask",
                                            "command",
                                            "sdc_planning_world",
                                            "sdc_planning_past",
                                            "sdc_planning_mask_past",
                                            "gt_pre_command_sdc",
                                            "sdc_status",
                                            "no_at_fault_collisions",
                                            "drivable_area_compliance",
                                            "ego_progress",
                                            "time_to_collision_within_bound",
                                            "comfort",
                                            "score",
                                        ]
            ),
        ],
    ),
]


data = dict(
    samples_per_gpu=2,      # batch size
    workers_per_gpu=4,      # more workers do not increase speed
    train=dict(
        type=train_dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        nav_filter_path=nav_filter_path_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        past_steps=past_steps,
        fut_steps=fut_steps,
        planning_steps=planning_steps,
        load_interval=1,
        box_type_3d="LiDAR",
        fix_can_bus_rotation=True,
        finetune_yaml=finetune_yaml,
        normal_only=True,
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        use_valid_flag=True,
        ann_file=ann_file_val,
        nav_filter_path=nav_filter_path_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        past_steps=past_steps,
        fut_steps=fut_steps,
        classes=class_names,
        modality=input_modality,
        eval_mod=[],
        planning_steps=planning_steps,
        fix_can_bus_rotation=True,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        use_valid_flag=True,
        ann_file=ann_file_test,
        nav_filter_path=nav_filter_path_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        past_steps=past_steps,
        fut_steps=fut_steps,
        planning_steps=planning_steps,
        classes=class_names,
        modality=input_modality,
        eval_mod=[],
        fix_can_bus_rotation=True,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 8
evaluation = dict(interval=8, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
load_from = os.path.join(WORLDENGINE_ROOT, "data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth")
find_unused_parameters = True
