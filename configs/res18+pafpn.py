_base_ = [
    #'./_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    #pretrained='open-mmlab://detectron/resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
   #     frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,),
   #     style='caffe'),
    neck=dict(
        type='PAFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=5,
        in_channels=128,
        stacked_convs=4,
        feat_channels=128,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
train_data = dict(
     type = "UG2Dataset",
     classes=["car", "bus", "bicycle", "motorcycle", "person"],
     ann_file="/home/wang_shuai/RTTS/ImageSets/Main/test.txt",
     data_root="/home/wang_shuai/RTTS/",
     pipeline=train_pipeline
)
test_data = dict(
     type = "UG2Dataset",
     classes=["car", "bus", "bicycle", "motorbike", "person"],
     ann_file="/home/wang_shuai/RTTS/ImageSets/Main/demo_test.txt",
     data_root="/home/wang_shuai/RTTS/",
     pipeline= test_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=train_data,
    val=test_data,
    test=test_data)
# optimizer
optimizer = dict(type = "AdamW",
    lr=0.01,
    weight_decay=0.001,
                 )
optimizer_config = dict(
    #_delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=30)
