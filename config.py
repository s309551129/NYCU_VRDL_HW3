# _base_ = './configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'
_base_ = './configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'


# model config
model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4])
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True,
                loss_weight=18.0
            )
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            sampler=dict(
                num=512,
            )
        ),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=2000
        ),
        rcnn=dict(
            sampler=dict(
                num=1000,
                pos_fraction=0.5
            )
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=2000
        ),
        rcnn=dict(
            max_per_img=1000,
            mask_thr_binary=0.5
        )
    ))


# pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
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


# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4)


# dataset
dataset_type = 'COCODataset'
classes = ('nucleus',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='../dataset/dataset/train_images/',
        classes=classes,
        ann_file='../dataset/dataset/train_coco_format.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='../dataset/dataset/valid_images/',
        classes=classes,
        ann_file='../dataset/dataset/valid_coco_format.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='../dataset/dataset/test/',
        classes=classes,
        ann_file='../dataset/dataset/test_coco_format.json',
        pipeline=test_pipeline))

# epoch
runner = dict(type='EpochBasedRunner', max_epochs=400)


# pretrained weight
# load_from = './weight/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth'
load_from = './weight/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'


# log config
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)

# evaluation
# evaluation = dict(metric=['segm'])
