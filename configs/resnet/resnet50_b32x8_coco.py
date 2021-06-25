_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/coco_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(head=dict(num_classes=80))