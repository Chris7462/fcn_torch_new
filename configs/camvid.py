"""
CamVid Dataset Training Configuration
11 classes for semantic segmentation
"""


# Transform pipelines (CONFIG-DRIVEN!)
train_processes = [
    dict(type='Resize', height=360, width=480),
    dict(type='CenterCrop', height=352, width=480),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    dict(type='Normalize', mean=[0.41, 0.42, 0.43], std=[0.28, 0.27, 0.29]),
    dict(type='ToTensor')
]

val_processes = [
    dict(type='Resize', height=360, width=480),
    dict(type='CenterCrop', height=352, width=480),
    dict(type='Normalize', mean=[0.41, 0.42, 0.43], std=[0.28, 0.27, 0.29]),
    dict(type='ToTensor')
]

test_processes = [
    dict(type='Resize', height=360, width=480),
    dict(type='CenterCrop', height=352, width=480),
    dict(type='Normalize', mean=[0.41, 0.42, 0.43], std=[0.28, 0.27, 0.29]),
    dict(type='ToTensor')
]

# Dataset paths
dataset_path = './data/CamVid/'
dataset = dict(
    train=dict(
        type='CamVid',
        img_dir='./data/CamVid/701_StillsRaw_full',
        label_dir='./data/CamVid/LabeledApproved_full',
        split_file='./data/CamVid/splits/train.txt',
        dataset_info_path='./data/CamVid/splits/dataset_info.json',
        processes=train_processes
    ),
    val=dict(
        type='CamVid',
        img_dir='./data/CamVid/701_StillsRaw_full',
        label_dir='./data/CamVid/LabeledApproved_full',
        split_file='./data/CamVid/splits/val.txt',
        dataset_info_path='./data/CamVid/splits/dataset_info.json',
        processes=val_processes
    ),
    test=dict(
        type='CamVid',
        img_dir='./data/CamVid/701_StillsRaw_full',
        label_dir='./data/CamVid/LabeledApproved_full',
        split_file='./data/CamVid/splits/test.txt',
        dataset_info_path='./data/CamVid/splits/dataset_info.json',
        processes=test_processes
    )
)

#   # Model configuration
#   net = dict(
#       type='FCNs',
#   )

#   backbone = dict(
#       type='VGG16',
#       pretrained=True,
#   )

#   decoder = dict(
#       type='FCNHead',
#   )

#   # Optimizer configuration
#   optimizer = dict(
#       type='sgd',
#       lr=0.002,
#       weight_decay=5e-4,
#       momentum=0.9
#   )

#   # Learning rate scheduler
#   scheduler = dict(
#       type='StepLR',
#       step_size=30,
#       gamma=0.5
#   )

#   # Image settings
#   img_height = 352
#   img_width = 480

#   # Note: mean and std are loaded from dataset_info.json
#   # These values are computed by tools/prepare_camvid.py
#   img_norm = dict(
#       mean=[0.485, 0.456, 0.406],  # Will be overridden by dataset_info.json
#       std=[0.229, 0.224, 0.225]     # Will be overridden by dataset_info.json
#   )

# Training hyperparameters
batch_size = 16
num_workers = 4
#   num_classes = 11
#   ignore_label = 255
#   epochs = 200

#   # Logging and checkpointing
#   log_interval = 50
#   save_ep = 10  # Save checkpoint every N epochs
#   eval_ep = 1   # Evaluate every N epochs

#   # Work directory (will be extended with dataset name and timestamp by Recorder)
#   work_dirs = './work_dirs'

#   # Load checkpoint (set by command line args)
#   load_from = None

#   # Flags (set by command line args)
#   test = False
#   validate = False
