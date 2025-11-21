from utils import Config, Registry, build_from_cfg
from datasets import build_dataset, build_dataloader


# Load config
cfg = Config.from_file('configs/camvid.py')

# Build datasets
train_dataset = build_dataset(cfg.dataset.train, cfg)

# Build dataloaders (recommended for training)
train_loader = build_dataloader(cfg.dataset.train, cfg, is_train=True)



# Step 1: Create registries for different component types
#   MODELS = Registry('models')
#   BACKBONES = Registry('backbones')
#   DECODERS = Registry('decoders')
#   OPTIMIZERS = Registry('optimizers')
DATASETS = Registry('datasets')

# Step 2: Register your actual classes
#   @MODELS.register_module
#   class FCNs:
#       def __init__(self, num_classes=11):
#           self.num_classes = num_classes
#           print(f"FCNs model created with {num_classes} classes")

#   @BACKBONES.register_module
#   class VGG16:
#       def __init__(self, pretrained=False):
#           self.pretrained = pretrained
#           print(f"VGG16 backbone created, pretrained={pretrained}")

#   @DECODERS.register_module
#   class FCNHead:
#       def __init__(self, in_channels=512, num_classes=11):
#           self.in_channels = in_channels
#           self.num_classes = num_classes
#           print(f"FCNHead decoder created")

#   @OPTIMIZERS.register_module
#   class sgd:
#       def __init__(self, lr, weight_decay, momentum):
#           self.lr = lr
#           self.weight_decay = weight_decay
#           self.momentum = momentum
#           print(f"SGD optimizer: lr={lr}, momentum={momentum}")

@DATASETS.register_module
class CamVid:
    def __init__(self, img_dir, label_dir, split_file, dataset_info_path):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.split_file = split_file
        print(f"CamVid dataset created with split: {split_file}")

# Step 3: Load the config file
cfg = Config.from_file('configs/camvid.py')

# Step 4: Build components using build_from_cfg
#   print("\n=== Building Model ===")
#   model = build_from_cfg(cfg.net, MODELS, num_classes=cfg.num_classes)

#   print("\n=== Building Backbone ===")
#   backbone = build_from_cfg(cfg.backbone, BACKBONES)

#   print("\n=== Building Decoder ===")
#   decoder = build_from_cfg(cfg.decoder, DECODERS, num_classes=cfg.num_classes)

#   print("\n=== Building Optimizer ===")
#   optimizer = build_from_cfg(cfg.optimizer, OPTIMIZERS)

print("\n=== Building Datasets ===")
train_dataset = build_from_cfg(cfg.dataset.train, DATASETS)
val_dataset = build_from_cfg(cfg.dataset.val, DATASETS)
test_dataset = build_from_cfg(cfg.dataset.test, DATASETS)

# Step 5: Access the built objects
#   print("\n=== Verification ===")
#   print(f"Model: {model}")
#   print(f"Backbone pretrained: {backbone.pretrained}")
#   print(f"Optimizer LR: {optimizer.lr}")
#   print(f"Train dataset split: {train_dataset.split_file}")

cfg = Config({'batch_size': 16}, filename=None)
