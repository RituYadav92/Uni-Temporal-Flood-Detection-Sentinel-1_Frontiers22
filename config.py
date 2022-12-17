from pathlib import Path

ROOT_PATH = Path(<ADD DATASET PATH HERE---------->)
DATA_PATH = ROOT_PATH / 'Data'
IMG_PATH = DATA_PATH / "S1"
DEM_PATH = DATA_PATH / "TEST10"
JRC_PATH = DATA_PATH / "JRCWaterHand"
LABEL_PATH = DATA_PATH / "Labels"
WEIGHT_PATH = Path(<ADD WEIGHT PATH HERE---------->)
WEIGHT_file = 'ADD WEIGHT FILE NAME HERE----------'
lr = 0.0001
IMG_HEIGHT = 512
IMG_WIDTH = 512
val_batchSize = 1
train_batchSize = 2
lr = 0.0001
CLASSES = ['flood']
OUT_FOLDER = 'ADD OUTPUT FOLDER NAME HERE----------'
