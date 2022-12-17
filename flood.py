import tensorflow as tf
tf.version.VERSION
from tensorflow.keras.layers import Input
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import os
from utils import load_data, Cust_DatasetGenerator, Inference
from model import Resnet50_UNet
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
from config import *

def train_fusion():
    model = Resnet50_UNet(n_classes, in_img, in_inf)
    model.compile(optimizer, loss = total_loss, metrics = metrics)
    earlystopper = EarlyStopping(patience=100, verbose=1)
    checkpointer = ModelCheckpoint('Fusion_unet_checkpoint.h5', verbose=1, save_best_only=True)
    # freeze and tune
    for layer in model.layers:  
      if 'DEC_' not in layer.name:
        layer.trainable = False
    model.fit(my_training_batch_generator, validation_data=my_validation_batch_generator,  epochs=2,  steps_per_epoch=int(len(train_x)/train_batchSize), validation_steps=int(len(val_x)/val_batchSize) ,callbacks=[scheduler, earlystopper, checkpointer])
    # unfreeze and train
    for layer in model.layers:
      layer.trainable = True
    model.fit(my_training_batch_generator, validation_data=my_validation_batch_generator,  epochs=100,  steps_per_epoch=int(len(train_x)/train_batchSize), validation_steps=int(len(val_x)/val_batchSize) ,callbacks=[scheduler, earlystopper, checkpointer])

def evaluate_fusion():
    model = Resnet50_UNet(n_classes, in_img, in_inf)
    model.load_weights(WEIGHT_PATH/WEIGHT_file)
    intersection, union, iou = 0, 0, 0
    file_x, file_y = val_x, val_y
    OUT_FOLDER = WEIGHT_PATH / 'Pred_Mask'
    if not os.path.exists(OUT_FOLDER): os.mkdir(OUT_FOLDER)
    
    for ind in range(len(file_x)):
      ints, un = Inference(ind, file_x, file_y, model)
      intersection = intersection + ints
      union = union + un
      
    iou = intersection / union
    print("IOU Score", iou)

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train the network.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    args = parser.parse_args()
    
    # load data
    train_x, train_y, val_x, val_y = load_data()
    my_training_batch_generator = Cust_DatasetGenerator(train_x, train_y, batch_size=train_batchSize)
    my_validation_batch_generator = Cust_DatasetGenerator(val_x, val_y, batch_size=val_batchSize)
    
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    in_img = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
    in_inf = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
    
    # define loss, optimizer, lr etc.
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = 0.2 * dice_loss + (0.8 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

    if args.command == "train":
        print('In training Fusion Network')
        train_fusion()
        
    if args.command == "evaluate":
        print('Evaluating Fusion Network')
        evaluate_fusion()
