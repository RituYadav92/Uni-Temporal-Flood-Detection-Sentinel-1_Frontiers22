import tensorflow as tf
tf.version.VERSION
from augmentation import augment_seg
from pathlib import Path
import numpy as np
import rasterio
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
import cv2 as cv
import csv
import random
from config import *

def load_csv_data(fname):  
  my_file = ROOT_PATH / fname
  x = []
  y = []
  with open(my_file,'r') as f:
    for line in csv.reader(f):      
      x.append(line[0])
      y.append(line[1])
  return x, y

def load_data():
  print('Load train images and masks ... ')
  train1_x, train1_y = load_csv_data(fname = "flood_train_data.csv")
  train2_x, train2_y = load_csv_data(fname = "flood_valid_data.csv")
  train_x, train_y = (train1_x + train2_x), (train1_y + train2_y)
  val_x, val_y = load_csv_data(fname = "flood_test_data.csv")
  return train_x, train_y, val_x, val_y

def scale_img(matrix):
    max3 = int(np.max(matrix[:, :, 2]))
    min3 = int(np.min(matrix[:, :, 2]))
    # Set min/max values
    min_values = np.array([-23, -28, min3])
    max_values = np.array([0, -5, max3])

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)

    # Scale by min/max
    matrix = np.nan_to_num((matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :]))
    matrix = np.reshape(matrix, [w, h, d])

    return matrix.clip(0, 1)
    
def GRD_toRGB(fname):
    path_img = IMG_PATH / fname

    # Read VV/VH bands
    with rasterio.open(path_img) as sar:
        sar_img = sar.read((1,2))
    sar_img = np.moveaxis(sar_img, 0, -1)

    vv_img = sar_img[:, :, 0]
    vh_img = sar_img[:, :, 1]
    x_arr = np.stack([vv_img, vh_img], axis=-1)  

    name_Split = str.split(fname, '_')
    jrc_fname = name_Split[0] + '_' + name_Split[1] + '_' + 'JRCWaterHand'+ '.tif'
    path_jrc = JRC_PATH / jrc_fname
    with rasterio.open(path_jrc) as jrc:
        jrc_img = jrc.read(1)
  
    dem_fname = name_Split[0] + '_' + name_Split[1] + '_' + 'nasadem'+ '.tif'
    path_dem = DEM_PATH / dem_fname
    with rasterio.open(path_dem) as dem:
        nasadem_img = dem.read(1)

    x,y = nasadem_img.shape
    if (x>512 or y>512):
        nasadem_img = cv.resize(nasadem_img, (512,512), interpolation = cv.INTER_AREA)

    x_img = np.zeros((512, 512, 3), dtype=np.float32)
    x_img[:, :, :2] = x_arr.copy()
    x_img[:, :, 2] = jrc_img

    x_inf = np.zeros((512, 512, 3), dtype=np.float32)
    x_inf[:, :, :2] = x_arr.copy()
    x_inf[:, :, 2] = nasadem_img[:]
  
    return scale_img(x_img), scale_img(x_inf)

# data Generator
class Cust_DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_files, label_files, batch_size = 64):              
        self.img_files = img_files
        self.label_files = label_files       
        self.batch_size = batch_size
        self.n = len(self.img_files)
        self.on_epoch_end()        

    def __len__(self):      
      return self.n // self.batch_size

    def __getitem__(self, idx):

        while True:
          batch_ind = np.random.choice(len(self.img_files), self.batch_size)          
          batch_input_img  = []
          batch_input_inf  = []
          batch_output = []

          for ind in batch_ind:            
            img, inf_raw = GRD_toRGB(self.img_files[ind])           
            img_raw = cv.GaussianBlur(img,(3,3),1)      
            
            label_path = LABEL_PATH / self.label_files[ind]            
            with rasterio.open(label_path) as lp:
              lbl = lp.read(1)

            # mask invalid pixel and set value to 0
            NA_VALUE = -1
            invalid_mask = lbl == NA_VALUE
            img_raw[:,:,0] = np.nan_to_num(np.ma.masked_array(img_raw[:,:,0], invalid_mask).filled(0) )
            img_raw[:,:,1] = np.nan_to_num(np.ma.masked_array(img_raw[:,:,1], invalid_mask).filled(0) )
            img_raw[:,:,2] = np.nan_to_num(np.ma.masked_array(img_raw[:,:,2], invalid_mask).filled(0) )

            invalid_mask = lbl == NA_VALUE
            inf_raw[:,:,0] = np.nan_to_num(np.ma.masked_array(inf_raw[:,:,0], invalid_mask).filled(0) )
            inf_raw[:,:,1] = np.nan_to_num(np.ma.masked_array(inf_raw[:,:,1], invalid_mask).filled(0) )
            inf_raw[:,:,2] = np.nan_to_num(np.ma.masked_array(inf_raw[:,:,2], invalid_mask).filled(0) )

            lbl_raw = np.where((lbl == -1), 0, lbl)
            img, inf, lbl = augment_seg(img_raw, inf_raw, lbl_raw, augmentation_name= "aug_geometric")
            
            batch_input_img += [ img]
            batch_input_inf += [ inf]
            batch_output += [ lbl.astype(np.float32) ]
          
          batch_imgx = np.array( batch_input_img )
          batch_infx = np.array( batch_input_inf )
          batch_y = np.array( batch_output )          
          
          return ([batch_imgx, batch_infx], batch_y)
          
def Inference(ind, file_x, file_y, model):
  
  fname = file_x[ind]
  name_Split = str.split(fname, '_')  

  img, inf_raw = GRD_toRGB(file_x[ind])  
  img_raw = cv.GaussianBlur(img,(3,3),1)      
  
  label_path = LABEL_PATH / file_y[ind]
  with rasterio.open(label_path) as lp:
    lbl = lp.read(1)

  # mask invalid pixel and set value to 0
  NA_VALUE = -1
  invalid_mask = lbl == NA_VALUE
  img_raw[:,:,0] = np.nan_to_num(np.ma.masked_array(img_raw[:,:,0], invalid_mask).filled(0) )
  img_raw[:,:,1] = np.nan_to_num(np.ma.masked_array(img_raw[:,:,1], invalid_mask).filled(0) )
  img_raw[:,:,2] = np.nan_to_num(np.ma.masked_array(img_raw[:,:,2], invalid_mask).filled(0) )

  inf_raw[:,:,0] = np.nan_to_num(np.ma.masked_array(inf_raw[:,:,0], invalid_mask).filled(0) )
  inf_raw[:,:,1] = np.nan_to_num(np.ma.masked_array(inf_raw[:,:,1], invalid_mask).filled(0) )
  inf_raw[:,:,2] = np.nan_to_num(np.ma.masked_array(inf_raw[:,:,2], invalid_mask).filled(0) )

  img_input = tf.expand_dims(img_raw, axis=0)
  inf_input = tf.expand_dims(inf_raw, axis=0)
  pred_mask = model.predict([img_input,inf_input])
  pred_mask = np.squeeze(pred_mask[0]).round()

  imsave(OUT_FOLDER / f"{name_Split[0]}_{name_Split[1]}_Fusion_mask.tif", pred_mask.astype(np.uint8))

  lbl = np.ma.masked_array(lbl, invalid_mask).filled(0)
  pred_mask = np.ma.masked_array(pred_mask, invalid_mask).filled(0)
  GT = lbl
  intersection = np.logical_and(GT, pred_mask).sum()
  union = np.logical_or(GT, pred_mask).sum()
  return intersection, union
