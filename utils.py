import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.ndimage
from scipy import special
from osgeo import gdal

# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle

def normalize_sar(im):
    return ((np.log(np.clip(im,0.24,np.max(im))) - m) * 255 / (M - m)).astype('float32')

def denormalize_sar(im):
    return np.exp((M - m) * (np.squeeze(im)).astype('float32') + m)

def tiff_open( imgPath):

    imgPath = imgPath

    image = gdal.Open(imgPath)
    num_bands = image.RasterCount
    # print('original_bands:',num_bands)
    if num_bands == 1:
        band1 = np.expand_dims(np.array(image.GetRasterBand(1).ReadAsArray()), axis=2)
        img_array = band1
    elif num_bands == 3:
        band1 = np.expand_dims(np.array(image.GetRasterBand(1).ReadAsArray()), axis=2)
        band2 = np.expand_dims(np.array(image.GetRasterBand(2).ReadAsArray()), axis=2)
        band3 = np.expand_dims(np.array(image.GetRasterBand(3).ReadAsArray()), axis=2)
        img_array = np.concatenate([band1, band2, band3], axis=2)
    else:
        raise ValueError('This function only supprots images with 1 or 3 bands ')

    original_path = imgPath
    height, width, bands = img_array.shape
    return img_array.astype(np.float32)

def save_gdal_image(image, outfile, format, image_original_path):

    outdriver = gdal.GetDriverByName('GTiff')
    
    original = gdal.Open(image_original_path)

    trans = original.GetGeoTransform()
    projs = original.GetProjection()

    width, height= image.shape
    bands = 1
    bands_model_through = original.RasterCount


    if format == "uint8":
        datatype = gdal.GDT_Byte
    elif format == "uint16":
        datatype = gdal.GDT_UInt16
    elif format == "float":
        datatype = gdal.GDT_Float64
    elif format == "uint32":
        datatype = gdal.GDT_UInt32
    else:
        raise ValueError("Unsupported format. Please use 'uint8', 'uint16', 'float', or 'uint32'.")

    outdata = outdriver.Create(outfile, width, height, bands, datatype)
    outdata.SetGeoTransform(trans)
    outdata.SetProjection(projs)
    if bands == 1 :
        outdata.GetRasterBand(1).WriteArray(image[:, :])
    elif bands == 3 and bands_model_through == 3:
        outdata.GetRasterBand(1).WriteArray(image[:, :, 0])
        outdata.GetRasterBand(2).WriteArray(image[:, :, 1])
        outdata.GetRasterBand(3).WriteArray(image[:, :, 2])
    elif bands == 3 and bands_model_through == 1:
        outdata.GetRasterBand(1).WriteArray((image[:, :, 0] + image[:,:,1] + image[:,:,2])//3)
    elif bands == 4:
        outdata.GetRasterBand(1).WriteArray(image[:, :, 0])
        outdata.GetRasterBand(2).WriteArray(image[:, :, 1])
        outdata.GetRasterBand(3).WriteArray(image[:, :, 2])
        outdata.GetRasterBand(4).WriteArray(image[:, :, 3])
    else:
        raise ValueError("This function only supports images with 1, 3, or 4 bands.")

    outdata.FlushCache()
    outdata = None


def load_sar_images(filelist):
    if not isinstance(filelist, list):
        # im = np.load(filelist)
        im = tiff_open(filelist)
        im = normalize_sar(im)
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = np.load(file)
        im = normalize_sar(im)
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1))
    return data

def median_filter(image, size, color_type=None):
    radius = int((size-1)/2)
    if(len(image.shape)==2):
        width, height = image.shape[0], image.shape[1]
        result = np.zeros((width-2*radius, height-2*radius))
        for i in range(width-2*radius):
            for j in range(height-2*radius):
                result[i,j] = np.median(image[i:i+size,j:j+size])
    return np.array(result, dtype='uint8')


def store_data_and_plot(im, threshold):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    return im

def save_sar_images(denoised, noisy, imagename, save_dir, original_file_path):
    threshold = np.mean(noisy)+3*np.std(noisy)
    denoisedfilename = save_dir + "/denoised_" + imagename
    im_denoised = store_data_and_plot(denoised, threshold)
    im_denoised = median_filter(im_denoised,6)
    save_gdal_image(im_denoised,denoisedfilename,'uint8',original_file_path)


