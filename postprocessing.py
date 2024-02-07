from PIL import Image
import os
import numpy as np
from osgeo import gdal

filepath = 'result_testA'
filelist = os.listdir(filepath)

def median_filter(image, size, color_type=None):
    radius = int((size-1)/2)
    if(len(image.shape)==3):
        width, height = image.shape[0], image.shape[1]
        result = np.zeros((width-2*radius, height-2*radius))
        for i in range(width-2*radius):
            for j in range(height-2*radius):
                result[i,j] = np.median(image[i:i+size,j:j+size])
   
    return np.array(result, dtype='uint8')

def tiff_open(imgPath):

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

if __name__ == '__main__':
    for file in filelist:

        path = os.path.join(filepath,file)
        print(path)
        image = tiff_open(path)
        image_median = median_filter(image,7)
        output_file = os.path.join('testA/',file)
        save_gdal_image(image_median,output_file,'uint8',path)
