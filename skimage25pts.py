import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import PiecewiseAffineTransform
from scipy.ndimage import map_coordinates
import time

#----------------------------------------
#ask for the image and read out the rows and cols
#----------------------------------------

image = input("load image "'\n')
image_path = image+'.png'
image = io.imread(image_path)

rows, cols = image.shape[1], image.shape[0]

#----------------------------------------
#define the nodes
#----------------------------------------

control_points = [[0              ,0],
                [round(cols/4)  ,0],
                [round(cols/2)  ,0],
                [round((3*cols)/4),0],
                [cols-1         ,0],
                
                
                [0              ,round(rows/4)],
                [round(cols/4)  ,round(rows/4)],
                [round(cols/2)  ,round(rows/4)],
                [round((3*cols)/4),round(rows/4)],
                [cols-1         ,round(rows/4)],
   
                
                [0              ,round(rows/2)],
                [round(cols/4)  ,round(rows/2)],
                [round(cols/2)  ,round(rows/2)],
                [round((3*cols)/4),round(rows/2)],
                [cols-1         ,round(rows/2)],
                
                [0              ,round((3*rows)/4)],
                [round(cols/4)  ,round((3*rows)/4)],
                [round(cols/2)  ,round((3*rows)/4)],
                [round((3*cols)/4),round((3*rows)/4)],
                [cols-1         ,round((3*rows)/4)],
                
                [0              ,rows-1],
                [round(cols/4)  ,rows-1],
                [round(cols/2)  ,rows-1],
                [round((3*cols)/4),rows-1],
                [cols-1         ,rows-1]]

t= time.time()

#----------------------------------------
#show the points and the original image
#----------------------------------------

def grid(image, control_points):
    plt.imshow(image)

    for x, y in control_points:
        plt.scatter(x, y, color='k')
    plt.show()

#----------------------------------------
#deform the image 
#----------------------------------------

def deform_image_with_grid(image, control_points, deformation_points):
    tform = PiecewiseAffineTransform()
    tform.estimate(control_points, deformation_points)

#----------------------------------------
#creates the mesh out of the cols and rows
#----------------------------------------

    cols, rows = image.shape[1], image.shape[0]
    x, y = np.meshgrid(np.arange(cols),np.arange(rows))
    
#----------------------------------------
#np.vstack().T creates one array out of the x,y arrays 
#[[x1,y1]
# [x2,y2]]
#----------------------------------------
    
    coords = np.vstack((x.ravel(),y.ravel())).T
    warped_coords = tform(coords)
    warped_x, warped_y = warped_coords[:, 0], warped_coords[:, 1]
    warped_image=np.zeros_like(image)

#----------------------------------------
#to get the result image in color 
#----------------------------------------  
    
    for channel in range(image.shape[2]):
        channel_image = image[:, :, channel]
        warped_channel = map_coordinates(channel_image, [warped_y, warped_x], order=1, mode='reflect')
        warped_channel = warped_channel.reshape((rows, cols))
        warped_image[:, :, channel] = warped_channel


    return warped_image

#----------------------------------------
#define the deformations 
#----------------------------------------

deformation_points = control_points.copy()
deformation_points[12] = (deformation_points[12][0] + 20, deformation_points[12][1] + 10)
    
grid(image, control_points)
   
warped_image = deform_image_with_grid(image, control_points, deformation_points)

#----------------------------------------
# Plot the deformed image
#----------------------------------------

plt.imshow(warped_image)
plt.show()
elapsed= time.time()-t
print(elapsed)