from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
import subprocess as sp
import matplotlib.pylab as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.io import imread, imsave
from skimage.transform import resize

# -----------------------------------------------------------------
#File path to run Nastran 
# -------------------------------------------------------------------
GENESIS   = "C:/Program Files/vrand/genesis18.0/bin/genesis180.exe"

#Nastran   = "C:/Program Files/MSC.Software/MSC_Nastran/2021.4/bin/nastranw.exe"
# -------------------------------------------------------------------
#create the op2 filename by slicing .bdf and add .op2
# -------------------------------------------------------------------

def getOP2Filename(bdfFile,iterID=0):

    idx = bdfFile.rindex('.')
    
    baseName = bdfFile[:idx]

    baseName = baseName +'_dsg%02d.op2'%(iterID)
    
    #show name of the op2 file
    print(baseName)    
    return baseName
getOP2Filename('platwithtwoholes.bdf',0)
    
# -------------------------------------------------------------------
#Run Nastran to create the op2 file
# -------------------------------------------------------------------
'''
def runGenesis(bdfFile):
    print('running ...')
    sp.run([GENESIS, bdfFile])
runGenesis('test.bdf')
'''
# -------------------------------------------------------------------
#read the informations of the bdf file and the op2 file.
#get the coordinates of every node and their deformations
# -------------------------------------------------------------------
def getObjectiveFn(bdfFile, loadCase, transMat=None):
    
    TIME=0
    
    modelBDF = BDF(debug=False)
    modelBDF.read_bdf(bdfFile, 'punch=True')

    op2File = getOP2Filename(bdfFile)
    modelOP2 = OP2(debug=False)
    modelOP2.read_op2(op2File)
    
    disp = modelOP2.displacements[loadCase]
    txyz = disp.data[TIME, :, :3]
    #print(txyz)

    #show the element connectivity
    #elements=modelBDF.elements
    #print(elements)
    
    iCnt = 0;
    gxyz = np.zeros( (txyz.shape[0], 3) )
    for (nid, ntype) in disp.node_gridtype:
        gxyz[iCnt] = modelBDF.nodes.get(nid).xyz
        iCnt = iCnt + 1
    
    #print(gxyz)
    #show the nodes
    
    
        
    #-----------------------------------------------------------------
    #scale the coordinates of the points and their deformations
    #scale them so that the maxima are around the original image size for a better
    #quality of the result image
    #find the maximum values of the x and y coordinates to resize the mesh image 
    #-----------------------------------------------------------------
    
    points=np.zeros((gxyz.shape[0],2))
    points[:,0]=10000*gxyz[:,0]
    points[:,1]=10000*gxyz[:,1]
    
    points_max=max(points[:,0])
    deformation_max=max(txyz[:,0])
    scale=1
    while (scale*deformation_max)<(0.4*points_max):
        scale=scale+5000
        #print(scale)

    x=10000*gxyz[:,0]+scale*txyz[:,0]
    y=10000*gxyz[:,1]+scale*txyz[:,1]
    
    new_width=max(x)
    print(new_width)
    
    new_height=max(y)
    print(new_height)
    
    plt.axis('equal')
    plt.scatter(x,y, s=20)
    plt.show()

    #-----------------------------------------------------------------
    #scale the image of the mesh to the new height and width from the coordinates
    #-----------------------------------------------------------------
    
    def resize_image(orig_img, resized_img, new_width, new_height):
        image = imread(orig_img)
        resized_image = resize(image, (new_height, new_width))
        imsave(resized_img, resized_image)
        
    img1 = '3.png'
    img2 = 'output3.png'
    
    resize_image(img1,img2, new_width, new_height)
    
    #-----------------------------------------------------------------
    #deform the resized image, 
    #-----------------------------------------------------------------       
    
    def deform_image_with_grid(image, points, deformation_points):
        tform = PiecewiseAffineTransform()
        tform.estimate(deformation_points,points)
            
        warped_image = warp(image,tform)
   
        return warped_image

    #----------------------------------------
    #define the points and deformations_points
    #scale the deformations for a better visibility
    #----------------------------------------
    
    points=np.zeros((gxyz.shape[0],2))
    points[:,0]=10000*gxyz[:,0]
    points[:,1]=10000*gxyz[:,1]
    
    deformation_points=np.zeros((gxyz.shape[0],2))
    deformation_points[:,0]=10000*gxyz[:,0]+scale*txyz[:,0]
    deformation_points[:,1]=10000*gxyz[:,1]+scale*txyz[:,1]
   
    image=imread(img2)
    
    warped_image = deform_image_with_grid(image, points, deformation_points)
    
    #-------------------------------------------
    #show the images
    #use origin='lower' so the coordinate origin is in the left corner
    #use nplflipud() to rotate the image to the right position
    #-------------------------------------------
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(np.flipud(image),origin="lower")
    ax2.imshow(np.flipud(warped_image),origin='lower')
    plt.show()

getObjectiveFn('platwithtwoholes.bdf', 1)
