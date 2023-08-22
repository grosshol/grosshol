import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

#----------------------------------------
#ask for an image which should get read and deformed
#----------------------------------------

picture=input("load image\n")
image=plt.imread(picture+(".png"))

#----------------------------------------
#create the deformed image out of the deformed mesh
#----------------------------------------
     
print("setup mesh", end="")
nRow, nCol=image.shape[:2]
print(image.shape[:2],end="")
mesh=np.zeros((nRow*nCol,2))

def deform(image,perturbed_mesh):
    h,w=image.shape[:2]#height and width of the image
    #print(h)
    #print(w)

    perturbed_mesh_x = perturbed_mesh[:,0]#all rows, column 0
    perturbed_mesh_y = perturbed_mesh[:,1]#all rows,column 1
    
    perturbed_mesh_x = perturbed_mesh_x.reshape((h,w))
    perturbed_mesh_y = perturbed_mesh_y.reshape((h,w))
    
    remapped = cv2.remap(image,perturbed_mesh_x, perturbed_mesh_y,cv2.INTER_AREA)
    return remapped

#----------------------------------------
#generate the deformed mesh
#----------------------------------------

def generate(image,mesh,nRow,nCol,cx,cy,dx,dy):
    nRow, nCol=image.shape[:2]
    delta_x=2/(nCol-1)
    delta_y=2/(nRow-1)
    
    for i in range(0,nRow):
        t=-1+i*delta_y
        n1=n2=0.25*(1-t)
        n3=n4=0.25*(1+t)
        
        for j in range(0,nCol):
            s=-1+j*delta_x
            
            sf=[n1*(1-s),n2*(1+s),n3*(1+s),n4*(1-s)]
        
            dots=np.dot([dx,dy,cx,cy],sf)
            defx=dots[0]
            defy=dots[1]
            
            oldx=dots[2]
            oldy=dots[3]
            
            rowIdx=round(oldy)*nCol+round(oldx)
            
            mesh[rowIdx,0]=oldx+defx
            mesh[rowIdx,1]=oldy+defy
            
    return mesh.astype(np.float32)
  
#----------------------------------------
#define the nodes of the grid
#----------------------------------------
    
grid=np.array(([[0              ,0],
                [round(nCol/4)  ,0],
                [round(nCol/2)  ,0],
                [round((3*nCol)/4),0],
                [nCol-1         ,0],
                
                
                [0              ,round(nRow/4)],
                [round(nCol/4)  ,round(nRow/4)],
                [round(nCol/2)  ,round(nRow/4)],
                [round((3*nCol)/4),round(nRow/4)],
                [nCol-1         ,round(nRow/4)],
   
                
                [0              ,round(nRow/2)],
                [round(nCol/4)  ,round(nRow/2)],
                [round(nCol/2)  ,round(nRow/2)],
                [round((3*nCol)/4),round(nRow/2)],
                [nCol-1         ,round(nRow/2)],
                
                [0              ,round((3*nRow)/4)],
                [round(nCol/4)  ,round((3*nRow)/4)],
                [round(nCol/2)  ,round((3*nRow)/4)],
                [round((3*nCol)/4),round((3*nRow)/4)],
                [nCol-1         ,round((3*nRow)/4)],
                
                [0              ,nRow-1],
                [round(nCol/4)  ,nRow-1],
                [round(nCol/2)  ,nRow-1],
                [round((3*nCol)/4),nRow-1],
                [nCol-1         ,nRow-1]])) 
           
#print(grid)


#----------------------------------------
#the deformations of each node
#----------------------------------------
defG=np.array(([0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [20,10],#center
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0],
               [0,0]))       
     
#----------------------------------------
#the element connectivity
#----------------------------------------

elms = np.array(([[1,6,7,2],
                  [2,7,8,3],
                  [3,8,9,4],
                  [4,9,10,5],
                  
                  [6,11,12,7],   
                  [7,12,13,8],
                  [8,13,14,9],
                  [9,14,15,10],
                  
                  [11,16,17,12],
                  [12,17,18,13],
                  [13,18,19,14],
                  [14,19,20,15],
                  
                  [16,21,22,17],
                  [17,22,23,18],
                  [18,23,24,19],
                  [19,24,25,20]
                  ]))
t= time.time()           
for e in elms:
    print("deform mesh...\n", end="")
    cx = grid[e-1][:,0]
    cy = grid[e-1][:,1]
    dx = defG[e-1][:,0]
    dy = defG[e-1][:,1]
    
    mesh = generate(image,mesh,round(np.max(cy)-np.min(cy))+5,round(np.max(cx)-np.min(cx))+5,cx,cy,dx,dy)

#----------------------------------------
#print the elapsed time for the run
#----------------------------------------
elapsed=time.time()-t
print(elapsed)

#----------------------------------------
#deform the image
#----------------------------------------

result = deform(image,mesh)

#----------------------------------------
#show the original and the deformed image
#----------------------------------------

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(image)
ax2.imshow(result)

plt.show()
cv2.imwrite("./result.png", result)
