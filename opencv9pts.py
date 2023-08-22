import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
#----------------------------------------
#ask for an image which should get read and deformed
#ask which point should be deformed and the deformations in x and y direction
#----------------------------------------
picture=input("load image\n")
image=plt.imread(picture+(".png"))
choosepoint=input("which point you want to deform: ")

if choosepoint==("top left"):
     x1=input("deformation X-axis :")
     y1=input("deformation Y-axis :")
else:
     x1=0
     y1=0
if choosepoint==("top center"):
     x2=input("deformation X-axis :")
     y2=input("deformation Y-axis :")
else:
     x2=0
     y2=0     
if choosepoint==("top right"):
     x3=input("deformation X-axis :")
     y3=input("deformation Y-axis :")
else:
     x3=0
     y3=0
if choosepoint==("middle left"):
     x4=input("deformation X-axis :")
     y4=input("deformation Y-axis :")     
else:
     x4=0
     y4=0
if choosepoint==("middle center"):
     x5=input("deformation X-axis :")
     y5=input("deformation Y-axis :")     
else:
     x5=0
     y5=0     
if choosepoint==("middle right"):
     x6=input("deformation X-axis :")
     y6=input("deformation Y-axis :")     
else:
     x6=0
     y6=0     
if choosepoint==("bottom left"):
     x7=input("deformation X-axis :")
     y7=input("deformation Y-axis :")     
else:
     x7=0
     y7=0     
if choosepoint==("bottom center"):
     x8=input("deformation X-axis :")
     y8=input("deformation Y-axis :")     
else:
     x8=0
     y8=0     
if choosepoint==("bottom right"):
     x9=input("deformation X-axis :")
     y9=input("deformation Y-axis :")     
else:
     x9=0
     y9=0
#----------------------------------------
#create the deformed image out of the deformed mesh
#----------------------------------------
nRow, nCol=image.shape[:2]
#print(image.shape[:2],end="")
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
grid=np.array(([[0            ,0],
                [round(nCol/2),0],
                [nCol-1       ,0],
                
                [0            ,round(nRow/2)],
                [round(nCol/2),round(nRow/2)],
                [nCol-1       ,round(nRow/2)],
                
                [0            ,nRow-1],
                [round(nCol/2),nRow-1],
                [nCol-1       ,nRow-1]])) 
           
#print(grid)

#----------------------------------------
#the deformations of each node
#----------------------------------------

defG=np.array(([[int(x1),int(y1)],
                [int(x2),int(y2)],
                [int(x3),int(y3)],
                [int(x4),int(y4)],
                [int(x5),int(y5)],
                [int(x6),int(y6)],
                [int(x7),int(y7)],
                [int(x8),int(y8)],
                [int(x9),int(y9)]]))            

#----------------------------------------
#the element connectivity
#----------------------------------------

elms = np.array(([[1,4,5,2],
                  [2,5,6,3],
                  [4,7,8,5],
                  [5,8,9,6]]))         


t= time.time()    
#----------------------------------------
#get the coordinates of the deformed mesh
#----------------------------------------        
for e in elms:
    print("deform mesh...\n", end="")
    cx = grid[e-1][:,0]
    cy = grid[e-1][:,1]
    dx = defG[e-1][:,0]
    dy = defG[e-1][:,1]
    
    mesh = generate(image,mesh,round(np.max(cy)-np.min(cy))+5,round(np.max(cx)-np.min(cx))+5,cx,cy,dx,dy)

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

