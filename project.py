from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
import subprocess as sp
import matplotlib.pylab as plt
import time 

# -------------------------------------------------------------------
#File path to run Nastran 
# -------------------------------------------------------------------
#GENESIS   = "C:/Program Files/vrand/genesis18.0/bin/gfe.exe"

Nastran   = "C:/Program Files/MSC.Software/MSC_Nastran/2021.4/bin/nastranw.exe"
# -------------------------------------------------------------------
#create the op2 filename by slicing .bdf and add .op2
# -------------------------------------------------------------------
def getOP2Filename(bdfFile):

    idx = bdfFile.rindex('.')
    
    baseName = bdfFile[:idx]

    baseName = baseName + '.op2'
    
    #show name of the op2 file
    #print(baseName)    
    return baseName
getOP2Filename('test.dat')

# -------------------------------------------------------------------
#Run Nastran to create the op2 file
#time.sleep because the script doesnt wait until nastran finished.
#maybe because nastran just open a console ?
# -------------------------------------------------------------------
def runNastran(bdfFile):

    sp.run( [Nastran, bdfFile] )
    time.sleep(10)
    return
runNastran('test.dat')

# -------------------------------------------------------------------
#read the informations of the bdf file and the op2 file.
#get the coordinates of every node and their deformations
# -------------------------------------------------------------------
def getObjectiveFn(bdfFile, loadCase, transMat=None):
    
    TIME=0
    
    modelBDF = BDF(debug=False)
    modelBDF.read_bdf(bdfFile)

    op2File = getOP2Filename(bdfFile)
    modelOP2 = OP2(debug=False)
    modelOP2.read_op2(op2File)
    
    disp = modelOP2.displacements[loadCase]
    txyz = disp.data[TIME, :, :3]
    #print(txyz)

    iCnt = 0;
    gxyz = np.zeros( (txyz.shape[0], 3) )
    for (nid, ntype) in disp.node_gridtype:
        gxyz[iCnt] = modelBDF.nodes.get(nid).xyz
        iCnt = iCnt + 1
    
    #print(gxyz)
    #show the nodes
    x=gxyz[:,0]+txyz[:,0]
    y=gxyz[:,1]+txyz[:,1]
    #print(x)
    plt.scatter(x,y)
    plt.show()
    
    return 
getObjectiveFn('test.dat', 1)