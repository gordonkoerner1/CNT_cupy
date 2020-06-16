## This is a python port of Dr. Matt Maschmann's Matlab CNT growth simulation code. This script simulates an iterative CNT
#  growth process by introducing new material at each time step and then solving for the finite element interactions (stress
#  and strain of the finite elements are modeled as a beam) between steps. NOTE: THE GLOBAL VARIABLES DEFINED ON LINES 5 AND
#  6 IN MATLAB ARE DEFINED AS LOCAL ONES HERE FOR SIMPLICITY (classes and mods to be introduced to the python script later)
from pathlib import Path
import math
import numpy as np
import time
import scipy
from scipy.sparse import *
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy import *
import random as rand
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import imageio
import os
import numba
from numba import vectorize, jit, cuda
from numba import *
import cupy as cp

#initialize these variables first - the sizes of initialized arrays are functions of these 3 parameters
span=10e-6 #modeling regime (meters) - want 10 CNT per micron - this way they interact during growth
numberBeams=250 #adjust to fit troubleshooting needs, original is 250 - need to make sure they interact during growth
steps=100  #number of time steps

##List of all arrays that grow in matlab
meanvdw=cp.zeros((steps,1));meanvdwsmall=cp.zeros((steps,1))
medianvdw=cp.zeros((steps,1));maxvdw=cp.zeros((steps,1))
minvdw=cp.zeros((steps,1));origin=cp.zeros(numberBeams)
ang=cp.zeros(numberBeams);rate=cp.zeros(numberBeams)
ro=cp.zeros(steps*numberBeams);ri=cp.zeros(numberBeams)
nodeCoordinates=cp.zeros(((steps+2)*numberBeams,2))
nucleationSite=cp.zeros((numberBeams,2))
growthNode=cp.zeros((numberBeams,2))
elementNodes=cp.zeros(((steps+2)*numberBeams,2),dtype=int)
L=cp.zeros((steps+1)*numberBeams);A=cp.zeros((steps+1)*numberBeams)
E=cp.zeros((steps+1)*numberBeams);I=cp.zeros((steps+1)*numberBeams)
C=cp.zeros(steps*numberBeams);S=cp.zeros(steps*numberBeams)

E_0=cp.ones(numberBeams)*1E12;rout=5E-9; rin=0.7*rout # Modulus ; outer radius; inner radius
g_mod=1e11; a_cc=math.pi*(rout**2-rin**2) #Shear modulus; cross-sectional area
I_def=pi/4*(rout**4-rin**4)
gap=50e-9; gap2=gap**2; A_0=cp.ones(numberBeams)*a_cc #note, this 
            #does not support unique CNT radii

fname=Path("/Users/Gordo/Desktop/CNT/Images/Junk/") # NOTE: Currently, files are saved to this directory

title='Test' #These lines set up image names
name=title
title2=name+'_Float3'

avgRate=60e-9 # Average growth per time step (meters)
rate_stdev=4 #growth rate standard deviation
ang_stdev=3 #growth angle standard deviation


## INPUT SETTINGS - should we use boolean data types in python?
computeVDW=0 # compute vdW forces? 0=  no
ContinuousPlot=0 # 0=plotting off.  1=plotting on
PeriodicBoundary=1 # 0=off, 1=on
beamType=0 # 0=Euler Beam ; 1=Timoshenko Beam
totalCompress=0 # if using compression
compressiveLoad=0 # if using compression
element=numberBeams # sets the number of elements = number of CNTs
nodeCount=2*numberBeams # initial number of evaluation nodes

## Define nucleate_Uniform function
def nucleate_Uniform(numberBeams,span,avgRate,rout,rin,rate_stdev,ang_stdev,E):
    
    for o in range(numberBeams):
        origin[o]=span/(numberBeams)*(o+1/2);##+span/(numberBeams-1)*rand()
    
   
    
    rand.seed() #Re-seed random number generator
    
    ## Here we nucleate the rates of the beams

    for ii in range(numberBeams): ## 1/3 of the beams will grow more slowly
        mu_ang=cp.array(math.pi/2)
        Sigma_ang=cp.atleast_2d((ang_stdev*math.pi/180)**2)
        R_ang=cp.linalg.cholesky(Sigma_ang)
        ang[ii]=(mu_ang+cp.random.normal()*R_ang).flatten()[0]

        rand.seed()
        mu_rate=avgRate
        Sigma_rate=cp.atleast_2d((rate_stdev/100*avgRate)**2)
        R_rate=cp.linalg.cholesky(Sigma_rate)
        rate[ii]=(mu_rate+cp.random.normal()*R_rate).flatten()[0]
    
        ro[ii]=rout
        ri[ii]=rin
        I[ii]= (math.pi/4)*(rout**4-rin**4)

        
    E=E #recyle the initial E - an array defining the 

    nodeCount=0
    element=0
    
    for num in range(numberBeams,2*numberBeams): ## Assign bottom coordinates to first generation of CNTs
        nodeCount=nodeCount+1
        nodeCoordinates[num,0]=origin[num-numberBeams]  ##defines the x-location of nucleation.
        nodeCoordinates[num,1]=0
    
        nucleationSite[num-numberBeams,0]=nodeCoordinates[num,0] ##xx(num)
        nucleationSite[num-numberBeams,1]=nodeCoordinates[num,1] ##yy(num)
    
    for num in range(numberBeams): ##Setting position of CNT free ends (top nodes)
        nodeCount=nodeCount+1
        nodeCoordinates[num,0]=nodeCoordinates[num+numberBeams,0]+math.cos(ang[num])*rate[num]
        nodeCoordinates[num,1]=nodeCoordinates[num+numberBeams,1]+math.sin(ang[num])*rate[num]
       
        growthNode[num,0]=nodeCoordinates[num,0]
        growthNode[num,1]=nodeCoordinates[num,1]
        
        elementNodes[element,0]=element
        elementNodes[element,1]=nodeCount-1
        element=element+1
        

    return [elementNodes,ang,rate,nodeCoordinates,nodeCount,element,nucleationSite,growthNode,ro,ri,E,I]

## Define FindCloseNodes_Range2b function
def FindCloseNodesSparse(nodeCoordinates,nodeCount,numberBeams,ro):
    dist=cp.zeros((nodeCount,nodeCount))
    gap=2*ro+10e-9
    
    if rout < 15e-9:
        gap=50e-9
    
    #find the distance matrix for the spatial coordinates of the nodes
    
    dist=(cp.transpose(cp.tile(nodeCoordinates,(nodeCount,1,1)),(1,0,2))-cp.tile(nodeCoordinates,(nodeCount,1,1)))
    dist_sq=cp.sum(dist**2,axis=2)  #distance matrix (between all nodes)
    dist_sq=cp.triu(dist_sq)           #distance matrix (between all nodes)
    #dist_sq[dist_sq<0]=0
    dist_sq[dist_sq>gap2]=0
    
    [test3,test4]=cp.nonzero(dist_sq) #Returns the node numbers of CNTs within a given distance
    test3=cp.transpose(cp.atleast_2d(test3))
    test4=cp.transpose(cp.atleast_2d(test4))
    closeNodes=cp.concatenate((test3,test4),axis=1)
    #[II,JJ]=ind2sub(size(sep),contact) #converts from index to column,row notation
    #closeNodes=[II,JJ]
    if closeNodes.size==0:
        closeNodes=cp.atleast_2d(cp.asarray([1,0]))
        
    return [closeNodes]


## Define StiffnessPartial function
def StiffnessPartial(E,A,I,L,C,S,G,element,elementNodes,GDof,beamType,totalDOFs):
    I=I.flatten()
    ## In the 2D stiffness matrix, there are 6 unique entries. They are
    #  labeled as w1... w6 here.

    alpha=1 # this has never been activated...
    if beamType==0:
        epsilon=0
    else:
        epsilon=(12*alpha/g_mod)*cp.multiply(E,cp.divide(I,A,out=cp.zeros_like(A),where=A!=0))

        
    #Ni and Nj are indices that take element nodes and assigns each node an
    #appropriate location within the stiffness matrix, K

    Ni=cp.atleast_2d(3*(elementNodes[:A.shape[0],0])-1)
    Nj=cp.atleast_2d(3*(elementNodes[:A.shape[0],1])-1)
    
    #Each element will have 36 entries into a stiffness matrix. ii_local and
    #jj_local are the indices associated with each of the 36 entries. This step
    #was taken for vectorization
    ii_local=cp.concatenate((Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3),axis=0)
    jj_local=cp.concatenate((Ni+1,Ni+1,Ni+1,Ni+1,Ni+1,Ni+1,Ni+2,Ni+2,Ni+2,Ni+2,Ni+2,Ni+2,Ni+3,Ni+3,Ni+3,Ni+3,Ni+3,Ni+3,Nj+1,Nj+1,Nj+1,Nj+1,Nj+1,Nj+1,Nj+2,Nj+2,Nj+2,Nj+2,Nj+2,Nj+2,Nj+3,Nj+3,Nj+3,Nj+3,Nj+3,Nj+3),axis=0)
    
    #multiply arrays first
    E_L=cp.divide(E,L,out=cp.zeros_like(L))
    A_C_C=cp.multiply(A,cp.square(C))
    A_S_S=cp.multiply(A,cp.square(S))
    I_S_S_12=12*cp.multiply(I,cp.square(S))
    I_C_C_12=12*cp.multiply(I,cp.square(C))
    L_L=cp.square(L)
    eps_1=cp.ones(L.shape)+epsilon
    L_L_eps1=cp.multiply(L_L,eps_1)
    S_C=cp.multiply(S,C)
    I_S=cp.multiply(I,S)
    I_C=cp.multiply(I,C)
    I_L_L_eps1=cp.divide(I,L_L_eps1,out=cp.zeros_like(L_L_eps1))
    L_eps1=cp.multiply(L,eps_1)
    # The six unique stiffness matrix entries are computed below
    w1=cp.multiply(E_L,(A_C_C + cp.divide(I_S_S_12,L_L_eps1,out=cp.zeros_like(L_L_eps1))))
    w2=cp.multiply(E_L,(A_S_S + cp.divide(I_C_C_12,L_L_eps1,out=cp.zeros_like(L_L_eps1))))
    w3=cp.multiply(E_L,(cp.multiply((A-12*I_L_L_eps1),S_C)))
    w4=cp.multiply(E_L,(6.*cp.divide(I_S,L_eps1,out=cp.zeros_like(L_L_eps1))))
    w5=cp.multiply(E_L,(6.*cp.divide(I_C,L_eps1,out=cp.zeros_like(L_L_eps1))))
    
    ## Creates a vectorized set of K matrix entries, Kg. Each element will have 
    # 36 entries. The code below create a vector of 36 entries for each element
    # in the system
    Kg=cp.zeros((E.shape[0],36))
    Kg[0:E.shape[0],0]=w1
    Kg[0:E.shape[0],1]=w3
    Kg[0:E.shape[0],2]=-w4
    Kg[0:E.shape[0],3]=-w1
    Kg[0:E.shape[0],4]=-w3
    Kg[0:E.shape[0],5]=-w4
    Kg[0:E.shape[0],6]=w3
    Kg[0:E.shape[0],7]=w2
    Kg[0:E.shape[0],8]=w5
    Kg[0:E.shape[0],9]=-w3
    Kg[0:E.shape[0],10]=-w2
    Kg[0:E.shape[0],11]=w5
    Kg[0:E.shape[0],12]=-w4
    Kg[0:E.shape[0],13]=w5
    Kg[0:E.shape[0],14]=4*cp.multiply(I,E_L)
    Kg[0:E.shape[0],15]=w4
    Kg[0:E.shape[0],16]=-w5
    Kg[0:E.shape[0],17]=2*cp.multiply(I,E_L)
    Kg[0:E.shape[0],18]=-w1
    Kg[0:E.shape[0],19]=-w3
    Kg[0:E.shape[0],20]=w4
    Kg[0:E.shape[0],21]=w1
    Kg[0:E.shape[0],22]=w3
    Kg[0:E.shape[0],23]=w4
    Kg[0:E.shape[0],24]=-w3
    Kg[0:E.shape[0],25]=-w2
    Kg[0:E.shape[0],26]=-w5
    Kg[0:E.shape[0],27]=w3
    Kg[0:E.shape[0],28]=w2
    Kg[0:E.shape[0],29]=-w5
    Kg[0:E.shape[0],30]=-w4
    Kg[0:E.shape[0],31]=w5
    Kg[0:E.shape[0],32]=2*cp.multiply(I,E_L)
    Kg[0:E.shape[0],33]=w4
    Kg[0:E.shape[0],34]=-w5
    Kg[0:E.shape[0],35]=4*cp.multiply(I,E_L)
    Kg=cp.transpose(Kg)
    
    # Creates a sparse matrix based upon the indices and values created above
    Kg_flat=Kg.flatten()
    ii_flat=ii_local.flatten()
    jj_flat=jj_local.flatten()
    K=cp.sparse.coo_matrix((Kg_flat,(ii_flat,jj_flat)),shape=(totalDOFs,totalDOFs)).tocsr()
    # the dimensions are: [row, column, value, sparse matrix n dim, sparse matrix m dim], COO format
    return K


## Define ConnectionStiffness_Partial
def ConnectionStiffness_Partial(E,rout,CC,SS,sizeClose,closeNodes,GDof,beamType,totalDOFs):
#erased unused quantities E, Ac, Ic, LL, G, EI

    ## In the 2D stiffness matrix, there are 6 unique entries. They are
    #  labeled as w1... w6 here.
    vdwk=cp.zeros(CC.shape[0])

    alpha=1 # this has never been activated...
    if beamType==0:
        epsilon=0
    else:
        epsilon=0
        print('this has unintended effect')

    #vdwk=E
    if cp.any(ro==5e-09):
        vdwk=273*cp.ones(vdwk.shape) #Spring stiffness of bar element between CNTs
    elif cp.any(ro==12.5e-09):
        vdwk=430*cp.ones(vdwk.shape)
    else:
        print('check CNT Radii')

    #Ni and Nj are indices that take element nodes and assigns each node an
    #appropriate location within the stiffness matrix, K

    Ni=cp.atleast_2d(3*(closeNodes[:,0])-1)
    Nj=cp.atleast_2d(3*(closeNodes[:,1])-1)        
        
    #Each element will have 36 entries into a stiffness matrix. ii_local and
    #jj_local are the indices associated with each of the 36 entries. This step
    #was taken for vectorization
    ii_local=cp.concatenate((Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3,Ni+1,Ni+2,Ni+3,Nj+1,Nj+2,Nj+3),axis=0)
    jj_local=cp.concatenate((Ni+1,Ni+1,Ni+1,Ni+1,Ni+1,Ni+1,Ni+2,Ni+2,Ni+2,Ni+2,Ni+2,Ni+2,Ni+3,Ni+3,Ni+3,Ni+3,Ni+3,Ni+3,Nj+1,Nj+1,Nj+1,Nj+1,Nj+1,Nj+1,Nj+2,Nj+2,Nj+2,Nj+2,Nj+2,Nj+2,Nj+3,Nj+3,Nj+3,Nj+3,Nj+3,Nj+3),axis=0)

    w1=cp.multiply(vdwk,cp.square(CC))
    w2=cp.multiply(vdwk,cp.multiply(SS,CC))    
    w3=cp.multiply(vdwk,cp.square(SS))

    ## Creates a vectorized set of K matrix entries, Kg. Each element will have 
    # 36 entries. The code below create a vector of 36 entries for each element
    # in the system
    Kg=cp.zeros((sizeClose,36))
    Kg[0:sizeClose,0]=w1
    Kg[0:sizeClose,1]=w2
    Kg[0:sizeClose,2]=0
    
    Kg[0:sizeClose,3]=-w1
    Kg[0:sizeClose,4]=-w2
    Kg[0:sizeClose,5]=0
    
    Kg[0:sizeClose,6]=w2
    Kg[0:sizeClose,7]=w3
    Kg[0:sizeClose,8]=0
    
    Kg[0:sizeClose,9]=-w2
    Kg[0:sizeClose,10]=-w3
    Kg[0:sizeClose,11]=0
    
    Kg[0:sizeClose,12]=0
    Kg[0:sizeClose,13]=0
    Kg[0:sizeClose,14]=0
    Kg[0:sizeClose,15]=0
    Kg[0:sizeClose,16]=0
    Kg[0:sizeClose,17]=0
    
    Kg[0:sizeClose,18]=-w1
    Kg[0:sizeClose,19]=-w2
    Kg[0:sizeClose,20]=0
    
    Kg[0:sizeClose,21]=w1
    Kg[0:sizeClose,22]=w2
    Kg[0:sizeClose,23]=0
    
    Kg[0:sizeClose,24]=-w2
    Kg[0:sizeClose,25]=-w3
    Kg[0:sizeClose,26]=0
    
    Kg[0:sizeClose,27]=w2
    Kg[0:sizeClose,28]=w3
    Kg[0:sizeClose,29]=0
    
    Kg[0:sizeClose,30]=0
    Kg[0:sizeClose,31]=0
    Kg[0:sizeClose,32]=0
    Kg[0:sizeClose,33]=0
    Kg[0:sizeClose,34]=0
    Kg[0:sizeClose,35]=0
    
    Kg=cp.transpose(Kg)
    
    # Creates a sparse matrix based upon the indices and values created above
    Kg_flat=Kg.flatten()
    ii_flat=ii_local.flatten()
    jj_flat=jj_local.flatten()
    ii_flat[ii_flat<0]=0
    jj_flat[jj_flat<0]=0
    AA=cp.sparse.coo_matrix((Kg_flat,(ii_flat,jj_flat)),shape=(totalDOFs,totalDOFs)).tocsr()
    
    return [AA,vdwk]

## Define CNTPlotFast for a fast plotting algorithm
def CNTPlotFast(fname,nodeCoordinates,t,title):
    nodeCoordinates=cp.asnumpy(nodeCoordinates)
    fig=plt.figure(figsize=(5.0,5.0),dpi=800)
    plt.title('')
    plt.xlabel('Substrate Position (µm)')
    plt.ylabel('Forest Height (µm)')
    plt.xlim((0,span*1e6))
    plt.ylim((0,span*1e6))
    plt.plot(nodeCoordinates[:nodeCount,0]*1e6,nodeCoordinates[:nodeCount,1]*1e6,marker='.',ms=1,lw=0,color='black');
    fname=Path("/Users/Gordo/Desktop/CNT/Images/Junk/"+str(t)) # NOTE: Currently, files are saved to this directory
    plt.savefig(fname,dpi=800)
    
    
########################################################################
##/ nucleating CNTs with distributed properties according to inputs####/
########################################################################

[elementNodes,ang,rate,nodeCoordinates,nodeCount,element,nucleationSite,growthNode,ro,ri,E,I]=nucleate_Uniform(numberBeams,span,avgRate,rout,rin,rate_stdev,ang_stdev,E)

ss=steps+2
angle=cp.transpose(cp.tile((cp.transpose(ang)),ss)) #steps=1350  #number of time stepsangle of each node

E=cp.transpose(E) #reshape modulus

sizeClose=0 #number of closeNodes. See below
closeNodes=cp.zeros((1,2),dtype=int) #closeNodes are paired lists of nodes (node numbers) in contact
closeNodesOLD=cp.zeros((1,2),dtype=int);closeNodesNew=[0,0]
firstNodeCompare=0
removedNodes1=[0,0];removedNodes2=[0,0]

###########################################################################
FF=(steps+1)*numberBeams*3 ### I dont think this is used. It was intended
Uaccumulated=cp.zeros((FF,1)) ### to track the total translation of each node
Ucurrent=cp.zeros((FF,1))
###########################################################################

totalNodes=numberBeams*(steps+2)  # total number of nodes at the end of simulation
totalDOFs=numberBeams*(steps+2)*3 # DOF = degrees of freedom at the end of simulation

########################################################################
        ####/ initialize t=0 element material variables  ####/
########################################################################

e_0=cp.arange(0,numberBeams) #index of initial beam
L[e_0]=rate.flatten() #should we track rate for each element at each time step?
A[e_0]=A_0
E[e_0]=E_0
I[e_0]=I[e_0]
ro[e_0]=ro[e_0]

#%%
###########################################################################
##################   BEGIN GROWTH STEPS  ##################################sa
timer1=time.time()
for t in range(steps):
    closeNodes=closeNodesOLD; ## initializes nodes in contact to null 
    timer2=time.time()
    dx=cp.zeros(element)
    dy=cp.zeros(element)
    if t>0: #assigning properties to newly-grown elements
        e=((element-numberBeams)*cp.ones(cp.arange(0,numberBeams).size)+cp.arange(0,numberBeams)).astype(int)
        e_e=cp.arange(0,e.shape[0])
        L[e]=rate.flatten() #should we track rate for each element at each time step?
        A[e]=A_0
        E[e]=E_0
        I[e]=I[e_e]
        ro[e]=ro[e_e]
        
    GDof=3*(nodeCount); # Global Degrees of Freedom

    [closeNodes]=FindCloseNodesSparse(nodeCoordinates[:nodeCount,:],nodeCount,numberBeams,ro[0])
    
    U=cp.zeros((GDof,1)); force=cp.zeros((GDof,1)) ##initializing variables
    i=0; ii=cp.zeros((36,1)); iii=cp.zeros((36,1)); j=0; jj=cp.zeros((36,1)); jjj=cp.zeros((36,1))
    vdwEquilForce=cp.zeros((GDof, 1))
    
    dx[:element]=(nodeCoordinates[elementNodes[:nodeCount,1],0]-nodeCoordinates[elementNodes[:nodeCount,0],0])[:element]
    dy[:element]=(nodeCoordinates[elementNodes[:nodeCount,1],1]-nodeCoordinates[elementNodes[:nodeCount,0],1])[:element]
    
    #if PeriodicBoundary>0;         # For periodic boundary
    #conditions, if an element moves 6-1-20 Matt - can add later
    #    crossedl=find(xa>span/2);  # across a boundary, it shows up on the other side of 
    #    xa(crossedl)=xa(crossedl)-span; # the simulation domain 
    #    crossedr=find( xa <(-span/2 ) );
    #    xa(crossedr)=xa(crossedr)+span;
    #end

    Ldef=cp.sqrt(cp.square(dx)+cp.square(dy))
    C=cp.divide(dx,Ldef,out=cp.zeros_like(Ldef)) ###,where=Ldef!=0 removed
    S=cp.divide(dy,Ldef,out=cp.zeros_like(Ldef))  #Computes Cosine and Sine for each element
    K=StiffnessPartial(E[:element],A[:element],I[:element],L[:element],C,S,g_mod,element,elementNodes[0:int(GDof),:],GDof,beamType,totalDOFs)
        
    ##### ATTRACIVE FORCES BY VAN DER WAALS ATTRACTION #############
    ################################################################
    
    vdwk=273 ####Should change this command line
    
    if closeNodes[0,:].all !=[0,0] and closeNodesOLD[0,:].all != [0,0]: #This isn't sufficient because closeNodes can get dropped
                
        closeNodesCombined=cp.vstack((closeNodes,closeNodesOLD)) #This ensures that nodes were not accidentally missed by closeNodes call
        closeNodesCombined=cp.asnumpy(closeNodesCombined)
        closeNodes_nump=np.unique(closeNodesCombined,axis=0) #Updated list of closeNodes for time t
        closeNodes=cp.array(closeNodes) ###NOTE: PREVIOUS LINE IS A POTENTIAL BOTTLENECK
        if (closeNodes[0,:]==cp.array([0,0])).all(): #this is formatting for below
            closeNodes=cp.delete(closeNodes,0,0);
            
        droppedNodesAdded=1
        print('droppedNodesAdded = '+str(droppedNodesAdded))
            
        ######################################################################
    
        # FINDING NEW CLOSENODES AND ESTABLISHING THEIR COORDINATES OF FIRST CONTACT         

        new_pairs=closeNodes #The first time close nodes are present
        vdwInfo=0

        ################ Computes the AA stiffness matrix  ###################
        sizeClose=closeNodes.shape[0]

        xa=nodeCoordinates[closeNodes[:,1],0]-nodeCoordinates[closeNodes[:,0],0]
        ya=nodeCoordinates[closeNodes[:,1],1]-nodeCoordinates[closeNodes[:,0],1]

        LL=cp.sqrt(cp.square(xa)+cp.square(ya)); CC=cp.divide(xa,LL); SS=cp.divide(ya,LL)
        [AA,vdwk]=ConnectionStiffness_Partial(E,ro,CC,SS,sizeClose,closeNodes,GDof,beamType,totalDOFs)

        ######################################################################
        ######################################################################             

        K=K+AA # Adds vdW force to the global stiffness matrix
        
    ## Setting boundary conditions to solve for displacement ##################

    ##########################################################################
    
    pdof=cp.arange(numberBeams)
    
    #The lines below are boundary conditions that set the location for the
    #top node of the bottom element. U is displacement.
    U[GDof-3*numberBeams+3*pdof+0]=cp.transpose(cp.atleast_2d(cp.multiply(rate,cp.cos(ang))))
    U[GDof-3*numberBeams+3*pdof+1]=cp.transpose(cp.atleast_2d(cp.multiply(rate,cp.sin(ang))))
    U[GDof-3*numberBeams+3*pdof+2]=0
    
    alldof=cp.arange((GDof-3*numberBeams),GDof) #% alldof is all fixed degrees of freedom
    prescribedDof=cp.transpose(alldof) # Re-alinging alldof into a vector

    ##########################################################################
 
    activeDof=np.setdiff1d(np.arange(GDof),cp.asnumpy(prescribedDof)) #All free nodes in the system
    
    K_slice_row=K[activeDof[0]:activeDof[-1]+1] #can only slice the rows of a CSR matrix 
    K_slice_row=cp.sparse.csc_matrix(K_slice_row) #convert the K sparse matrix to CSC (to facilitate column slicing)
    K_slice_all=K_slice_row[:,prescribedDof[0]:prescribedDof[-1]+1]
    minus_K=(-1)*K_slice_all    
    
    bc_U=cp.sparse.csr_matrix(U[prescribedDof])
    K_U=minus_K*bc_U
    K_U=(K_U.toarray()).flatten() #reshape
    K_square=cp.sparse.csr_matrix(cp.sparse.csc_matrix(K[activeDof[0]:activeDof[-1]+1])[:,activeDof[0]:activeDof[-1]+1])
    displacements=cp.zeros(K_U.shape)
    displacements=cp.sparse.linalg.lsqr(K_square,K_U)[0]
    displacements[cp.isnan(displacements)]=0
    U[activeDof[0]:activeDof[-1]+1]=cp.transpose(cp.atleast_2d(displacements)) #Defines the displacement of the free nodes
    i_i=cp.arange(nodeCount)  #-numberBeams; ##Translates all nodes
    nodeCoordinates[i_i,0]=nodeCoordinates[i_i,0]+U[3*i_i].flatten()
    nodeCoordinates[i_i,1]=nodeCoordinates[i_i,1]+U[3*i_i+1].flatten()
    angle[i_i]=angle[i_i]+U[3*i_i+2].flatten()
    
    cp.asnumpy(prescribedDof[0].flatten())[0]
    
    #if PeriodicBoundary>0; 6-1-20 matt - add later
    #         crossedl=find(nodeCoordinates(:,1)<0);% moves across left boundary
    #         nodeCoordinates(crossedl,1)=nodeCoordinates(crossedl,1)+span;
    #         crossedr=find(nodeCoordinates(:,1)>span);
    #         nodeCoordinates(crossedr,1)=nodeCoordinates(crossedr,1)-span;
    #end

    currentCount=nodeCount
    
    ## Dividing the bottom-most elements ##
    for kk in range(currentCount,currentCount+numberBeams):
        nodeCoordinates[kk,0]=nucleationSite[kk-currentCount,0]
        nodeCoordinates[kk,1]=nucleationSite[kk-currentCount,1]
        elementNodes[element,0]=element
        elementNodes[element,1]=nodeCount
        element+=1
        nodeCount+=1

    print('t='+str(t))
    print('Elapsed time is '+str(time.time()-timer2)+' seconds.')
    if (t+1)%25==0: #plots every 100 steps
        print('plot me')
        CNTPlotFast(fname,nodeCoordinates[:nodeCount],t,title) #for python, only send the written nodeCoordinates

time1=time.time()-timer1
print('Total Elapsed time is '+str(time.time()-timer1)+' seconds.')