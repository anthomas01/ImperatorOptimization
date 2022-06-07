# Import libraries
import sys
import numpy as np
import stl
from stl import mesh
from scipy.spatial import Delaunay
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

#######################################################

##Parameters - converted from inches to meters
iToM = 0.0254

diameter = 6*iToM
noseconeLength = 36*iToM
bodyLength = 128*iToM
rootChord = 12*iToM
tipChord = 6*iToM
maxSpan = 6*iToM
minSpan = 3*iToM
finDistance = noseconeLength+bodyLength-rootChord-1*iToM #Distance from top of nosecone to base of fins
finCanLength = rootChord+2*iToM
finThickness = 0.375*iToM
edgeWidth = 0.25*iToM
boattailLength = 4*iToM
nozzleLength = 2*iToM
nozzleRadius = 1.5*iToM

args = str(sys.argv)
param = np.array([float(args(1)),float(args(2)),float(args(3))]) # [-100,100] | [-100,100] | [1.75*iToM, 0.5*diameter*iToM]

numRad = 40

stlPath = "./constant/triSurface/"

#######################################################

##Generate Nosecone

numPts = 10000
numL = int(numPts/numRad)

#Curvature Equation
def noseconeEq(x):
    return 0.5*(diameter/np.sqrt(noseconeLength))*np.sqrt(x)

#Nosecone Parameterization
u=np.linspace(0,noseconeLength,numL)*np.linspace(0,1,numL)
v=np.linspace(0,2*np.pi,numRad)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

noseconeVertices = np.zeros([len(u),3])
noseconeVertices[:,0]=u
noseconeVertices[:,1]=noseconeEq(u)*np.cos(v)
noseconeVertices[:,2]=noseconeEq(u)*np.sin(v)

#Generate Triangles
noseconeTri = Delaunay(np.array([u,v]).T) #Triangulate
points3D=np.vstack((noseconeVertices[:,0], noseconeVertices[:,1], noseconeVertices[:,2])).T
tri_vertices=map(lambda index: points3D[index], noseconeTri.simplices) #Map Vertices
noseconeTriangles = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices]) #Convert to numpy arr

numTri = len(noseconeTriangles)
noseconeData = np.zeros(numTri, dtype=mesh.Mesh.dtype)

#Save STL Data
for i in range(numTri):
    #Write point data
    noseconeData["vectors"][i,0]=noseconeTriangles[i,0,:]
    noseconeData["vectors"][i,1]=noseconeTriangles[i,1,:]
    noseconeData["vectors"][i,2]=noseconeTriangles[i,2,:]

noseconeMesh = mesh.Mesh(noseconeData)
noseconeMesh.save(stlPath+"nosecone.stl", mode=stl.Mode.ASCII)

#######################################################

##Generate Body Tube

numPts = 7600
numL = int(numPts/numRad)

#Body Tube Parameterization
u=np.linspace(noseconeLength,noseconeLength+bodyLength-finCanLength,numL)
v=np.linspace(0,2*np.pi,numRad)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

bodyVertices = np.zeros([len(u),3])
bodyVertices[:,0]=u
bodyVertices[:,1]=(diameter/2)*np.cos(v)
bodyVertices[:,2]=(diameter/2)*np.sin(v)

#Generate Triangles
bodyTri = Delaunay(np.array([u,v]).T) #Triangulate
points3D=np.vstack((bodyVertices[:,0], bodyVertices[:,1], bodyVertices[:,2])).T
tri_vertices=map(lambda index: points3D[index], bodyTri.simplices) #Map Vertices
bodyTriangles = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices]) #Convert to numpy arr

numTri = len(bodyTriangles)
bodyData = np.zeros(numTri, dtype=mesh.Mesh.dtype)

#Save STL Data
for i in range(numTri):
    #Write point data
    bodyData["vectors"][i,0]=bodyTriangles[i,0,:]
    bodyData["vectors"][i,1]=bodyTriangles[i,1,:]
    bodyData["vectors"][i,2]=bodyTriangles[i,2,:]

bodyMesh = mesh.Mesh(bodyData)
bodyMesh.save(stlPath+"bodyTube.stl", mode=stl.Mode.ASCII)

#######################################################

##Generate Fins

numPts = 10000-numPts
numPts1 = 2400
numChord = 50
numSpan = int(numPts/numChord)
numL = int(numPts1/numRad)
theta = np.arcsin(finThickness/diameter)
add = (diameter/2)*(1-np.cos(theta))

#Equations
def finSpanEq(x):
    #Clipped Delta Geometry
    if x<=(rootChord-tipChord):
        span = minSpan + x*(maxSpan-minSpan)/(rootChord-tipChord)
    else:
        span = maxSpan
    return span

def finEdgeEq(x,y):
    #Triangular Edge
    span = finSpanEq(x)
    if x<edgeWidth:
        z = 0.5*finThickness*x/edgeWidth
        if y>(minSpan):
            z = z*(span-y)/(span-minSpan)
    elif x<(rootChord-edgeWidth):
        if y>(span-edgeWidth):
            z = 0.5*finThickness*(span-y)/edgeWidth
        else:
            z = finThickness/2
    else:
        z = 0.5*finThickness*(rootChord-x)/edgeWidth
        if y>(maxSpan-edgeWidth):
            z = (0.5*finThickness*(span-y)/edgeWidth)*(rootChord-x)/edgeWidth
    return z

#Fin Parameterization
u=np.linspace(finDistance,rootChord+finDistance,numChord)
v=np.linspace(-add,maxSpan,numSpan)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

u1=np.linspace(noseconeLength+bodyLength-finCanLength,noseconeLength+bodyLength,numL)
v1=np.linspace(0,2*np.pi,numRad)
u1,v1=np.meshgrid(u1,v1)
u1=u1.flatten()
v1=v1.flatten()

for i in range(len(u)):
    span = finSpanEq(u[i]-finDistance)
    if v[i]>=span:
        v[i] = span

z = np.zeros_like(u)
for i in range(len(z)):
    z[i] = finEdgeEq(u[i]-finDistance,v[i])

finVertices1 = np.zeros([len(u),3])
finVertices1[:,0] = u
finVertices1[:,1] = v+0.5*diameter
finVertices1[:,2] = z

spacing = np.radians(120)
finVertices2A = np.copy(finVertices1)
finVertices2A[:,1] = finVertices1[:,1]*np.cos(spacing) - finVertices1[:,2]*np.sin(spacing)
finVertices2A[:,2] = finVertices1[:,1]*np.sin(spacing) + finVertices1[:,2]*np.cos(spacing)

finVertices2B = np.copy(finVertices1)
finVertices2B[:,1] = finVertices1[:,1]*np.cos(spacing) + finVertices1[:,2]*np.sin(spacing)
finVertices2B[:,2] = finVertices1[:,1]*np.sin(spacing) - finVertices1[:,2]*np.cos(spacing)

finVertices3A = np.copy(finVertices1)
finVertices3A[:,1] = finVertices1[:,1]*np.cos(spacing) - finVertices1[:,2]*np.sin(-spacing)
finVertices3A[:,2] = finVertices1[:,1]*np.sin(-spacing) + finVertices1[:,2]*np.cos(spacing)

finVertices3B = np.copy(finVertices1)
finVertices3B[:,1] = finVertices1[:,1]*np.cos(spacing) - finVertices1[:,2]*np.sin(spacing)
finVertices3B[:,2] = -finVertices1[:,1]*np.sin(spacing) - finVertices1[:,2]*np.cos(spacing)

finCanVertices = np.zeros([len(u1),3])
finCanVertices[:,0] = u1
finCanVertices[:,1] = 0.5*diameter*np.cos(v1)
finCanVertices[:,2] = 0.5*diameter*np.sin(v1)

#Generate Triangles
finTri1A = Delaunay(np.array([u,v]).T) #Triangulate
finTri1B = Delaunay(np.array([u,v]).T) #Triangulate
finTri2A = Delaunay(np.array([u,v]).T) #Triangulate
finTri2B = Delaunay(np.array([u,v]).T) #Triangulate
finTri3A = Delaunay(np.array([u,v]).T) #Triangulate
finTri3B = Delaunay(np.array([u,v]).T) #Triangulate
finCanTri = Delaunay(np.array([u1,v1]).T)
points3D1A=np.vstack((finVertices1[:,0], finVertices1[:,1], finVertices1[:,2])).T
points3D1B=np.vstack((finVertices1[:,0], finVertices1[:,1], -finVertices1[:,2])).T
points3D2A=np.vstack((finVertices2A[:,0], finVertices2A[:,1], finVertices2A[:,2])).T
points3D2B=np.vstack((finVertices2B[:,0], finVertices2B[:,1], finVertices2B[:,2])).T
points3D3A=np.vstack((finVertices3A[:,0], finVertices3A[:,1], finVertices3A[:,2])).T
points3D3B=np.vstack((finVertices3B[:,0], finVertices3B[:,1], finVertices3B[:,2])).T
points3DCan=np.vstack((finCanVertices[:,0], finCanVertices[:,1], finCanVertices[:,2])).T
tri_vertices1A=map(lambda index: points3D1A[index], finTri1A.simplices) #Map Vertices
tri_vertices1B=map(lambda index: points3D1B[index], finTri1B.simplices) #Map Vertices
tri_vertices2A=map(lambda index: points3D2A[index], finTri2A.simplices) #Map Vertices
tri_vertices2B=map(lambda index: points3D2B[index], finTri2B.simplices) #Map Vertices
tri_vertices3A=map(lambda index: points3D3A[index], finTri3A.simplices) #Map Vertices
tri_vertices3B=map(lambda index: points3D3B[index], finTri3B.simplices) #Map Vertices
tri_verticesCan=map(lambda index: points3DCan[index], finCanTri.simplices) #Map Vertices
finTriangles1A = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices1A]) #Convert to numpy arr
finTriangles1B = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices1B]) #Convert to numpy arr
finTriangles2A = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices2A]) #Convert to numpy arr
finTriangles2B = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices2B]) #Convert to numpy arr
finTriangles3A = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices3A]) #Convert to numpy arr
finTriangles3B = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices3B]) #Convert to numpy arr
finCanTriangles = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_verticesCan]) #Convert to numpy arr

numTri = len(finTriangles1A)
finData1A = np.zeros(numTri, dtype=mesh.Mesh.dtype)
finData1B = np.zeros(numTri, dtype=mesh.Mesh.dtype)
finData2A = np.zeros(numTri, dtype=mesh.Mesh.dtype)
finData2B = np.zeros(numTri, dtype=mesh.Mesh.dtype)
finData3A = np.zeros(numTri, dtype=mesh.Mesh.dtype)
finData3B = np.zeros(numTri, dtype=mesh.Mesh.dtype)
numTri1 = len(finCanTriangles)
finCanData = np.zeros(numTri1, dtype=mesh.Mesh.dtype)

#Save STL Data
for i in range(numTri):
    #Write point data
    finData1A["vectors"][i,0]=finTriangles1A[i,0,:]
    finData1A["vectors"][i,1]=finTriangles1A[i,1,:]
    finData1A["vectors"][i,2]=finTriangles1A[i,2,:]
    finData1B["vectors"][i,0]=finTriangles1B[i,0,:]
    finData1B["vectors"][i,1]=finTriangles1B[i,1,:]
    finData1B["vectors"][i,2]=finTriangles1B[i,2,:]
    finData2A["vectors"][i,0]=finTriangles2A[i,0,:]
    finData2A["vectors"][i,1]=finTriangles2A[i,1,:]
    finData2A["vectors"][i,2]=finTriangles2A[i,2,:]
    finData2B["vectors"][i,0]=finTriangles2B[i,0,:]
    finData2B["vectors"][i,1]=finTriangles2B[i,1,:]
    finData2B["vectors"][i,2]=finTriangles2B[i,2,:]
    finData3A["vectors"][i,0]=finTriangles3A[i,0,:]
    finData3A["vectors"][i,1]=finTriangles3A[i,1,:]
    finData3A["vectors"][i,2]=finTriangles3A[i,2,:]
    finData3B["vectors"][i,0]=finTriangles3B[i,0,:]
    finData3B["vectors"][i,1]=finTriangles3B[i,1,:]
    finData3B["vectors"][i,2]=finTriangles3B[i,2,:]
    finCanData["vectors"][i,0]=finCanTriangles[i,0,:]
    finCanData["vectors"][i,1]=finCanTriangles[i,1,:]
    finCanData["vectors"][i,2]=finCanTriangles[i,2,:]

finMesh1A = mesh.Mesh(finData1A)
finMesh1B = mesh.Mesh(finData1B)
finMesh2A = mesh.Mesh(finData2A)
finMesh2B = mesh.Mesh(finData2B)
finMesh3A = mesh.Mesh(finData3A)
finMesh3B = mesh.Mesh(finData3B)
finCanMesh = mesh.Mesh(finCanData)
finMeshCombo = mesh.Mesh(np.concatenate([finMesh1A.data,finMesh1B.data,finMesh2A.data,finMesh2B.data,finMesh3A.data,finMesh3B.data,finCanMesh.data]))
finMeshCombo.save(stlPath+"finCan.stl", mode=stl.Mode.ASCII)

#######################################################

##Generate Boattail

numPts = 5000
numL = int(numPts/numRad)

#Curvature Equation
def boattailEq(x,a,b,r2):
    c = (r2-0.5*diameter-b*np.power(boattailLength,3)-a*np.power(boattailLength,4))/(np.power(boattailLength,2))
    return a*np.power(x,4)+b*np.power(x,3)+c*np.power(x,2)+0.5*diameter

#Boattail Parameterization
u=np.linspace(0,boattailLength,numL)
v=np.linspace(0,2*np.pi,numRad)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

u1=np.linspace(0,param[2],5)
v1=np.linspace(0,2*np.pi,numRad)
u1,v1=np.meshgrid(u1,v1)
u1=u1.flatten()
v1=v1.flatten()

x1 = np.ones_like(u1)*(noseconeLength+bodyLength+boattailLength)

boattailVertices = np.zeros([len(u),3])
boattailVertices[:,0]=u+noseconeLength+bodyLength
boattailVertices[:,1]=boattailEq(u,param[0],param[1],param[2])*np.cos(v)
boattailVertices[:,2]=boattailEq(u,param[0],param[1],param[2])*np.sin(v)

boattailVertices1 = np.zeros([len(u1),3])
boattailVertices1[:,0]=x1
boattailVertices1[:,1]=u1*np.cos(v1)
boattailVertices1[:,2]=u1*np.sin(v1)

#Generate Triangles
boattailTri1 = Delaunay(np.array([u,v]).T) #Triangulate
boattailTri2 = Delaunay(np.array([u1,v1]).T) #Triangulate
points3D1=np.vstack((boattailVertices[:,0], boattailVertices[:,1], boattailVertices[:,2])).T
points3D2=np.vstack((boattailVertices1[:,0], boattailVertices1[:,1], boattailVertices1[:,2])).T
tri_vertices1=map(lambda index: points3D1[index], boattailTri1.simplices) #Map Vertices
tri_vertices2=map(lambda index: points3D2[index], boattailTri2.simplices) #Map Vertices
boattailTriangles1 = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices1]) #Convert to numpy arr
boattailTriangles2 = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices2]) #Convert to numpy arr

boattailData1 = np.zeros(len(boattailTriangles1), dtype=mesh.Mesh.dtype)
boattailData2 = np.zeros(len(boattailTriangles2), dtype=mesh.Mesh.dtype)

#Save STL Data
for i in range(len(boattailTriangles1)):
    #Write point data
    boattailData1["vectors"][i,0]=boattailTriangles1[i,0,:]
    boattailData1["vectors"][i,1]=boattailTriangles1[i,1,:]
    boattailData1["vectors"][i,2]=boattailTriangles1[i,2,:]
for i in range(len(boattailTriangles2)):
    #Write point data
    boattailData2["vectors"][i,0]=boattailTriangles2[i,0,:]
    boattailData2["vectors"][i,1]=boattailTriangles2[i,1,:]
    boattailData2["vectors"][i,2]=boattailTriangles2[i,2,:]

boattailMesh1 = mesh.Mesh(boattailData1)
boattailMesh2 = mesh.Mesh(boattailData2)
boattailMesh = mesh.Mesh(np.concatenate([boattailMesh1.data,boattailMesh2.data]))
boattailMesh.save(stlPath+"boattail.stl", mode=stl.Mode.ASCII)

#######################################################

##Generate Nozzle

numPts = 5000
numL = int(numPts/numRad)

#Boattail Parameterization
u=np.linspace(0,nozzleLength,numL)
v=np.linspace(0,2*np.pi,numRad)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

u1=np.linspace(0,nozzleRadius,5)
v1=np.linspace(0,2*np.pi,numRad)
u1,v1=np.meshgrid(u1,v1)
u1=u1.flatten()
v1=v1.flatten()

x1 = np.ones_like(u1)*(noseconeLength+bodyLength+boattailLength+nozzleLength)

nozzleVertices = np.zeros([len(u),3])
nozzleVertices[:,0]=u+noseconeLength+bodyLength+boattailLength
nozzleVertices[:,1]=nozzleRadius*np.cos(v)
nozzleVertices[:,2]=nozzleRadius*np.sin(v)

nozzleVertices1 = np.zeros([len(u1),3])
nozzleVertices1[:,0]=x1
nozzleVertices1[:,1]=u1*np.cos(v1)
nozzleVertices1[:,2]=u1*np.sin(v1)

#Generate Triangles
nozzleTri1 = Delaunay(np.array([u,v]).T) #Triangulate
nozzleTri2 = Delaunay(np.array([u1,v1]).T) #Triangulate
points3D1=np.vstack((nozzleVertices[:,0], nozzleVertices[:,1], nozzleVertices[:,2])).T
points3D2=np.vstack((nozzleVertices1[:,0], nozzleVertices1[:,1], nozzleVertices1[:,2])).T
tri_vertices1=map(lambda index: points3D1[index], nozzleTri1.simplices) #Map Vertices
tri_vertices2=map(lambda index: points3D2[index], nozzleTri2.simplices) #Map Vertices
nozzleTriangles1 = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices1]) #Convert to numpy arr
nozzleTriangles2 = np.array([[t[0,:],t[1,:],t[2,:]] for t in tri_vertices2]) #Convert to numpy arr

nozzleData1 = np.zeros(len(nozzleTriangles1), dtype=mesh.Mesh.dtype)
nozzleData2 = np.zeros(len(nozzleTriangles2), dtype=mesh.Mesh.dtype)

#Save STL Data
for i in range(len(nozzleTriangles1)):
    #Write point data
    nozzleData1["vectors"][i,0]=nozzleTriangles1[i,0,:]
    nozzleData1["vectors"][i,1]=nozzleTriangles1[i,1,:]
    nozzleData1["vectors"][i,2]=nozzleTriangles1[i,2,:]
for i in range(len(nozzleTriangles2)):
    #Write point data
    nozzleData2["vectors"][i,0]=nozzleTriangles2[i,0,:]
    nozzleData2["vectors"][i,1]=nozzleTriangles2[i,1,:]
    nozzleData2["vectors"][i,2]=nozzleTriangles2[i,2,:]

nozzleMesh1 = mesh.Mesh(nozzleData1)
nozzleMesh2 = mesh.Mesh(nozzleData2)
nozzleMesh = mesh.Mesh(np.concatenate([nozzleMesh1.data,nozzleMesh2.data]))
nozzleMesh.save(stlPath+"nozzle.stl", mode=stl.Mode.ASCII)