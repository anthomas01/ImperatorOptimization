#!/usr/bin/env python
"""
DAFoam run script for the Imperator Rocket using SMT and numpy stl
"""

# =============================================================================
# Imports
# =============================================================================
from cmath import pi
import os
import argparse
from mpi4py import MPI
from dafoam import PYDAFOAM#, optFuncs
import numpy as np
from smt.applications.ego import EGO 
from smt.sampling_methods import LHS #Sampling

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="type of run to do", type=str, default="smt")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# global parameters
U0 = 300
p0 = 101325.0
nuTilda0 = 4.47e-5
T0 = 300.0
alpha0 = 0.0
rho0 = 1.225  # density for normalizing CD and CL
H0 = 4.318 # height
A0 = 0.0217634 #cross sectional area
#D0 = 0.1524 #body caliper
#COM = [x,0.0,0.0] #center of mass (x from nosecone tip)

def calcUAndDir(UMag, alpha1):
    dragDir = [float(np.cos(alpha1 * np.pi / 180)), float(np.sin(alpha1 * np.pi / 180)), 0.0]
    liftDir = [float(-np.sin(alpha1 * np.pi / 180)), float(np.cos(alpha1 * np.pi / 180)), 0.0]
    inletU = [float(UMag * np.cos(alpha1 * np.pi / 180)), float(UMag * np.sin(alpha1 * np.pi / 180)), 0.0]
    return inletU, dragDir, liftDir

inletU, dragDir, liftDir = calcUAndDir(U0, alpha0)

# Set the parameters for optimization
daOptions = {
    "designSurfaces": [""],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-6,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["front","back","bot","top","left","right"], "value": inletU},
        "p0": {"variable": "p", "patches": ["front","back","bot","top","left","right"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["front","back","bot","top","left","right"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["front","back","bot","top","left","right"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    # variable bounds for compressible flow conditions
    "primalVarBounds": {
        "UMax": 1000.0,
        "UMin": -1000.0,
        "pMax": 500000.0,
        "pMin": 20000.0,
        "eMax": 500000.0,
        "eMin": 100000.0,
        "rhoMax": 5.0,
        "rhoMin": 0.2,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["nosecone","bodyTube","finCan","boattailStraight","nozzle"],
                "directionMode": "fixedDirection",
                "direction": dragDir,
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["nosecone","bodyTube","finCan","boattailStraight","nozzle"],
                "directionMode": "fixedDirection",
                "direction": liftDir,
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        # "COP": {
        #     "part1": {
        #         "type": "centerOfPressure",
        #         "source": "patchToFace",
        #         "patches": ["R5-Body-Solid-1","R5-Fin-Solid-1","R5-Fin-Solid-2","R5-Fin-Solid-3","R5-NoseCone-Solid-1"],
        #         "axis": [1.0, 0.0, 0.0],
        #         "forceAxis": [0.0, 1.0, 0.0],
        #         "center": COM,
        #         "scale": 1.0 / H0,
        #         "addToAdjoint": True,
        #     }
        # },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "gmresMaxIters": 2000, "gmresRestart": 2000},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjPCLag": 1,
    "designVar": {},
    "writeDeformedFFDs": True,
    "checkMeshThreshold":{"maxNonOrth": 75.0,"maxSkewness": 8.0,"maxAspectRatio": 2000.0}
}

# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Surrogate Modeling Function
# =============================================================================
def smt(x):
    #Print Input
    print("input=",x)
    inputLength = len(x[:,0])
    outputArr = np.ones([inputLength,1])
    numInput = 0
    for input in x:
        #Set new design variables
        c1=input[0]
        c2=input[1]
        r2=input[2]

        #Clear old mesh files?

        #Generate STL's
        os.system("python rocketStl.py " + c1 + " " + c2 + " " + r2)

        #Snappy Hex Mesh
        os.system("./preProcessing.sh")

        #Run CFD and get output
        DASolver()
        output={}
        DASolver.evalFunctions(output, evalFuncs=evalFuncs)

        #Print Output
        print("output=",output)
        outputArr[numInput,0] = output["CD"]
        numInput+=1

    return outputArr

# =============================================================================
# Task
# =============================================================================
if args.task == "smt":
    #Limits for shape variables
    xlimits=np.array([[-100,100],[-100,100],[1.75*0.0254,3*0.0254]])
    
    criterion='SBO' #'EI' or 'SBO' or 'LCB'

    #number of points in the initial DOE
    ndoe = 5 #(at least number of design variables + 1)

    #number of iterations with EGO 
    n_iter = 1

    #Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
    sampling = LHS(xlimits=xlimits, random_state=1)
    xdoe = sampling(ndoe)

    #EGO call
    ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

    c1_opt, c2_opt, r2_opt, ind_best, c1_data, c2_data, r2_data = ego.optimize(fun=smt)
    print('Optimized design variables and the minimized objective: ', c1_opt, c2_opt, ' obtained using EGO criterion = ', criterion)

else:
    print("task arg not found!")
    exit(0)
