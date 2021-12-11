#!/usr/bin/env python
"""
DAFoam run script for the running analysis on Imperator
"""

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM
import numpy as np

# =============================================================================
# Input Parameters
# =============================================================================

# global parameters
U0 = 300.0
p0 = 101325.0
nuTilda0 = 4.47e-5
T0 = 300.0
alpha0 = 0.0
rho0 = 1.225  # density for normalizing CD and CL
#H0 = 4.318 # height
A0 = 0.0217634 #cross sectional area
#COM = [x,0.0,0.0] #center of mass (x from nosecone tip)

def calcUAndDir(UMag, alpha1):
    dragDir = [float(np.cos(alpha1 * np.pi / 180)), float(np.sin(alpha1 * np.pi / 180)), 0.0]
    liftDir = [float(-np.sin(alpha1 * np.pi / 180)), float(np.cos(alpha1 * np.pi / 180)), 0.0]
    inletU = [float(UMag * np.cos(alpha1 * np.pi / 180)), float(UMag * np.sin(alpha1 * np.pi / 180)), 0.0]
    return inletU, dragDir, liftDir

inletU, dragDir, liftDir = calcUAndDir(U0, alpha0)

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["boattailStraight"],
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
        #         "patches": ["nosecone","bodyTube","finCan","boattailStraight","nozzle"],
        #         "axis": [1.0, 0.0, 0.0],
        #         "forceAxis": [0.0, 1.0, 0.0],
        #         "center": COM,
        #         "scale": 1.0 / H0,
        #         "addToAdjoint": True,
        #     }
        # },
    },
    "checkMeshThreshold": {"maxAspectRatio": 5000.0},
}

DASolver = PYDAFOAM(options=daOptions, comm=MPI.COMM_WORLD)
DASolver()
funcs = {}
evalFuncs = ["CD", "CL"]
DASolver.evalFunctions(funcs, evalFuncs)
