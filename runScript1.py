#!/usr/bin/env python
"""
DAFoam run script for the Imperator Rocket at transonic speed
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
from pygeo import *
from pyspline import *
from idwarp import *
from pyoptsparse import Optimization, OPT
import numpy as np

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
# which optimizer to use. Options are: slsqp (default), snopt, or ipopt
parser.add_argument("--opt", help="optimizer to use", type=str, default="ipopt")
# which task to run. Options are: opt (default), run, testSensShape, or solveCL
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
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
    "designSurfaces": ["nosecone","bodyTube","finCan","boattailStraight","nozzle"],
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
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjPCLag": 1,
    "designVar": {},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "openfoam",
    "useRotations": False,
    # point and normal for the symmetry plane
    #"symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
}

# options for optimizers
if args.opt == "snopt":
    optOptions = {
        "Major feasibility tolerance": 1.0e-7,
        "Major optimality tolerance": 1.0e-7,
        "Minor feasibility tolerance": 1.0e-7,
        "Verify level": -1,
        "Function precision": 1.0e-7,
        "Major iterations limit": 50,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.opt == "ipopt":
    optOptions = {
        "tol": 1.0e-7,
        "constr_viol_tol": 1.0e-7,
        "max_iter": 150,
        "print_level": 8,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.opt == "slsqp":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 50,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("opt arg not valid!")
    exit(0)


# =============================================================================
# Design variable setup
# =============================================================================
DVGeo = DVGeometry("./FFD/rocketFFD.xyz")

# angle of attack
#def alpha(val, geo):
#    aoa = val[0] * np.pi / 180.0
#    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
#    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["front","back","bot","top","left","right"], "value": inletU}})
#    DASolver.updateDAOption()

# select points
boattailVol = 3
pts = DVGeo.getLocalIndex(boattailVol)
indexList = pts[1:, :, :].flatten()
bPS = geo_utils.PointSelect("list", indexList)
#bP1 = geo_utils.PointSelect("list", pts[1:,0,0].flatten())
#bP2 = geo_utils.PointSelect("list", pts[1:,0,1].flatten())
#bP3 = geo_utils.PointSelect("list", pts[1:,1,0].flatten())
#bP4 = geo_utils.PointSelect("list", pts[1:,1,1].flatten())
# shape
DVGeo.addGeoDVLocal("boattail_shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=bPS)
#DVGeo.addLocalSectionDV('boattail_trans1', secIndex='j', lower=-1, upper=1, axis=2, pointSelect=bP1)
#DVGeo.addLocalSectionDV('boattail_trans2', secIndex='k', lower=-1, upper=1, axis=2, pointSelect=bP2)
#DVGeo.addLocalSectionDV('boattail_trans3', secIndex='j', lower=-1, upper=1, axis=2, pointSelect=bP3)
#DVGeo.addLocalSectionDV('boattail_trans4', secIndex='k', lower=-1, upper=1, axis=2, pointSelect=bP4)
daOptions["designVar"]["boattail_shapey"] = {"designVarType": "FFD"}
#daOptions["designVar"]["boattail_trans1"] = {"designVarType": "FFD"}
#daOptions["designVar"]["boattail_trans2"] = {"designVarType": "FFD"}
#daOptions["designVar"]["boattail_trans3"] = {"designVarType": "FFD"}
#daOptions["designVar"]["boattail_trans4"] = {"designVarType": "FFD"}
# alpha
#DVGeo.addGeoDVGlobal("alpha", [alpha0], alpha, lower=0.0, upper=0.0, scale=1.0)
#daOptions["designVar"]["alpha"] = {"designVarType": "AOA", "patches": ["front","back","bot","top","left","right"], "flowAxis": "x", "normalAxis": "y"}

# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.addFamilyGroup(DASolver.getOption("designSurfaceFamily"), DASolver.getOption("designSurfaces"))
DASolver.printFamilyList()
DASolver.setMesh(mesh)
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Constraint setup
# =============================================================================
DVCon = DVConstraints()
#DVCon.setDVGeo(DVGeo)
DVCon.setSurface(DASolver.getTriangulatedMeshSurface(groupName=DASolver.getOption("designSurfaceFamily")))

#Boattail Constraints
leList = [[4.12, -0.075, 0], [4.2, 0.075, 0]]
teList = [[4.25, -0.075, 0], [4.25, -0.075, 0]]
# volume constraint
DVCon.addVolumeConstraint(leList, teList, nSpan=2, nChord=8, lower=0.5, upper=1.0, scaled=True)
# thickness constraint
DVCon.addThicknessConstraints2D(leList, teList, nSpan=2, nChord=8, lower=0.5, upper=1.0, scaled=True)
#circularity constraints
DVCon.addCircularityConstraint([4.1656,0,0], [1,0,0], 0.0762, [0,1,0], 0, 360, nPts=20, lower=1.0, upper=1.0, scaled=True)
DVCon.addCircularityConstraint([4.191,0,0], [1,0,0], 0.0762, [0,1,0], 0, 360, nPts=20, lower=0.5, upper=1.0, scaled=True)
DVCon.addCircularityConstraint([4.2164,0,0], [1,0,0], 0.0762, [0,1,0], 0, 360, nPts=20, lower=0.5, upper=1.0, scaled=True)
DVCon.addCircularityConstraint([4.2418,0,0], [1,0,0], 0.0762, [0,1,0], 0, 360, nPts=20, lower=0.5, upper=1.0, scaled=True)
DVCon.addCircularityConstraint([4.2672,0,0], [1,0,0], 0.0762, [0,1,0], 0, 360, nPts=20, lower=0.5, upper=1.0, scaled=True)

# Le/Te constraints
#DVCon.addLeTeConstraints(0, "iLow")
#DVCon.addLeTeConstraints(0, "iHigh")

# =============================================================================
# Initialize optFuncs for optimization
# =============================================================================
optFuncs.DASolver = DASolver
#optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# =============================================================================
# Task
# =============================================================================
if args.task == "opt":

    #alpha4CLTarget = optFuncs.solveCL(CL_target, "alpha", "CL")
    #alpha([alpha0], None)

    optProb = Optimization("opt", objFun=optFuncs.calcObjFuncValues, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj("CD", scale=1)
    # Add physical constraints
    #optProb.addCon("CL", lower=0.0, upper=0.0, scale=1)

    if gcomm.rank == 0:
        print(optProb)

    DASolver.runColoring()

    opt = OPT(args.opt, options=optOptions)
    histFile = "./%s_hist.hst" % args.opt
    sol = opt(optProb, sens=optFuncs.calcObjFuncSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

if args.task == "runPrimal":

    optFuncs.runPrimal()

elif args.task == "runAdjoint":

    optFuncs.runAdjoint()

elif args.task == "verifySens":

    optFuncs.verifySens()

elif args.task == "testAPI":

    DASolver.setOption("primalMinResTol", 1e-2)
    DASolver.updateDAOption()
    optFuncs.runPrimal()

else:
    print("task arg not found!")
    exit(0)
