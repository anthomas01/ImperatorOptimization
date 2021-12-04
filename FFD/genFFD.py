import numpy as np
import sys


def writeFFDFile(fileName, nBlocks, nx, ny, nz, points):
    """
    Take in a set of points and write the plot 3dFile
    """

    f = open(fileName, "w")

    f.write("%d\n" % nBlocks)
    for i in range(nBlocks):
        f.write("%d %d %d " % (nx[i], ny[i], nz[i]))
    # end
    f.write("\n")
    for block in range(nBlocks):
        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write("%f " % points[block][i, j, k, 0])
                # end
            # end
        # end
        f.write("\n")

        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write("%f " % points[block][i, j, k, 1])
                # end
            # end
        # end
        f.write("\n")

        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write("%f " % points[block][i, j, k, 2])
                # end
            # end
        # end
    # end
    f.close()
    return


def returnBlockPoints(corners, nx, ny, nz):
    """
    corners needs to be 8 x 3
    """
    points = np.zeros([nx, ny, nz, 3])

    # points 1 - 4 are the iMin face
    # points 5 - 8 are the iMax face

    for idim in range(3):
        edge1 = np.linspace(corners[0][idim], corners[4][idim], nx)
        edge2 = np.linspace(corners[1][idim], corners[5][idim], nx)
        edge3 = np.linspace(corners[2][idim], corners[6][idim], nx)
        edge4 = np.linspace(corners[3][idim], corners[7][idim], nx)

        for i in range(nx):
            edge5 = np.linspace(edge1[i], edge3[i], ny)
            edge6 = np.linspace(edge2[i], edge4[i], ny)
            for j in range(ny):
                edge7 = np.linspace(edge5[j], edge6[j], nz)
                points[i, j, :, idim] = edge7
            # end
        # end
    # end

    return points


################ FFD ##############
nBlocks = 5

nx = [2,2,2,10,2]
ny = [2,2,2,2,2]
nz = [2,2,2,2,2]

corners = np.zeros([nBlocks, 8, 3])

#nosecone
corners[0, 0, :] = [-0.01, -0.0862, -0.0862]
corners[0, 1, :] = [-0.01, -0.0862, 0.0862]
corners[0, 2, :] = [-0.01, 0.0862, -0.0862]
corners[0, 3, :] = [-0.01, 0.0862, 0.0862]
corners[0, 4, :] = [0.92, -0.0862, -0.0862]
corners[0, 5, :] = [0.92, -0.0862, 0.0862]
corners[0, 6, :] = [0.92, 0.0862, -0.0862]
corners[0, 7, :] = [0.92, 0.0862, 0.0862]

#bodytube
corners[1, 0, :] = [0.91, -0.0862, -0.0862]
corners[1, 1, :] = [0.91, -0.0862, 0.0862]
corners[1, 2, :] = [0.91, 0.0862, -0.0862]
corners[1, 3, :] = [0.91, 0.0862, 0.0862]
corners[1, 4, :] = [3.76, -0.0862, -0.0862]
corners[1, 5, :] = [3.76, -0.0862, 0.0862]
corners[1, 6, :] = [3.76, 0.0862, -0.0862]
corners[1, 7, :] = [3.76, 0.0862, 0.0862]

#fincan
corners[2, 0, :] = [3.75, -0.225, -0.225]
corners[2, 1, :] = [3.75, -0.225, 0.225]
corners[2, 2, :] = [3.75, 0.225, -0.225]
corners[2, 3, :] = [3.75, 0.225, 0.225]
corners[2, 4, :] = [4.17, -0.225, -0.225]
corners[2, 5, :] = [4.17, -0.225, 0.225]
corners[2, 6, :] = [4.17, 0.225, -0.225]
corners[2, 7, :] = [4.17, 0.225, 0.225]

#boattail
corners[3, 0, :] = [4.165, -0.8, -0.8]
corners[3, 1, :] = [4.165, -0.8, 0.8]
corners[3, 2, :] = [4.165, 0.8, -0.8]
corners[3, 3, :] = [4.165, 0.8, 0.8]
corners[3, 4, :] = [4.275, -0.8, -0.8]
corners[3, 5, :] = [4.275, -0.8, 0.8]
corners[3, 6, :] = [4.275, 0.8, -0.8]
corners[3, 7, :] = [4.275, 0.8, 0.8]

#exhaust
corners[4, 0, :] = [4.27, -0.0381, -0.0381]
corners[4, 1, :] = [4.27, -0.0381, 0.0381]
corners[4, 2, :] = [4.27, 0.0381, -0.0381]
corners[4, 3, :] = [4.27, 0.0381, 0.0381]
corners[4, 4, :] = [4.32, -0.0381, -0.0381]
corners[4, 5, :] = [4.32, -0.0381, 0.0381]
corners[4, 6, :] = [4.32, 0.0381, -0.0381]
corners[4, 7, :] = [4.32, 0.0381, 0.0381]

points = []
for block in range(nBlocks):
    points.append(returnBlockPoints(corners[block], nx[block], ny[block], nz[block]))

# print points
fileName = "rocketFFD.xyz"
writeFFDFile(fileName, nBlocks, nx, ny, nz, points)

