import struct
import numpy as np
from dataclasses import dataclass
from scipy import sparse

@dataclass
class GRAL:
    """
    All constants related to the GRAL domain.

    Parameters
    ----------
    nx : int
        Number of cells in x-direction.
    ny : int
        Number of cells in y-direction.
    nz : int
        Number of cells in z-direction.
    dx
        Cell width in x-direction in [m].
    dy 
        Cell width in y-direction in [m].
    xmin 
        x-coordinate of the left-lower (south-west) corner of the GRAMM domain in 
        Gauß-Krueger coordinates.
    ymin
        y-coordinate of the left-lower (south-west) corner of the GRAMM domain in 
        Gauß-Krueger coordinates.
    """
    nx: int = 1227
    ny: int = 1232
    nz: int = 400
    dx: float = 10.
    dy: float = 10.
    xmin: float = 3471259.
    ymin: float = 5468979.


def read_con_file(path, GRAL=GRAL):
    with path.open("rb") as f:
        data = f.read()
    # Check if empty
    if len(data) <= 4:
        return -1

    header = struct.unpack("i", data[:4])
    data_list = list(struct.iter_unpack("iif", data[4:]))
    datarr = np.array(data_list)
    con = np.zeros((GRAL.nx, GRAL.ny))

    x = datarr[:, 0]
    y = datarr[:, 1]

    idx = ((x - GRAL.xmin) / GRAL.dx).astype(int)
    idy = ((y - GRAL.ymin) / GRAL.dy).astype(int)

    con[idx, idy] = datarr[:, 2]

    return con

def con_file_as_sparse_matrix(path, GRAL=GRAL):
    con = read_con_file(path, GRAL)
    return sparse.csr_matrix(con)