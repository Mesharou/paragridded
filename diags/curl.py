import numpy as np
import multiprocessing as mp
from numba import jit


def keeptoplevel(var):
    var._arrays = {tile: array[-1].copy()
                   for tile, array in var._arrays.items()}


class Curl:
    def __init__(self, reader, vinterp, nthreads = 12):
        self.reader = reader
        self.nthreads = nthreads
        self.vi = vinterp

    def compute(self, var, constant_dxdy=False):
        u = var.new("u")
        v = var.new("v")
        if var.space.is_depth:
            self.vi.compute(u)
            self.vi.compute(v)
        elif var.space.is_level:
            assert var.space.level == -1
            self.reader.read(u)
            self.reader.read(v)
            keeptoplevel(u)
            keeptoplevel(v)

        f = var.new("f"); self.reader.read(f)

        if constant_dxdy:
            dx = 1e3
            tasks = iter([(u[tile], v[tile], f[tile], dx)
                         for tile in var])

            print(f"compute curl {var.varname}")

            with mp.Pool(processes=self.nthreads) as pool:
                data = pool.starmap(curl2d_overf, tasks)

        else:
            pm = var.new("pm"); self.reader.read(pm)
            pn = var.new("pn"); self.reader.read(pn)
        
            tasks = iter([(u[tile], v[tile], f[tile], pm[tile], pn[tile])
                       for tile in var])

            print(f"compute curl {var.varname}")

            with mp.Pool(processes=self.nthreads) as pool:
                data = pool.starmap(curl2d_overf_variabledx, tasks)

        var.update(data)
        var.staggering.horiz = "f"


@jit
def curl2d_overf(u, v, f, dx):
    ny, nx = u.shape
    vor = np.zeros(u.shape, dtype=u.dtype)
    for j in range(ny-1):
        for i in range(nx-1):
            cff = 1./(f[j, i]*dx)
            vor[j, i] = -1.*(v[j, i]-u[j, i]-v[j, i+1]+u[j+1, i])*cff
    return vor

@jit
def curl2d_overf_variabledx(u, v, f, pm, pn):
    ny, nx = u.shape
    vor = np.zeros(u.shape, dtype=u.dtype)
    for j in range(ny-1):
        for i in range(nx-1):
            cff = 1./(f[j, i])
            vor[j, i] = ( (v[j, i+1]-v[j, i])\
                          * 0.25 * (pm[j+1, i]+pm[j, i+1]+pm[j+1, i+1]+pm[j, i])\
                         -(u[j+1, i]-u[j, i])\
                          * 0.25 * (pn[j+1, i]+pn[j, i+1]+pn[j+1, i+1]+pn[j, i])\
                        ) * cff
            
    return vor
