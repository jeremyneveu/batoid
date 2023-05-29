import numpy as np

from .surface import Surface
from .trace import refract_grating, reflect_grating, rSplit_grating

from . import _batoid


class Grating(Surface):
    """Planar grating, with N grooves per meter and oriented with rot in radians.
    The surface sag follows the equation:

    .. math::

        z(x, y) = 0

    Parameters
    ----------
    order : int
        Diffraction order of the grating (default 1).

    """
    def __init__(self, order=1):
        self.order = order

    def getN(self, x, y):
        """Return local number of grooves per meter.

        Parameters
        ----------
        x, y : array_like, shape (n,)
            Positions at which to evaluate the surface normal.

        Returns
        -------
        N : float
            Local number of grooves per meter.
        """
        return self._surface.getN(x, y)

    def disp_axis(self, x, y):
        """Return local dispersion axis unit vector.

        Parameters
        ----------
        x, y : array_like, shape (n,)
            Positions at which to evaluate the surface normal.

        Returns
        -------
        disp_axis : array_like, shape (n, 3)
            Surface normals.
        """
        xx = np.asfortranarray(x, dtype=float)
        yy = np.asfortranarray(y, dtype=float)
        out = np.empty(xx.shape+(3,), order='F', dtype=float)
        size = len(xx.ravel())

        self._surface.disp_axis(
            xx.ctypes.data, yy.ctypes.data, size, out.ctypes.data
        )
        try:
            len(x)
        except TypeError:
            return out[0]
        else:
            return out

    def reflect(self, rv, coordSys=None, coating=None):
        """Calculate intersection of rays with this surface, and immediately
        reflect the rays at the points of intersection.

        Parameters
        ----------
        rv : RayVector
            Rays to reflect.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.
        coating : Coating, optional
            Apply this coating upon surface intersection.

        Returns
        -------
        outRays : RayVector
            New object corresponding to original rays propagated and reflected.
        """
        return reflect_grating(self, rv, coordSys, coating)

    def refract(self, rv, inMedium, outMedium, coordSys=None, coating=None):
        """Calculate intersection of rays with this surface, and immediately
        refract the rays through the surface at the points of intersection.

        Parameters
        ----------
        rv : RayVector
            Rays to refract.
        inMedium : Medium
            Refractive medium on the incoming side of the surface.
        outMedium : Medium
            Refractive medium on the outgoing side of the surface.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.
        coating : Coating, optional
            Apply this coating upon surface intersection.

        Returns
        -------
        outRays : RayVector
            New object corresponding to original rays propagated and refracted.
        """
        return refract_grating(self, rv, inMedium, outMedium, coordSys, coating)

    def rSplit(self, rv, inMedium, outMedium, coating, coordSys=None):
        """Calculate intersection of rays with this surface, and immediately
        split the rays into reflected and refracted rays, with appropriate
        fluxes.

        Parameters
        ----------
        rv : RayVector
            Rays to refract.
        inMedium : Medium
            Refractive medium on the incoming side of the surface.
        outMedium : Medium
            Refractive medium on the outgoing side of the surface.
        coating : Coating
            Coating object to control transmission coefficient.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.

        Returns
        -------
        reflectedRays, refractedRays : RayVector
            New objects corresponding to original rays propagated and
            reflected/refracted.
        """
        return rSplit_grating(self, rv, inMedium, outMedium, coating, coordSys)


class SimpleGrating(Grating):
    """Planar grating, with N grooves per meter. Dispersion axis is along x axis.
    The surface sag follows the equation:

    .. math::

        z(x, y) = 0

    Parameters
    ----------
    order: int
        Diffraction order of the grating (default: 1).
    N: float
        Number of grooves per meter (default: 1e5 per meter).

    """
    def __init__(self, order=1, N=1e5):
        Grating.__init__(self, order=order)
        self._surface = _batoid.CPPSimpleGrating(order, N)
        self.N = N

    def __hash__(self):
        return hash("batoid.SimpleGrating")

    def __setstate__(self, tup):
        order, N = tup
        self.__init__(order, N)

    def __getstate__(self):
        return (self.order, self.N)

    def __eq__(self, rhs):
        return isinstance(rhs, SimpleGrating)

    def __repr__(self):
        return f"SimpleGrating({self.order},{self.N})"


class HologramGrating(Grating):
    r"""

    Parameters
    ----------
    order : int
        Diffraction order of the grating (default 1).
    lambda_record: float
        Record wavelength of the hologram in meter (default: 500e-9 meter).
    sourceAVec: array_like, shape (n, 3)
        Position vector of virtual source A in meter with respect to grating center (default: [-1e-2, 0, 5e-2]).
    sourceBVec: array_like, shape (n, 3)
        Position vector of virtual source B in meter with respect to grating center (default: [1e-2, 0, 5e-2]).
    """

    def __init__(self, order=1, lambda_record=500e-9, sourceAVec=(-1e-2, 0, 5e-2), sourceBVec=(1e-2, 0, 5e-2)):
        Grating.__init__(self, order=order)
        self._surface = _batoid.CPPHologramGrating(order, lambda_record, *sourceAVec, *sourceBVec)
        self.sourceAVec = sourceAVec
        self.sourceBVec = sourceBVec
        self.lambda_record = lambda_record

    def __hash__(self):
        return hash("batoid.HologramGrating")

    def __setstate__(self, tup):
        order, lambda_record, sourceAVec, sourceBVec = tup
        self.__init__(order, lambda_record, sourceAVec, sourceBVec)

    def __getstate__(self):
        return (self.order, self.lambda_record, self.sourceAVec, self.sourceBVec)

    def __eq__(self, rhs):
        return isinstance(rhs, HologramGrating)

    def __repr__(self):
        return f"HologramGrating({self.order},{self.lambda_record},{self.sourceAVec},{self.sourceBVec})"

    @property
    def AB(self):
        return np.sqrt((self.sourceBVec[0] - self.sourceAVec[0]) ** 2 + (self.sourceBVec[1] - self.sourceAVec[1]) ** 2)

    def plot(self):
        import matplotlib.pyplot as plt
        xs = np.linspace(-self.AB/2-self.sourceAVec[0], self.AB/2+self.sourceBVec[0], 50)
        ys = np.linspace(-self.AB/2-self.sourceAVec[1], self.AB/2+self.sourceBVec[1], 50)
        xx, yy = np.meshgrid(xs, ys)
        NN = self.getN(xx, yy)
        arr = self.disp_axis(xx, yy)
        tx = arr[:,:,0]
        ty = arr[:,:,1]
        alpha = np.rad2deg(np.arctan2(ty, tx))
        fig = plt.figure()
        im = plt.pcolor(xs, ys, alpha, cmap="bwr", shading="auto")
        plt.grid()
        CS = plt.contour(xs, ys, NN*1e-3, 10, linewidths=0.5, colors='k')
        plt.clabel(CS, inline=1, fontsize=14, fmt='%.1f')
        labels = ['Lines per mm']
        #CS.collections[0].set_label(labels[0])
        cb = fig.colorbar(im, fraction=0.046, pad=0.08)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label(r'Rotation angle $\alpha$ [$^\circ$]', fontsize=14)
        plt.gca().set_aspect("equal")
        plt.scatter(self.sourceAVec[0], self.sourceAVec[1], s=200, facecolor='red', label="Optical center (A')", zorder=42)
        plt.scatter(self.sourceBVec[0], self.sourceBVec[1], s=200, facecolor='black', label=r"Order +1 position at $\lambda_R$ (B')",
                    zorder=42)
        plt.xlabel(r"$x$ [m]", fontsize=14)
        plt.ylabel(r"$y$ [m]", fontsize=14)
        plt.legend()
        plt.xlim(np.min(xs), np.max(xs))
        plt.ylim(np.min(ys), np.max(ys))
        fig.tight_layout()
        plt.show()
