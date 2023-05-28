import numpy as np

from .surface import Surface
from .trace import refract_grating, reflect_grating, rSplit_grating

from . import _batoid


class Grating(Surface):
    """Planar grating, with N grooves per meter and oriented with rot in radians.
    The surface sag follows the equation:

    .. math::

        z(x, y) = 0
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
        return self._grating.getN(x, y)

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
    """Planar grating, with N grooves per meter and oriented with rot in radians.
    The surface sag follows the equation:

    .. math::

        z(x, y) = 0
    """
    def __init__(self, order=1, N=1e5, rot=0):
        Grating.__init__(self, order=order)
        self._surface = _batoid.CPPSimpleGrating(order, N, rot)
        self.N = N
        self.rot = rot

    def __hash__(self):
        return hash("batoid.SimpleGrating")

    def __setstate__(self, tup):
        order, N, rot = tup
        self.__init__(order, N, rot)

    def __getstate__(self):
        return (self.order, self.N, self.rot)

    def __eq__(self, rhs):
        return isinstance(rhs, SimpleGrating)

    def __repr__(self):
        return f"SimpleGrating({self.order},{self.N},{self.rot})"

