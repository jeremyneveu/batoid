import batoid
import numpy as np
from test_helpers import timer, do_pickle, all_obj_diff, init_gpu


@timer
def test_properties():
    rng = np.random.default_rng(5)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)  # negative allowed
        conic = rng.uniform(-2.0, 1.0)
        ncoef = rng.choice(5)
        coefs = [rng.normal(0, 1e-8) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)
        assert asphere.R == R
        assert asphere.conic == conic
        assert np.array_equal(asphere.coefs, coefs)
        do_pickle(asphere)


def asphere_sag(R, conic, coefs):
    def f(x, y):
        r2 = x*x + y*y
        den = R*(1+np.sqrt(1-(1+conic)*r2/R/R))
        result = r2/den
        for i, a in enumerate(coefs):
            result += r2**(i+2) * a
        return result
    return f


def asphere_normal(R, conic, coefs):
    dzdrcoefs = coefs*np.arange(4, 4+2*len(coefs), 2)
    def f(x, y):
        r = np.hypot(x, y)
        dzdr = r/(R*np.sqrt(1-r*r*(1.+conic)/R/R))
        for i, a in enumerate(dzdrcoefs):
            dzdr += a*r**(3+2*i)
        nz = 1./np.sqrt(1+dzdr*dzdr)
        return np.dstack([-x/r*dzdr*nz, -y/r*dzdr*nz, nz])
    return f


@timer
def test_sag():
    rng = np.random.default_rng(57)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)  # negative allowed
        conic = rng.uniform(-2.0, 1.0)
        ncoef = rng.choice(5)
        coefs = [rng.normal(0, 1e-8) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)
        for j in range(100):
            lim = 0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 1
            x = rng.uniform(-lim, lim)
            y = rng.uniform(-lim, lim)
            result = asphere.sag(x, y)
            np.testing.assert_allclose(
                result,
                asphere_sag(R, conic, coefs)(x, y)
            )
            # Check that it returned a scalar float and not an array
            assert isinstance(result, float)
        # Check vectorization
        x = rng.uniform(-lim, lim, size=(10, 10))
        y = rng.uniform(-lim, lim, size=(10, 10))
        np.testing.assert_allclose(
            asphere.sag(x, y),
            asphere_sag(R, conic, coefs)(x, y)
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            asphere.sag(x[::5,::2], y[::5,::2]),
            asphere_sag(R, conic, coefs)(x, y)[::5,::2]
        )


@timer
def test_normal():
    rng = np.random.default_rng(577)
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)  # negative allowed
        conic = rng.uniform(-2.0, 1.0)
        ncoef = rng.choice(5)
        coefs = [rng.normal(0, 1e-8) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)
        for j in range(10):
            lim = 0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 1
            x = rng.uniform(-lim, lim)
            y = rng.uniform(-lim, lim)
            result = asphere.normal(x, y)
            np.testing.assert_allclose(
                result,
                asphere_normal(R, conic, coefs)(x, y)[0,0],
                rtol=0, atol=1e-14
            )
        # Check 0,0
        np.testing.assert_equal(asphere.normal(0, 0), np.array([0, 0, 1]))
        # Check vectorization
        x = rng.uniform(-lim, lim, size=(10, 10))
        y = rng.uniform(-lim, lim, size=(10, 10))
        np.testing.assert_allclose(
            asphere.normal(x, y),
            asphere_normal(R, conic, coefs)(x, y),
            rtol=0, atol=1e-14
        )
        # Make sure non-unit stride arrays also work
        np.testing.assert_allclose(
            asphere.normal(x[::5,::2], y[::5,::2]),
            asphere_normal(R, conic, coefs)(x, y)[::5, ::2],
            rtol=0, atol=1e-14
        )


@timer
def test_intersect():
    rng = np.random.default_rng(5772)
    size = 1
    for i in range(100):
        R = 1./rng.normal(0.0, 0.3)  # negative allowed
        conic = rng.uniform(-2.0, 1.0)
        ncoef = rng.choice(5)
        coefs = [rng.normal(0, 1e-8) for i in range(ncoef)]
        asphere = batoid.Asphere(R, conic, coefs)
        asphereCoordSys = batoid.CoordSys(origin=[0, 0, -1])
        lim = min(0.7*abs(R)/np.sqrt(1+conic) if conic > -1 else 10, 10)
        x = rng.uniform(-lim, lim, size=size)
        y = rng.uniform(-lim, lim, size=size)
        z = np.full_like(x, -100.0)
        # If we shoot rays straight up, then it's easy to predict the intersection
        vx = np.zeros_like(x)
        vy = np.zeros_like(x)
        vz = np.ones_like(x)
        rv = batoid.RayVector(x, y, z, vx, vy, vz)
        np.testing.assert_allclose(rv.z, -100.0)
        coordTransform = batoid.CoordTransform(rv.coordSys, asphereCoordSys)
        rv2 = batoid.intersect(asphere, rv.copy(), coordTransform)
        assert rv2.coordSys == asphereCoordSys

        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, asphere.sag(x, y)-1,
            rtol=0, atol=1e-9
        )

        # Check default intersect coordTransform
        rv2 = rv.copy().toCoordSys(asphereCoordSys)
        batoid.intersect(asphere, rv2)
        assert rv2.coordSys == asphereCoordSys
        rv2 = rv2.toCoordSys(batoid.CoordSys())
        np.testing.assert_allclose(rv2.x, x)
        np.testing.assert_allclose(rv2.y, y)
        np.testing.assert_allclose(
            rv2.z, asphere.sag(x, y)-1,
            rtol=0, atol=1e-9
        )


@timer
def test_ne():
    objs = [
        batoid.Asphere(10.0, 1.0, []),
        batoid.Asphere(10.0, 1.0, [0]),
        batoid.Asphere(10.0, 1.0, [0,1]),
        batoid.Asphere(10.0, 1.0, [1,0]),
        batoid.Asphere(10.0, 1.1, []),
        batoid.Asphere(10.1, 1.0, []),
        batoid.Quadric(10.0, 1.0)
    ]
    all_obj_diff(objs)


@timer
def test_fail():
    asphere = batoid.Asphere(1.0, 1.0, [1.0, 1.0])
    # This one should fail, since already passed the quadric.
    rv = batoid.RayVector(0, 0, -1, 0, 0, -1)
    rv2 = batoid.intersect(asphere, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([True]))
    # This one passes
    rv = batoid.RayVector(0, 0, -1, 0, 0, +1)
    rv2 = batoid.intersect(asphere, rv.copy())
    np.testing.assert_equal(rv2.failed, np.array([False]))


if __name__ == '__main__':
    init_gpu()
    test_properties()
    test_sag()
    test_normal()
    test_intersect()
    test_ne()
    test_fail()
