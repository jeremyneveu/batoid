#include "plane.h"
#include <cmath>

namespace batoid {
    Ray Plane::intercept(const Ray& r) const {
        if (r.failed)
            return Ray(true);
        double t = -r.p0.z/r.v.z + r.t0;
        Vec3 point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    Intersection Plane::intersect(const Ray& r) const {
        if (r.failed)
            return Intersection(true);
        double t = -r.p0.z/r.v.z + r.t0;
        Vec3 point = r.positionAtTime(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal);
    }

    std::string Plane::repr() const {
        return std::string("Plane()");
    }

    inline std::ostream& operator<<(std::ostream& os, const Plane& p) {
        return os << p.repr();
    }
}
