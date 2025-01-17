#ifndef batoid_plane_h
#define batoid_plane_h

#include "surface.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Plane : public Surface {
    public:
        Plane();
        ~Plane();

        virtual const Surface* getDevPtr() const override;

        virtual double sag(double x, double y) const override;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const override;
        virtual bool timeToIntersect(
            double x, double y, double z,
            double vx, double vy, double vz,
            double& dt
        ) const override;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif
}
#endif
