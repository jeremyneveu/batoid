#include "grating.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Grating::Grating(double N, double rot) :
        _N(N), _rot(rot) {}

    Grating::~Grating() {}

    double Grating::sag(double x, double y) const {
        return 0.0;
    }

    void Grating::normal(double x, double y, double& nx, double& ny, double& nz) const {
        nx = 0.0;
        ny = 0.0;
        nz = 1.0;
    }

    void Grating::disp_axis(double x, double y, double& tx, double& ty, double& tz) const {
        tx = cos(_rot);
        ty = sin(_rot);
        tz = 0.0;
    }

    bool Grating::timeToIntersect(
        double x, double y, double z, double vx, double vy, double vz, double& dt
    ) const {
        if (vz == 0)
            return false;
        dt = -z/vz;
        return true;
    }

    double Grating::getN(double x, double y) const {
        return _N;
    }


    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Grating* Grating::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Grating* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new Grating(_N, _rot);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
