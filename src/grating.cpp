#include "grating.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    // Grating parent class
    Grating::Grating(int order) :
        _order(order)
    {}

    Grating::~Grating() {}

    double Grating::sag(double x, double y) const {
        return 0.0;
    }

    void Grating::normal(double x, double y, double& nx, double& ny, double& nz) const {
        nx = 0.0;
        ny = 0.0;
        nz = 1.0;
    }

    bool Grating::timeToIntersect(
        double x, double y, double z, double vx, double vy, double vz, double& dt
    ) const {
        if (vz == 0)
            return false;
        dt = -z/vz;
        return true;
    }

    int Grating::getOrder() const {
        return _order;
    }

    double Grating::getN(double x, double y) const {
        return 0.0;
    }

    void Grating::disp_axis(double x, double y, double& tx, double& ty, double& tz) const {
        tx = 1.0;
        ty = 0.0;
        tz = 0.0;
    }

    // SimpleGrating child class
    SimpleGrating::SimpleGrating(int order, double N, double rot) :
        Grating(order), _N(N), _rot(rot) {}

    SimpleGrating::~SimpleGrating() {}

    void SimpleGrating::disp_axis(double x, double y, double& tx, double& ty, double& tz) const {
        tx = cos(_rot);
        ty = sin(_rot);
        tz = 0.0;
    }

    double SimpleGrating::getN(double x, double y) const {
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
                    ptr = new Grating(_order);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

    const Grating* SimpleGrating::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                SimpleGrating* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new SimpleGrating(_N, _rot, _order);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
