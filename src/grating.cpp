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
    SimpleGrating::SimpleGrating(int order, double N) :
        Grating(order), _N(N) {}

    SimpleGrating::~SimpleGrating() {}

    void SimpleGrating::disp_axis(double x, double y, double& tx, double& ty, double& tz) const {
        tx = 1.0;
        ty = 0.0;
        tz = 0.0;
    }

    double SimpleGrating::getN(double x, double y) const {
        return _N;
    }

    // HologramGrating child class
    HologramGrating::HologramGrating(int order, double lbdaRec,
                                     double xA, double yA, double zA, 
                                     double xB, double yB, double zB) :
        Grating(order), _lbdaRec(lbdaRec), _xA(xA), _yA(yA), _zA(zA), _xB(xB), _yB(yB), _zB(zB) {}

    HologramGrating::~HologramGrating() {}

    double HologramGrating::_computeAPDist(double x, double y, double z) const {
        // Distance between virtual source A and the computing point P(x,y,z).
        double xDist = x - _xA;
        double yDist = y - _yA;
        double zDist = z - _zA;
        return sqrt(xDist * xDist + yDist * yDist + zDist * zDist);
    }

    double HologramGrating::_computeBPDist(double x, double y, double z) const {
        // Distance between virtual source B and the computing point P(x,y,z).
        double xDist = x - _xB;
        double yDist = y - _yB;
        double zDist = z - _zB;
        return sqrt(xDist * xDist + yDist * yDist + zDist * zDist);
    }

    double HologramGrating::getN(double x, double y) const {
        double dndx, dndy = 0.0;
        double rA = _computeAPDist(x, y, 0.);
        double rB = _computeBPDist(x, y, 0.);
        if (rA != 0)
            dndx += -(_xA - x) / rA;
            dndy += -(_yA - y) / rA;
        if (rB != 0)
            dndx += (_xB - x) / rB;
            dndy += (_yB - y) / rB;
        dndx /= _lbdaRec;
        dndy /= _lbdaRec;
        return dndx * sqrt(1 + (dndy * dndy) / (dndx * dndx));
    }

    void HologramGrating::disp_axis(double x, double y, double& tx, double& ty, double& tz) const {
        double dndx, dndy = 0.0;
        double rA = _computeAPDist(x, y, 0.);
        double rB = _computeBPDist(x, y, 0.);
        if (rA != 0)
            dndx += -(_xA - x) / rA;
            dndy += -(_yA - y) / rA;
        if (rB != 0)
            dndx += (_xB - x) / rB;
            dndy += (_yB - y) / rB;
        dndx /= _lbdaRec;
        dndy /= _lbdaRec;
        double Neff = dndx * sqrt(1 + (dndy * dndy) / (dndx * dndx));
        double rot = atan2(dndy, Neff);
        tx = cos(rot);
        ty = sin(rot);
        tz = 0.0;
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

    const Grating* HologramGrating::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                HologramGrating* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new HologramGrating(_N, _rot, _order);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
