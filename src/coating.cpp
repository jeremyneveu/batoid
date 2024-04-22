#include "coating.h"
#include <new>
#include <cmath>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Coating::Coating() :
        _devPtr(nullptr)
    {}

    Coating::~Coating() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                freeDevPtr();
            }
        #endif
    }

    SimpleCoating::SimpleCoating(double reflectivity, double transmissivity) :
        _reflectivity(reflectivity), _transmissivity(transmissivity)
    {}

    SimpleCoating::~SimpleCoating() {}

    void SimpleCoating::getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const {
        reflect = _reflectivity;
        transmit = _transmissivity;
    }

    double SimpleCoating::getReflect(double wavelength, double cosIncidenceAngle) const {
        return _reflectivity;
    }

    double SimpleCoating::getTransmit(double wavelength, double cosIncidenceAngle) const {
        return _transmissivity;
    }

    TableCoating::TableCoating(
            const double* args, const double* reflectivities, const double* transmissivities, const size_t size
    ) :
        Coating(), _args(args), _reflectivities(reflectivities), _transmissivities(transmissivities), _size(size)
    {}

    TableCoating::~TableCoating() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                const double* args = _args;
                const double* reflectivities = _reflectivities;
                const double* transmissivities = _transmissivities;
                #pragma omp target exit data \
                    map(release:args[:_size], reflectivities[:_size], transmissivities[:_size])
            }
        #endif
    }

    void TableCoating::getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const {
        reflect = getReflect(wavelength, cosIncidenceAngle);
        transmit = getTransmit(wavelength, cosIncidenceAngle);
    }

    double TableCoating::getReflect(double wavelength, double cosIncidenceAngle) const {
        // Linear search.  Better for GPU's?  and not that painful for small arrays?
        // TODO: Need to implement cosIncidentAngle dependency
        if (wavelength < _args[0])
            return NAN;
        if (wavelength > _args[_size-1])
            return NAN;
        int upperIdx;
        for(upperIdx=1; upperIdx<_size; upperIdx++) {
            if (wavelength < _args[upperIdx])
                break;
        }
        double out = (wavelength - _args[upperIdx-1]);
        out *= (_reflectivities[upperIdx] - _reflectivities[upperIdx-1]);
        out /= (_args[upperIdx] - _args[upperIdx-1]);
        out += _reflectivities[upperIdx-1];
        return out;
    }

    double TableCoating::getTransmit(double wavelength, double cosIncidenceAngle) const {
        // Linear search.  Better for GPU's?  and not that painful for small arrays?
        // TODO: Need to implement cosIncidentAngle dependency
        if (wavelength < _args[0])
            return NAN;
        if (wavelength > _args[_size-1])
            return NAN;
        int upperIdx;
        for(upperIdx=1; upperIdx<_size; upperIdx++) {
            if (wavelength < _args[upperIdx])
                break;
        }
        double out = (wavelength - _args[upperIdx-1]);
        out *= (_transmissivities[upperIdx] - _transmissivities[upperIdx-1]);
        out /= (_args[upperIdx] - _args[upperIdx-1]);
        out += _transmissivities[upperIdx-1];
        return out;
    }




    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    #if defined(BATOID_GPU)
    void Coating::freeDevPtr() const {
        if(_devPtr) {
            Coating* ptr = _devPtr;
            _devPtr = nullptr;
            #pragma omp target is_device_ptr(ptr)
            {
                delete ptr;
            }
        }
    }
    #endif

    const Coating* SimpleCoating::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Coating* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new SimpleCoating(_reflectivity, _transmissivity);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

    const Coating* TableCoating::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Coating* ptr;
                // Allocate arrays on device
                const double* args = _args;
                const double* reflectivities = _reflectivities;
                const double* transmissivities = _transmissivities;
                #pragma omp target enter data \
                    map(to:args[:_size], reflectivities[:_size], transmissivities[:_size])
                #pragma omp target map(from:ptr)
                {
                    ptr = new TableCoating(args, reflectivities, transmissivities, _size);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }


}
