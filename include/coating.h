#ifndef batoid_coating_h
#define batoid_coating_h

#include <cstdlib>


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Coating {
    public:
        Coating();
        virtual ~Coating();

        virtual void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double&transmit) const = 0;
        virtual double getReflect(double wavelength, double cosIncidenceAngle) const = 0;
        virtual double getTransmit(double wavelength, double cosIncidenceAngle) const = 0;

        virtual const Coating* getDevPtr() const = 0;

    protected:
        mutable Coating* _devPtr;

    private:
        #if defined(BATOID_GPU)
        void freeDevPtr() const;
        #endif
    };

    class SimpleCoating : public Coating {
    public:
        SimpleCoating(double reflectivity, double transmissivity);
        ~SimpleCoating();

        void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const override;
        double getReflect(double wavelength, double cosIncidenceAngle) const override;
        double getTransmit(double wavelength, double cosIncidenceAngle) const override;

        virtual const Coating* getDevPtr() const override;

    private:
        double _reflectivity;
        double _transmissivity;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    class TableCoating : public Coating {
    public:
        TableCoating(const double* args, const double* reflectivities, const double* transmissivities, const size_t size);
        ~TableCoating();

        void getCoefs(double wavelength, double cosIncidenceAngle, double& reflect, double& transmit) const override;
        double getReflect(double wavelength, double cosIncidenceAngle) const override;
        double getTransmit(double wavelength, double cosIncidenceAngle) const override;

        const Coating* getDevPtr() const override;

    private:
        const double* _args;
        const double* _reflectivities;
        const double* _transmissivities;
        const size_t _size;
    };



}

#endif
