#ifndef batoid_grating_h
#define batoid_grating_h

#include "surface.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Grating : public Surface {
    public:
        Grating(int order);
        ~Grating();

        virtual const Grating* getDevPtr() const override;
        virtual int getOrder() const;
        virtual double getN(double x, double y) const;

        virtual double sag(double, double) const override;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const override;
        virtual void disp_axis(
            double x, double y,
            double& tx, double& ty, double& tz
        ) const;
        virtual bool timeToIntersect(
            double x, double y, double z,
            double vx, double vy, double vz,
            double& dt
        ) const override;

    protected:
        const int _order;
    };

    class SimpleGrating : public Grating {
    public:
        SimpleGrating(int order, double N, double rot);
        ~SimpleGrating();

        virtual const Grating* getDevPtr() const override;
        virtual double getN(double x, double y) const override;

        virtual void disp_axis(
            double x, double y,
            double& tx, double& ty, double& tz
        ) const override;

    protected:
        const double _N, _rot;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

}
#endif
