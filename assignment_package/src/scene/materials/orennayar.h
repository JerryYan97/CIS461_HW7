#ifndef ORENNAYAR_H
#define ORENNAYAR_H
#include "bsdf.h"

class OrenNayar : public BxDF
{
public:
    OrenNayar(const Color3f &R, float sigmaDeg)
        : BxDF(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {
        float sigmaRad = glm::radians(sigmaDeg);
        float sigmaRad2 = sigmaRad * sigmaRad;
        A = 1.f - (sigmaRad2 / (2.f * (sigmaRad2 + 0.33f)));
        B = 0.45f * sigmaRad2 / (sigmaRad2 + 0.09f);
    }
    Color3f f(const Vector3f &wo, const Vector3f &wi) const;
    virtual Color3f Sample_f(const Vector3f &wo, Vector3f *wi,
                              const Point2f &sample, Float *pdf,
                              BxDFType *sampledType = nullptr) const;
    virtual float Pdf(const Vector3f &wo, const Vector3f &wi) const;
private:
    const Color3f R;
    float A, B;
};

#endif // ORENNAYAR_H
