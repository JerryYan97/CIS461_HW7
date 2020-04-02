#pragma once
#include "bsdf.h"
#include "fresnel.h"
#include "microfacet.h"

class MicrofacetBTDF : public BxDF
{
public:
    MicrofacetBTDF(const Color3f &T, MicrofacetDistribution *distribution, float etaA, float etaB, Fresnel* fresnel)
        :BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_GLOSSY)),
          T(T), etaA(etaA), etaB(etaB), distribution(distribution), fresnel(fresnel) {}

    virtual ~MicrofacetBTDF(){delete fresnel; delete distribution;}

    Color3f f(const Vector3f &wo, const Vector3f &wi) const;

    virtual Color3f Sample_f(const Vector3f &wo, Vector3f *wi,
                              const Point2f &xi, Float *pdf,
                              BxDFType *sampledType = nullptr) const;
    virtual float Pdf(const Vector3f &wo, const Vector3f &wi) const;

  private:
    const Color3f T; // The energy scattering coefficient of this BTDF (i.e. its color)
    float etaA, etaB;
    const MicrofacetDistribution* distribution;
    const Fresnel* fresnel; // A class that will determine the reflectivity coefficient at this point
};

