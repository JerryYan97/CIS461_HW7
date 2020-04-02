#include "microfacetbtdf.h"

Color3f MicrofacetBTDF::Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &xi, Float *pdf, BxDFType *sampledType) const
{
    if(wo.z == 0)
    {
        return Color3f(0.f);
    }
    Vector3f wh = distribution->Sample_wh(wo, xi);

    if(glm::dot(wo, wh) < 0.f)
    {
        return Color3f(0.f);
    }

    Float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
    if (!Refract(wo, (Normal3f)wh, eta, wi))
    {
        return Color3f(0.f);
    }
    *sampledType = this->type;
    *pdf = Pdf(wo, *wi);
    return f(wo, *wi);
}

Color3f MicrofacetBTDF::f(const Vector3f &wo, const Vector3f &wi) const
{
    // Transmission only.
    if(SameHemisphere(wo, wi))
    {
        return Color3f(0.f);
    }

    Float cosThetaO = CosTheta(wo);
    Float cosThetaI = CosTheta(wi);
    if (cosThetaI == 0 || cosThetaO == 0)
    {
        return Color3f(0.f);
    }

    Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);

    Vector3f wh = glm::normalize(wo + wi * eta);

    if(wh.z < 0.f)
    {
        wh = -wh;
    }

    Color3f F = fresnel->Evaluate(glm::dot(wo, wh));

    Float sqrtDenom = glm::dot(wo, wh) + eta * glm::dot(wi, wh);

    Float factor = 1.f;
    // Float factor = 1.f / eta;

    return (Color3f(1.f) - F) * T * glm::abs(distribution->D(wh) * distribution->G(wo, wi) * eta * eta *
                                             AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
                                             (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

Float MicrofacetBTDF::Pdf(const Vector3f &wo, const Vector3f &wi) const
{
    if(SameHemisphere(wo, wi))
    {
        return 0.f;
    }
    Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
    Vector3f wh = glm::normalize(wo + wi * eta);
    Float sqrtDenom = glm::dot(wo, wh) + eta * glm::dot(wi, wh);
    Float dwh_dwi = std::abs((eta * eta * glm::dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return distribution->Pdf(wo, wh) * dwh_dwi;
}
