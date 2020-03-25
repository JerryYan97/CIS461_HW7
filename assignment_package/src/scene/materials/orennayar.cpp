#include "orennayar.h"
#include <warpfunctions.h>

Color3f OrenNayar::f(const Vector3f &wo, const Vector3f &wi) const
{
    float sinThetaI = SinTheta(wi);
    float sinThetaO = SinTheta(wo);

    // Compute cosine term of Oren-Nayar model:
    float maxCos = 0.f;
    if(sinThetaI > 1e-4 && sinThetaO > 1e-4)
    {
        float sinPhiI = SinPhi(wi);
        float cosPhiI = CosPhi(wi);
        float sinPhiO = SinPhi(wo);
        float cosPhiO = CosPhi(wo);
        float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
        maxCos = std::max((float)0, dCos);
    }

    // Compute sine and tangent terms of Oren-Nayar model:
    float sinAlpha, tanBeta;
    if (AbsCosTheta(wi) > AbsCosTheta(wo))
    {
        sinAlpha = sinThetaO;
        tanBeta = sinThetaI / AbsCosTheta(wi);
    }
    else
    {
        sinAlpha = sinThetaI;
        tanBeta = sinThetaO / AbsCosTheta(wo);
    }

    return R * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
}

Color3f OrenNayar::Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &sample, Float *pdf, BxDFType *sampledType) const
{
    return BxDF::Sample_f(wo, wi, sample, pdf, sampledType);
}

float OrenNayar::Pdf(const Vector3f &wo, const Vector3f &wi) const
{
    return BxDF::Pdf(wo, wi);
}
