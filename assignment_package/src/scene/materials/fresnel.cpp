#include "fresnel.h"

Color3f FresnelDielectric::Evaluate(float cosThetaI) const
{
    //TODO
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    // Potentially swap indices of refraction.
    float currEtaI = etaI;
    float currEtaT = etaT;
    bool entering = (cosThetaI > 0.f);
    // if(!entering)
    if(!entering)
    {
        std::swap(currEtaI, currEtaT);
        cosThetaI = std::abs(cosThetaI);
    }

    // Compute cosThetaT using Snell's law.
    float sinThetaI = std::sqrt(std::max(0.f, 1.f - cosThetaI * cosThetaI));

    float sinThetaT = currEtaI / currEtaT * sinThetaI;

    // Handle total internal reflection.
    if(sinThetaT >= 1.f)
    {
       return Color3f(1.f);
    }

    float cosThetaT = std::sqrt(std::max(0.f, 1.f - sinThetaT * sinThetaT));

    float Rparl = ((currEtaT * cosThetaI) - (currEtaI * cosThetaT)) / ((currEtaT * cosThetaI) + (currEtaI * cosThetaT));
    float Rperp = ((currEtaI * cosThetaI) - (currEtaT * cosThetaT)) / ((currEtaI * cosThetaI) + (currEtaT * cosThetaT));

    return Color3f((Rparl * Rparl + Rperp * Rperp) / 2.f);
    // return Color3f(1.f);
}


Color3f FrConductor(float cosThetaI, const Color3f &etai, const Color3f &etat, const Color3f &k)
{
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    Color3f eta = etat / etai;
    Color3f etak = k / etai;

    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1.f - cosThetaI2;
    Color3f eta2 = eta * eta;
    Color3f etak2 = etak * etak;

    Color3f t0 = eta2 - etak2 - sinThetaI2;
    Color3f a2plusb2 = glm::sqrt(t0 * t0 + 4.f * eta2 * etak2);
    Color3f t1 = a2plusb2 + cosThetaI2;
    Color3f a = glm::sqrt(0.5f * (a2plusb2 + t0));
    Color3f t2 = (Float)2 * cosThetaI * a;
    Color3f Rs = (t1 - t2) / (t1 + t2);

    Color3f t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    Color3f t4 = t2 * sinThetaI2;
    Color3f Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5f * (Rp + Rs);
}

Color3f FresnelConductor::Evaluate(float cosThetaI) const
{
    return FrConductor(std::abs(cosThetaI), etaI, etaT, k);
}
