#ifndef MICROFACETMATERIAL_H
#define MICROFACETMATERIAL_H
#include "material.h"



class MicrofacetMaterial
{
public:
    MicrofacetMaterial();
};

class MicrofacetReflectionMaterial : public Material
{
public:
    MicrofacetReflectionMaterial(const Color3f &Kd,
                                 float sigma,
                                 float iRoughness,
                                 int distributionType,
                                 const std::shared_ptr<QImage> &textureMap,
                                 const std::shared_ptr<QImage> &normalMap)
                       : Kd(Kd), sigma(sigma), textureMap(textureMap), normalMap(normalMap), roughness(iRoughness), mType(distributionType)
                   { }

    void ProduceBSDF(Intersection *isect) const;


private:
    Color3f Kd;                    // The diffuse spectral reflection of this material.
                                   // This is just the base color of the material

    float sigma;                   // The roughness of this material (a single instance is a float). This determines whether or not
                                   // we use a Lambertian or Oren-Nayar BRDF in ComputeScatteringFunctions.
                                   // When nonzero, we use Oren-Nayar.
    float roughness;

    int mType;

    std::shared_ptr<QImage> textureMap; // The color obtained from this (assuming it is non-null) is multiplied with the base material color.
    std::shared_ptr<QImage> normalMap;
};

class MicrofacetTransmissiveMaterial : public Material
{
public:
    MicrofacetTransmissiveMaterial(const Color3f &Kt, const Color3f &Kr, float indexOfRefraction,
                                   float sigma,
                                   float iRoughness,
                                   int distributionType,
                                   const std::shared_ptr<QImage> &textureMapRefl,
                                   const std::shared_ptr<QImage> &textureMapTransmit,
                                   const std::shared_ptr<QImage> &normalMap)
                         : Kt(Kt), Kr(Kr), indexOfRefraction(indexOfRefraction),
                           textureMapRefl(textureMapRefl),
                           textureMapTransmit(textureMapTransmit),
                           normalMap(normalMap),
                           sigma(sigma),
                           roughness(iRoughness),
                           mType(distributionType)
                     {}

    void ProduceBSDF(Intersection *isect) const;
private:
    Color3f Kt;                    // The spectral transmission of this material.
                                   // This is just the base color of the material

    Color3f Kr;

    float indexOfRefraction;       // The IoR of this glass. We assume the IoR of
                                   // any external medium is that of a vacuum: 1.0

    std::shared_ptr<QImage> textureMapRefl;
    std::shared_ptr<QImage> textureMapTransmit;

    std::shared_ptr<QImage> normalMap;

    float sigma;

    float roughness;

    int mType;
};

#endif // MICROFACETMATERIAL_H
