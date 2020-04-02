#include "microfacetmaterial.h"
#include "microfacetbtdf.h"
#include "microfacetbrdf.h"
#include "specularbrdf.h"
#include "specularbtdf.h"
#include "lambertbrdf.h"


MicrofacetMaterial::MicrofacetMaterial()
{

}

void MicrofacetReflectionMaterial::ProduceBSDF(Intersection *isect) const
{
    // Important! Must initialize the intersection's BSDF!
    isect->bsdf = std::make_shared<BSDF>(*isect);

    Color3f color = Kd;
    if(this->textureMap)
    {
        color *= Material::GetImageColor(isect->uv, this->textureMap.get());
    }
    if(this->normalMap)
    {
        isect->bsdf->normal = isect->bsdf->tangentToWorld *  Material::GetImageColor(isect->uv, this->normalMap.get());
        //Update bsdf's TBN matrices to support the new normal
        Vector3f tangent, bitangent;
        CoordinateSystem(isect->bsdf->normal, &tangent, &bitangent);
        isect->bsdf->UpdateTangentSpaceMatrices(isect->bsdf->normal, tangent, bitangent);
    }

    if(sigma == 0.f)
    {
        // isect->bsdf->Add(new LambertBRDF(color));
        float tempAlpha = RoughnessToAlpha(roughness);
        isect->bsdf->Add(new MicrofacetBRDF(color, new TrowbridgeReitzDistribution(tempAlpha, tempAlpha), new FresnelNoOp()));
    }
    //Else do Oren-Nayar (not required implementation)
}

void MicrofacetTransmissiveMaterial::ProduceBSDF(Intersection *isect) const
{
    isect->bsdf = std::make_shared<BSDF>(*isect, indexOfRefraction);

    Color3f reflectColor = Kr;
    if(this->textureMapRefl)
    {
        reflectColor *= Material::GetImageColor(isect->uv, this->textureMapRefl.get());
    }

    Color3f transmitColor = Kt;
    if(this->textureMapTransmit)
    {
        transmitColor *= Material::GetImageColor(isect->uv, this->textureMapTransmit.get());
    }

    if(this->normalMap)
    {
        isect->bsdf->normal = isect->bsdf->tangentToWorld *  Material::GetImageColor(isect->uv, this->normalMap.get());
        //Update bsdf's TBN matrices to support the new normal
        Vector3f tangent, bitangent;
        CoordinateSystem(isect->bsdf->normal, &tangent, &bitangent);
        isect->bsdf->UpdateTangentSpaceMatrices(isect->bsdf->normal, tangent, bitangent);
    }

    if(sigma == 0.f)
    {
        // Note: if we assign Specular to Both microFacet material, then the combination would not be black.
        // That is the base of a glass material.

        // isect->bsdf->Add(new LambertBRDF(color));
        // float tempAlpha = RoughnessToAlpha(roughness);
        // isect->bsdf->Add(new MicrofacetBRDF(reflectColor, new TrowbridgeReitzDistribution(roughness, roughness), new FresnelDielectric(1.f, indexOfRefraction)));
        // isect->bsdf->Add(new SpecularBRDF(reflectColor, new FresnelDielectric(1.f, indexOfRefraction)));
        // isect->bsdf->Add(new SpecularBTDF(transmitColor, 1.f, indexOfRefraction, new FresnelDielectric(1.f, indexOfRefraction)));
        isect->bsdf->Add(new MicrofacetBTDF(transmitColor, new TrowbridgeReitzDistribution(roughness, roughness), 1.f, indexOfRefraction, new FresnelDielectric(1.f, indexOfRefraction)));
    }
}
