#include "microfacetmaterial.h"
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
        if(mType == 0)
        {
            isect->bsdf->Add(new MicrofacetBRDF(color, new TrowbridgeReitzDistribution(tempAlpha, tempAlpha), new FresnelNoOp()));
        }
        else
        {
            isect->bsdf->Add(new MicrofacetBRDF(color, new BeckmannDistribution(tempAlpha, tempAlpha), new FresnelNoOp()));
        }
    }
    //Else do Oren-Nayar (not required implementation)
}
