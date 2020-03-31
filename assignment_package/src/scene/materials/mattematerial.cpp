#include "mattematerial.h"
#include "lambertbrdf.h"

void MatteMaterial::ProduceBSDF(Intersection *isect) const
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
        // Check whether the normal map is truely applied.
        // std::cout << "normal:" << isect->bsdf->normal.x << " " << isect->bsdf->normal.y << " " << isect->bsdf->normal.z << std::endl;
    }

    if(sigma == 0.f)
    {
        isect->bsdf->Add(new LambertBRDF(color));
    }
    //Else do Oren-Nayar (not required implementation)
}
