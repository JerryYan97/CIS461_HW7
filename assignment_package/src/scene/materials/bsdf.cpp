#include "bsdf.h"
#include <warpfunctions.h>

BSDF::BSDF(const Intersection& isect, float eta /*= 1*/)
//TODO: Properly set worldToTangent and tangentToWorld
    : worldToTangent(/*COMPUTE ME*/),
      tangentToWorld(/*COMPUTE ME*/),
      normal(isect.normalGeometric),
      eta(eta),
      numBxDFs(0),
      bxdfs{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}
{
    tangentToWorld = glm::mat3(isect.tangent, isect.bitangent, isect.normalGeometric);
    worldToTangent = glm::inverse(tangentToWorld);
}


void BSDF::UpdateTangentSpaceMatrices(const Normal3f& n, const Vector3f& t, const Vector3f b)
{
    //TODO: Update worldToTangent and tangentToWorld based on the normal, tangent, and bitangent
    tangentToWorld = glm::mat3(t, b, n);
    worldToTangent = glm::inverse(tangentToWorld);
}


Color3f BSDF::f(const Vector3f &woW, const Vector3f &wiW, BxDFType flags /*= BSDF_ALL*/) const
{
    //TODO
    Vector3f wiT = worldToTangent * wiW;
    Vector3f woT = worldToTangent * woW;
    bool reflect = glm::dot(wiW, normal) * glm::dot(woW, normal);
    Color3f resCol(0.f, 0.f, 0.f);
    for(int i = 0; i < numBxDFs; i++)
    {
        if(bxdfs[i]->MatchesFlags(flags) &&
                ((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) ||
                 (!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
        {
            resCol += bxdfs[i]->f(woT, wiT);
        }
    }
    return resCol;
}

// Use the input random number _xi_ to select
// one of our BxDFs that matches the _type_ flags.

// After selecting our random BxDF, rewrite the first uniform
// random number contained within _xi_ to another number within
// [0, 1) so that we don't bias the _wi_ sample generated from
// BxDF::Sample_f.

// Convert woW and wiW into tangent space and pass them to
// the chosen BxDF's Sample_f (along with pdf).
// Store the color returned by BxDF::Sample_f and convert
// the _wi_ obtained from this function back into world space.

// Iterate over all BxDFs that we DID NOT select above (so, all
// but the one sampled BxDF) and add their PDFs to the PDF we obtained
// from BxDF::Sample_f, then average them all together.

// Finally, iterate over all BxDFs and sum together the results of their
// f() for the chosen wo and wi, then return that sum.

/*
http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions.html#BSDF::Sample_f
?:
This is just a function that is responsible for these staffs:
1. Get wi;
2. Get albedo;
3. Get pdf;
*/
Color3f BSDF::Sample_f(const Vector3f &woW, Vector3f *wiW, const Point2f &xi,
                       float *pdf, BxDFType type, BxDFType *sampledType) const
{
    //TODO
    // Choose which BxDF to sample:
    int matchingComps = this->BxDFsMatchingFlags(type);
    if(matchingComps == 0)
    {
        *pdf = 0;
        return Color3f(0.f);
    }
    int comp = std::min((int)std::floor(xi[0] * matchingComps), matchingComps - 1);
    BxDF *bxdf = nullptr;
    int count = comp;
    for(int i = 0; i < numBxDFs; i++)
    {
        if(bxdfs[i]->MatchesFlags(type) && count-- == 0)
        {
            bxdf = bxdfs[i];
            break;
        }
    }

    // Remap BxDF sample u to [0,1)^2
    Point2f uRemapped(xi[0] * matchingComps - comp, xi[1]);

    // Sample Chosen BxDF:
    Vector3f wiT, woT = worldToTangent * woW;
    *pdf = 0;
    if(sampledType)
    {
        *sampledType = bxdf->type;
    }
    Color3f f = bxdf->Sample_f(woT, &wiT, uRemapped, pdf, sampledType);

    if(*pdf == 0)
    {
        return Color3f(0.f, 0.f, 0.f);
    }
    *wiW = tangentToWorld * wiT;

    // Compute overall PDF with all matching BxDFs:
    if(!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1)
    {
        for(int i = 0; i < numBxDFs; i++)
        {
            if(bxdfs[i] != bxdf && bxdfs[i]->MatchesFlags(type))
            {
                *pdf += bxdfs[i]->Pdf(woT, wiT);
            }
        }
    }
    if(matchingComps > 1)
    {
        *pdf /= matchingComps;
    }

    // Compute value of BSDF for sampled direction:
    if(!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1)
    {
        bool reflect = glm::dot(*wiW, normal) * glm::dot(woW, normal) > 0;
        f = Color3f(0.f, 0.f, 0.f);
        for(int i = 0; i < numBxDFs; i++)
        {
            if(bxdfs[i]->MatchesFlags(type) &&
                    ((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) ||
                    (!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
            {
                f += bxdfs[i]->f(woT, wiT);
            }
        }
    }
    return f;
}


float BSDF::Pdf(const Vector3f &woW, const Vector3f &wiW, BxDFType flags) const
{
    //TODO
    int matchingComps = this->BxDFsMatchingFlags(flags);
    if(matchingComps == 0)
    {
        return 0.f;
    }

    float resPdf = 0.f;
    Vector3f woT = worldToTangent * woW;
    Vector3f wiT = worldToTangent * wiW;
    for(int i = 0; i < numBxDFs; i++)
    {
        if(bxdfs[i]->MatchesFlags(flags))
        {
            resPdf += bxdfs[i]->Pdf(woT, wiT);
        }
    }
    return resPdf;
}

Color3f BxDF::Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &xi,
                       Float *pdf, BxDFType *sampledType) const
{
    //TODO
    // Cosine-sample the hemisphere, flipping the diection if necessary.
    *wi = WarpFunctions::squareToHemisphereUniform(xi);
    if(wo.z < 0)
    {
        wi->z *= -1;
    }
    *pdf = this->Pdf(wo, *wi);
    return this->f(wo, *wi);
}

// The PDF for uniform hemisphere sampling
float BxDF::Pdf(const Vector3f &wo, const Vector3f &wi) const
{
    // Square to hemisphere cosine PDF:
    return wi.z * InvPi;
    // Square to hemisphere uniform PDF:
    // return SameHemisphere(wo, wi) ? Inv2Pi : 0;
}

BSDF::~BSDF()
{
    for(int i = 0; i < numBxDFs; i++)
    {
        delete bxdfs[i];
    }
}
