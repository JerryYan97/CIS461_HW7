#include "fulllightingintegrator.h"
#include "directlightingintegrator.h"

Color3f FullLightingIntegrator::Li(const Ray &ray, const Scene &scene, std::shared_ptr<Sampler> sampler, int depth) const
{
    // Instantiate an accumlated ray color that begins as black;
    Color3f L(0.f);
    // Instantiate an accumulated ray throughput color that begins as white;
    // The throughput will be used to determine when your ray path terminates via the Russian Roulette heuristic.
    Color3f beta(1.f);

    Ray rayPath(ray);

    bool specBounce = false;

    // Simply declare a while loop that compares some current depth value to 0,
    // assuming that depth began as the maximum depth value.
    // Within this loop, we will add a check that breaks the loop early if the Russian Roulette conditions are satisfied.
    int bounceCounter = depth;
    while(bounceCounter > 0)
    {
        // Intersect ray with scene and store intersection in isect.
        // Find closest ray intersection or return background radiance.
        Intersection isect;
        if(!scene.Intersect(rayPath, &isect))
        {
            break;
        }

        // Initialize common variable for integrator.
        Vector3f wo = - rayPath.direction;

        // Compute emitted light if ray hit an area light source.
        // if(isect.objectHit->GetAreaLight() || (specBounce && (depth - bounceCounter == 1)))
        if(isect.objectHit->GetAreaLight())
        {
            if(bounceCounter == depth || specBounce)
            {
                Color3f lightSource = isect.Le(wo);
                L += beta * lightSource;
                break;
            }
            else
            {
                break;
            }
        }

        // Ask _objectHit_ to produce a BSDF
        isect.ProduceBSDF();

        // If previous hit is a specular point, while this hit is not light source.
        // Then, we reset the specBounce flag.
        specBounce = false;

        // Initialize normal for integrator:
        // Normal3f n = isect.normalGeometric;
        Normal3f n = isect.bsdf->normal;

        // Check whether hit a specular object
        Color3f mLiSpec(0.f);
        Vector3f wiSpec;
        float pdfSpec;
        BxDFType flagsSpec;
        Color3f fSpec = isect.bsdf->Sample_f(wo, &wiSpec, sampler->Get2D(), &pdfSpec, BSDF_ALL, &flagsSpec);
        if(flagsSpec & BxDFType::BSDF_SPECULAR)
        {
            beta *= (fSpec * AbsDot(wiSpec, n) / pdfSpec);
            specBounce = true;
            rayPath = isect.SpawnRay(wiSpec);
            --bounceCounter;
            continue;
        }



        // L calculation.
        // Computing the direct lighting component.
        Color3f LTerm(0.f);

        // Light PDF sampling:
        // Randomly select a light source from scene.lights and call its Sample_Li function:
        int nLights = int(scene.lights.size());
        if (nLights == 0)
        {
            L = Color3f(0.f);
            break;
        }
        int lightIdx = std::min((int)(sampler->Get1D() * nLights), nLights - 1);
        const std::shared_ptr<Light> &light = scene.lights[lightIdx];

        Vector3f wi(0.f);
        float pdf = 0.f;
        Color3f LiDir = light->Sample_Li(isect, sampler->Get2D(), &wi, &pdf);

        // Shadow Test and evaluate the LTE for light PDF sampling:
        // Ray rayWi = Ray(isect.point + isect.normalGeometric * 1e-4f, wi);
        Ray rayWi = isect.SpawnRay(wi);
        Intersection tempWiInsect;
        if(!scene.Intersect(rayWi, &tempWiInsect))
        {
            LiDir = Color3f(0.f);
        }
        else
        {
            if(light.get() != tempWiInsect.objectHit->GetAreaLight())
            {
                LiDir = Color3f(0.f);
            }
            else
            {
                Color3f f = isect.bsdf->f(wo, wi);
                float lMaterialPDF = isect.bsdf->Pdf(wo, wi);
                if(pdf <= 0.f || lMaterialPDF <= 0.f)
                {
                    LiDir = Color3f(0.f);
                }
                else
                {
                    float lightPDFWeight = PowerHeuristic(1, pdf, 1, lMaterialPDF);
                    LTerm += f * LiDir * AbsDot(wi, n) * lightPDFWeight / (pdf / nLights);
                }
            }
        }

        // ************************************************* //

        // BRDF PDF sampling:
        Vector3f mWiB(0.f);
        Color3f mLiB(0.f);
        float mPDFMaterial = 0.f;
        BxDFType flags;
        Color3f mF = isect.bsdf->Sample_f(wo, &mWiB, sampler->Get2D(), &mPDFMaterial, BSDF_ALL, &flags);
        if(!IsBlack(mF) && mPDFMaterial != 0.f)
        {
            Ray wibRay = isect.SpawnRay(mWiB);
            Intersection intersectBRDF;
            if(scene.Intersect(wibRay, &intersectBRDF))
            {
                if(intersectBRDF.objectHit->GetAreaLight() == light.get())
                {
                    const AreaLight* arealightPtr = intersectBRDF.objectHit->GetAreaLight();
                    float pdfLWB = arealightPtr->Pdf_Li(isect, mWiB);
                    float wWiB = PowerHeuristic(1, mPDFMaterial, 1, pdfLWB);
                    mLiB = mF * LiDir * AbsDot(mWiB, n) * wWiB / (pdfLWB / nLights);
                    LTerm += mLiB;
                }
            }
        }

        // Computing the ray bounce and global illumination.
        Color3f mLiG(0.f);
        Vector3f wiG;
        float pdfG;
        BxDFType flagsG;
        Color3f fG = isect.bsdf->Sample_f(wo, &wiG, sampler->Get2D(), &pdfG, BSDF_ALL, &flagsG);
        if(IsBlack(fG) || pdfG == 0.f)
        {
            break;
        }
        beta *= fG * AbsDot(wiG, n) / pdfG;
        L += (beta * LTerm);
        rayPath = isect.SpawnRay(wiG);

        // Correctly accounting for direct lighting.

        // Russian Roulette Ray Termination.
        // Compare the maximum RGB component of your throughput to a uniform random number and
        // stop your while loop if said component is smaller than the random number.
        float maxChannel = beta[0];
        for(int i = 1; i < 3; i++)
        {
            if(beta[i] > maxChannel)
            {
                maxChannel = beta[i];
            }
        }

        float zeta = sampler->Get1D();
        if(maxChannel < (1.f - zeta))
        {
            break;
        }
        else
        {
            beta *= (1.f / maxChannel);
        }

        --bounceCounter;
    }
    //TODO
    return L;
}
