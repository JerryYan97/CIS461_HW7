#include "fulllightingintegrator.h"
#include "directlightingintegrator.h"


Color3f DirectLight(const Ray &ray, const Scene &scene, std::shared_ptr<Sampler> sampler)
{
    //TODO
    Color3f L(0.f);


    // Find closest ray intersection or return background radiance.
    Intersection isect;

    if(!scene.Intersect(ray, &isect))
    {
        return L;
    }

    // Compute emitted and reflected light at ray intersection point

    // Initialize common variable for Whitted integrator
    Vector3f wo = - ray.direction;

    // Compute scattering functions for surface interaction

    // Compute emitted light if ray hit an area light source
    Color3f lightSource = isect.Le(wo);
    if(!IsBlack(lightSource))
    {
        L += lightSource;
        return L;
    }

    // Ask _objectHit_ to produce a BSDF
    // based on other data stored in this
    // Intersection, e.g. the surface normal
    // and UV coordinates
    isect.ProduceBSDF();
    Normal3f n = isect.bsdf->normal;

    // Light PDF sampling:
    // Randomly select a light source from scene.lights and call its Sample_Li function:
    int nLights = int(scene.lights.size());
    if (nLights == 0) return Color3f(0.f);
    int lightNum = std::min((int)(sampler->Get1D() * nLights), nLights - 1);
    const std::shared_ptr<Light> &light = scene.lights[lightNum];

    Vector3f wi(0.f);
    float pdf = 0.f;
    Color3f Li = light->Sample_Li(isect, sampler->Get2D(), &wi, &pdf);

    // Shadow Test and evaluate the LTE for Light PDF sampling:
    Ray tempRay = Ray(isect.point + isect.normalGeometric * 1e-4f, wi);
    Intersection tempInsect;
    if(!scene.Intersect(tempRay, &tempInsect))
    {
        Li = Color3f(0.f);
    }
    else
    {
        if(light.get() != tempInsect.objectHit->GetAreaLight())
        {
            Li = Color3f(0.f);
        }
        else
        {
            Color3f f = isect.bsdf->f(wo, wi);
            float lMaterialPDF = isect.bsdf->Pdf(wo, wi);
            if(pdf <= 0.f || lMaterialPDF <= 0.f)
            {
                Li = Color3f(0.f);
            }
            else
            {
                // W0 : Wj = 1 : nf, ng.
                float lightPDFWeight = PowerHeuristic(1, pdf, 1, lMaterialPDF);
                // float lightPDFWeight = 1.f;
                // std::cout << "lightPDFWeight:" << lightPDFWeight << std::endl;
                L += f * Li * AbsDot(wi, n) * lightPDFWeight / (pdf / nLights);
            }
        }
    }
    // ************************************************* //

    // BRDF PDF sampling:
    Vector3f mWiB(0.f);
    Color3f mLi(0.f);
    float mPDFMaterial = 0.f;
    BxDFType flags;
    Color3f mF = isect.bsdf->Sample_f(wo, &mWiB, sampler->Get2D(), &mPDFMaterial, BSDF_ALL, &flags);
    if(!IsBlack(mF) && mPDFMaterial != 0.f)
    {
        Ray tempRay = isect.SpawnRay(mWiB);
        Intersection insectBRDF;
        if(scene.Intersect(tempRay, &insectBRDF))
        {
            if(insectBRDF.objectHit->GetAreaLight() == light.get())
            {
                const AreaLight* arealightPtr = insectBRDF.objectHit->GetAreaLight();
                float pdfLWB = arealightPtr->Pdf_Li(isect, mWiB);
                float wWiB = PowerHeuristic(1, mPDFMaterial, 1, pdfLWB);
                mLi = mF * Li * AbsDot(mWiB, n) * wWiB / (pdfLWB / nLights);
                L += mLi;
            }
        }
    }

    return L;
}

Color3f FullLightingIntegrator::Li(const Ray &ray, const Scene &scene, std::shared_ptr<Sampler> sampler, int depth) const
{
    // Instantiate an accumlated ray color that begins as black;
    Color3f L(0.f);
    // Instantiate an accumulated ray throughput color that begins as white;
    // The throughput will be used to determine when your ray path terminates via the Russian Roulette heuristic.
    Color3f beta(1.f);

    Ray rayPath(ray);

    bool specBounce = false;

    int bounceCounter = depth;

    while(bounceCounter > 0)
    {
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

        Color3f LTerm = DirectLight(rayPath, scene, sampler);
        L += (beta * LTerm);

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



    return L;
}
