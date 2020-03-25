#include "naiveintegrator.h"
#include "spectrum.h"

Color3f NaiveIntegrator::Li(const Ray &ray, const Scene &scene, std::shared_ptr<Sampler> sampler, int depth) const
{
    //TODO
    Color3f L(0.f), beta(1.f);
    Ray tempRay(ray);
    bool specularBounce = false;
    Spectrum LSpectrum(0.f);
    Spectrum betaSpectrum(1.f);

    for(int bounces = 0;; bounces++)
    {
        // Find next path vertex and accumulate contribution:

        // Intersect ray with scene and store intersection in isect:
        Intersection isect;
        bool foundIntersection = scene.Intersect(tempRay, &isect);

        // Possibly add emitted light at intersection:

        if(foundIntersection)
        {
            Color3f tempLightCol = isect.Le(-tempRay.direction);
            Spectrum tempLightColSpectrum = Spectrum::FromRGB(tempLightCol);
            if(tempLightCol.x != 0.f && tempLightCol.y != 0.f && tempLightCol.z != 0.f)
            {
                L += beta * tempLightCol;
                LSpectrum += betaSpectrum * tempLightColSpectrum;
                break;
            }
        }

        // Terminate path if ray escaped or maxDepth was reached:
        if(!foundIntersection || bounces >= recursionLimit)
        {
            break;
        }

        // Compute scattering functions and skip over medium boundaries:
        if(!isect.ProduceBSDF())
        {
            break;
        }

        // Sample illumination from lights to find path contribution:

        // Sample BSDF to get new path direction:
        Vector3f wo = -tempRay.direction;
        Vector3f wi;
        float pdf;
        BxDFType flags;
        Color3f f = isect.bsdf->Sample_f(wo, &wi, sampler->Get2D(), &pdf, BSDF_ALL, &flags);
        float colVal = f.x + f.y + f.z;
        if(colVal < FLT_EPSILON || pdf == 0.f)
        {
            break;
        }
        // The function is the color from a direction
        beta *= (f * glm::abs(glm::dot(wi, isect.bsdf->normal)) / pdf);
        // beta *= (f * glm::abs(glm::dot(wi, isect.normalGeometric)) / pdf);
        betaSpectrum *= Spectrum::FromRGB(f * glm::abs(glm::dot(wi, isect.bsdf->normal)) / pdf);
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        tempRay = isect.SpawnRay(wi);

        // Account for subsurface scattering, if applicable:

        // Possibly terminate the path with Russian roulette:
    }
    // Use this commented-out code to get result from Specutrm calculation.
    // LSpectrum.ToRGB(L);
    return L;
}
