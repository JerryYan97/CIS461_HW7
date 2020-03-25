#include "diffusearealight.h"

Color3f DiffuseAreaLight::L(const Intersection &isect, const Vector3f &w) const
{
    //TODO
    // return Color3f(0.f);
    return glm::dot(isect.normalGeometric, w) > 0.f ? emittedLight : Color3f(0.f);
}

Color3f DiffuseAreaLight::Sample_Li(const Intersection &ref, const Point2f &xi,
                                     Vector3f *wi, Float *pdf) const
{
    // Get an Intersection on the surface of its Shape by invoking shape->Sample.
    Intersection pShape = shape->Sample(ref, xi, pdf);

    // Check if the resultant PDF is zero or that the reference Intersection and the
    // resultant Intersection are the same point in space, and return black if this is the case.
    if(*pdf == 0.f || glm::length(pShape.point - ref.point) < FLT_EPSILON)
    {
        return Color3f(0.f);
    }

    // Set ωi to the normalized vector from the reference Intersection's
    // point to the Shape's intersection point.
    *wi = glm::normalize(pShape.point - ref.point);

    // std::cout << emittedLight.x << " " << emittedLight.y << " " << emittedLight.z << std::endl;

    // Return the light emitted along ωi from our intersection point.
    return this->L(pShape, -*wi);
}

float DiffuseAreaLight::Pdf_Li(const Intersection &ref, const Vector3f &wi) const
{
    // Intersect sample ray with area light geometry:
    Ray ray = ref.SpawnRay(wi);
    // float tHit;
    Intersection isectLight;
    if(!shape->Intersect(ray, &isectLight))
    {
        return 0.f;
    }

    // Convert light sample weight to solid angle measure:
    float pdf = DistanceSquared(ref.point, isectLight.point) / (AbsDot(isectLight.normalGeometric, -wi) * shape->Area());

    return pdf;
}
