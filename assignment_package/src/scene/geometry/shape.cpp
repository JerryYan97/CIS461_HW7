#include "shape.h"
#include <QDateTime>

pcg32 Shape::colorRNG = pcg32(QDateTime::currentMSecsSinceEpoch());


void Shape::InitializeIntersection(Intersection *isect, float t, Point3f pLocal) const
{
    isect->point = Point3f(transform.T() * glm::vec4(pLocal, 1));
    ComputeTBN(pLocal, &(isect->normalGeometric), &(isect->tangent), &(isect->bitangent));
    isect->uv = GetUVCoordinates(pLocal);
    isect->t = t;
}

// you must implement in the base Shape class is the one that takes in three inputs rather than two.
// This function must invoke the two-input Sample,
// to be implemented by subclasses, and generate a ωi from the resulting Intersection.

// it must convert the PDF obtained from the other Sample function from a PDF with respect to surface area to a
// PDF with respect to solid angle at the reference point of intersection. As always, be careful of dividing by zero!
// Should this occur, you should set your PDF to zero.



Intersection Shape::Sample(const Intersection &ref, const Point2f &xi, float *pdf) const
{
    //TODO
    // Get an intersection point on the shape.
    Intersection isect = this->Sample(xi, pdf);
    Vector3f wi = glm::normalize(isect.point - ref.point);

    // Convert PDF to solid angle PDF at the reference point.
    *pdf = DistanceSquared(ref.point, isect.point) /
            (AbsDot(isect.normalGeometric, -wi) * this->Area());

    // float pdfRefPoint = ref.bsdf->Pdf();
    return isect;
}
