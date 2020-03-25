#ifndef SPECTRUM_H
#define SPECTRUM_H
#include "globals.h"

enum class SpectrumType { Reflectance, Illuminant };

static const int sampledLambdaStart = 400;
static const int sampledLambdaEnd = 700;
static const int nSpectralSamples = 60;

// Spectral Data Declarations
static const int nCIESamples = 471;
extern const Float CIE_X[nCIESamples];
extern const Float CIE_Y[nCIESamples];
extern const Float CIE_Z[nCIESamples];
extern const Float CIE_lambda[nCIESamples];
static const Float CIE_Y_integral = 106.856895;
static const int nRGB2SpectSamples = 32;
extern const Float RGB2SpectLambda[nRGB2SpectSamples];
extern const Float RGBRefl2SpectWhite[nRGB2SpectSamples];
extern const Float RGBRefl2SpectCyan[nRGB2SpectSamples];
extern const Float RGBRefl2SpectMagenta[nRGB2SpectSamples];
extern const Float RGBRefl2SpectYellow[nRGB2SpectSamples];
extern const Float RGBRefl2SpectRed[nRGB2SpectSamples];
extern const Float RGBRefl2SpectGreen[nRGB2SpectSamples];
extern const Float RGBRefl2SpectBlue[nRGB2SpectSamples];
extern const Float RGBIllum2SpectWhite[nRGB2SpectSamples];
extern const Float RGBIllum2SpectCyan[nRGB2SpectSamples];
extern const Float RGBIllum2SpectMagenta[nRGB2SpectSamples];
extern const Float RGBIllum2SpectYellow[nRGB2SpectSamples];
extern const Float RGBIllum2SpectRed[nRGB2SpectSamples];
extern const Float RGBIllum2SpectGreen[nRGB2SpectSamples];
extern const Float RGBIllum2SpectBlue[nRGB2SpectSamples];

extern bool SpectrumSamplesSorted(const Float *lambda, const Float *vals, int n);
extern void SortSpectrumSamples(Float *lambda, Float *vals, int n);
extern Float AverageSpectrumSamples(const Float *lambda, const Float *vals, int n, Float lambdaStart, Float lambdaEnd);


inline void XYZToRGB(const Float xyz[3], Float rgb[3]) {
    rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
    rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
    rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
}

inline void RGBToXYZ(const Float rgb[3], Float xyz[3]) {
    xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
    xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
    xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
}

template <int nSpectrumSamples>
class CoefficientSpectrum {
public:
    CoefficientSpectrum(Float v = 0.f) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] = v;
    }
#ifdef DEBUG
    CoefficientSpectrum(const CoefficientSpectrum &s) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] = s.c[i];
    }

    CoefficientSpectrum &operator=(const CoefficientSpectrum &s) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] = s.c[i];
        return *this;
    }
#endif // DEBUG
    void Print(FILE *f) const {
        fprintf(f, "[ ");
        for (int i = 0; i < nSpectrumSamples; ++i) {
            fprintf(f, "%f", c[i]);
            if (i != nSpectrumSamples-1) fprintf(f, ", ");
        }
        fprintf(f, "]");
    }
    CoefficientSpectrum &operator+=(const CoefficientSpectrum &s2) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] += s2.c[i];
        return *this;
    }
    CoefficientSpectrum operator+(const CoefficientSpectrum &s2) const {
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] += s2.c[i];
        return ret;
    }
    CoefficientSpectrum operator-(const CoefficientSpectrum &s2) const {
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] -= s2.c[i];
        return ret;
    }
    CoefficientSpectrum operator/(const CoefficientSpectrum &s2) const {
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] /= s2.c[i];
        return ret;
    }
    CoefficientSpectrum operator*(const CoefficientSpectrum &sp) const {
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] *= sp.c[i];
        return ret;
    }
    CoefficientSpectrum &operator*=(const CoefficientSpectrum &sp) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] *= sp.c[i];
        return *this;
    }
    CoefficientSpectrum operator*(Float a) const {
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] *= a;
        return ret;
    }
    CoefficientSpectrum &operator*=(Float a) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] *= a;
        return *this;
    }
    friend inline
    CoefficientSpectrum operator*(Float a, const CoefficientSpectrum &s) {
        return s * a;
    }
    CoefficientSpectrum operator/(Float a) const {
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] /= a;
        Assert(!ret.HasNaNs());
        return ret;
    }
    CoefficientSpectrum &operator/=(Float a) {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] /= a;
        return *this;
    }
    bool operator==(const CoefficientSpectrum &sp) const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != sp.c[i]) return false;
        return true;
    }
    bool operator!=(const CoefficientSpectrum &sp) const {
        return !(*this == sp);
    }
    bool IsBlack() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != 0.) return false;
        return true;
    }
    friend CoefficientSpectrum Sqrt(const CoefficientSpectrum &s) {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = std::sqrt(s.c[i]);
        return ret;
    }
    template <int n> friend inline CoefficientSpectrum<n> Pow(const CoefficientSpectrum<n> &s, Float e);
    CoefficientSpectrum operator-() const {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = -c[i];
        return ret;
    }
    friend CoefficientSpectrum Exp(const CoefficientSpectrum &s) {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = std::exp(s.c[i]);
        return ret;
    }
    friend std::ostream& operator<<(std::ostream& os, const CoefficientSpectrum &s) {
        os << "[";
        for (int i = 0; i < nSpectrumSamples; ++i) {
            os << s.c[i];
            if (i + 1 < nSpectrumSamples)
                os << ", ";
        }
        os << "]";
        return os;
    }
    CoefficientSpectrum Clamp(Float low = 0, Float high = FLT_MAX) const {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = glm::clamp(c[i], low, high);
        return ret;
    }
    bool HasNaNs() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (std::isnan(c[i])) return true;
        return false;
    }
    bool Write(FILE *f) const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (fprintf(f, "%f ", c[i]) < 0) return false;
        return true;
    }
    bool Read(FILE *f) {
        for (int i = 0; i < nSpectrumSamples; ++i) {
            double v;
            if (fscanf(f, "%lf ", &v) != 1) return false;
            c[i] = v;
        }
        return true;
    }
    Float &operator[](int i) {
        return c[i];
    }
    Float operator[](int i) const {
        return c[i];
    }
    static const int nSamples = nSpectrumSamples;
protected:
    float c[nSpectrumSamples];
};



class SampledSpectrum : public CoefficientSpectrum<nSpectralSamples>{
public:
    SampledSpectrum(Float v = 0.f) : CoefficientSpectrum(v) { }
    SampledSpectrum(const CoefficientSpectrum<nSpectralSamples> &v)
        : CoefficientSpectrum<nSpectralSamples>(v) { }
    static SampledSpectrum FromSampled(const Float *lambda, const Float *v, int n)
    {
        if (!SpectrumSamplesSorted(lambda, v, n)) {
            std::vector<Float> slambda(&lambda[0], &lambda[n]);
            std::vector<Float> sv(&v[0], &v[n]);
            SortSpectrumSamples(&slambda[0], &sv[0], n);
            return FromSampled(&slambda[0], &sv[0], n);
        }

        SampledSpectrum r;
        for (int i = 0; i < nSpectralSamples; ++i) {
            // Compute average value of given SPD over $i$th sample's range
            Float lambda0 = Lerp(Float(i) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
            Float lambda1 = Lerp(Float(i + 1) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
            r.c[i] = AverageSpectrumSamples(lambda, v, n, lambda0, lambda1);
        }
        return r;
    }

    static void Init() {
            // Compute XYZ matching functions for _SampledSpectrum_
            for (int i = 0; i < nSpectralSamples; ++i) {
                Float wl0 = Lerp(Float(i) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
                Float wl1 = Lerp(Float(i + 1) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
                X.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_X, nCIESamples, wl0,
                                                wl1);
                Y.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Y, nCIESamples, wl0,
                                                wl1);
                Z.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Z, nCIESamples, wl0,
                                                wl1);
            }

            // Compute RGB to spectrum functions for _SampledSpectrum_
            for (int i = 0; i < nSpectralSamples; ++i) {
                Float wl0 = Lerp(Float(i) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
                Float wl1 = Lerp(Float(i + 1) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
                rgbRefl2SpectWhite.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectWhite,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectCyan.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectCyan,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectMagenta.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectMagenta,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectYellow.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectYellow,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectRed.c[i] = AverageSpectrumSamples(
                    RGB2SpectLambda, RGBRefl2SpectRed, nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectGreen.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectGreen,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectBlue.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectBlue,
                                           nRGB2SpectSamples, wl0, wl1);

                rgbIllum2SpectWhite.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectWhite,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectCyan.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectCyan,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectMagenta.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectMagenta,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectYellow.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectYellow,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectRed.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectRed,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectGreen.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectGreen,
                                           nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectBlue.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectBlue,
                                           nRGB2SpectSamples, wl0, wl1);
            }
        }

    Float y() const {
        Float yy = 0.f;
        for (int i = 0; i < nSpectralSamples; ++i)
            yy += Y.c[i] * c[i];
        return yy * Float(sampledLambdaEnd - sampledLambdaStart) / Float(nSpectralSamples);
    }

    void ToXYZ(Float xyz[3]) const {
        xyz[0] = xyz[1] = xyz[2] = 0.f;
        for (int i = 0; i < nSpectralSamples; ++i) {
            xyz[0] += X.c[i] * c[i];
            xyz[1] += Y.c[i] * c[i];
            xyz[2] += Z.c[i] * c[i];
        }
        Float scale = Float(sampledLambdaEnd - sampledLambdaStart) /
                Float(CIE_Y_integral * nSpectralSamples);
        xyz[0] *= scale;
        xyz[1] *= scale;
        xyz[2] *= scale;
    }

    void ToRGB(Color3f& iColor) const {
        Float rgb[3];
        Float xyz[3];
        ToXYZ(xyz);
        XYZToRGB(xyz, rgb);
        iColor.r = rgb[0];
        iColor.g = rgb[1];
        iColor.b = rgb[2];
    }
    static SampledSpectrum FromRGB(const Color3f iColor, SpectrumType type = SpectrumType::Illuminant);

private:
    // SampledSpectrum Private Data
    static SampledSpectrum X, Y, Z;
    static SampledSpectrum rgbRefl2SpectWhite, rgbRefl2SpectCyan;
    static SampledSpectrum rgbRefl2SpectMagenta, rgbRefl2SpectYellow;
    static SampledSpectrum rgbRefl2SpectRed, rgbRefl2SpectGreen;
    static SampledSpectrum rgbRefl2SpectBlue;
    static SampledSpectrum rgbIllum2SpectWhite, rgbIllum2SpectCyan;
    static SampledSpectrum rgbIllum2SpectMagenta, rgbIllum2SpectYellow;
    static SampledSpectrum rgbIllum2SpectRed, rgbIllum2SpectGreen;
    static SampledSpectrum rgbIllum2SpectBlue;

};
typedef SampledSpectrum Spectrum;
#endif // SPECTRUM_H
