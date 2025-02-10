export AbstractSpectralDiscretization,
    TwoBandSpectralDiscretization, HyperspectralDiscretization

abstract type AbstractSpectralDiscretization{FT} end

"""
    SolarIrradianceCurve(λ::FT) where {FT}

Function to approximate the solar irradiance curve at a given wavelength. The
solar irradiance curve gives the intensity of solar radiation over the spectrum.
"""
function SolarIrradianceCurve(λ::FT) where {FT <: AbstractFloat}
    return FT(
        (2 * 6.26e-34 * 3e8 / λ^5) *
        (1 / (exp(6.26e-34 * 3e8 / (λ * 1.381e-23 * 5778)) - 1)),
    )
end

"""
    SolarIrradianceArea(λ1::FT, λ2::FT) where{FT}

Quickly computes the area under the solar irradiance curve between two
wavelengths using a step size of 1 nm for trapezoidal integration.
"""
function SolarIrradianceArea(λ1::FT, λ2::FT) where {FT <: AbstractFloat}
    # Create an mvector of the steps
    num_steps = Int(round((λ2 - λ1) * 1e9))
    # Compute the area via the trapezoid method
    return FT(
        sum(SolarIrradianceCurve.(range(λ1, stop = λ2, length = num_steps))) *
        (λ2 - λ1) / num_steps,
    )
end

"""
    TwoBandSpectralDiscretization{FT <: AbstractFloat}

A type used to represent the discretization of the shortwave solar irradiance
spectrum into two bands. Typically used for a PAR band and an NIR band.
"""
struct TwoBandSpectralDiscretization{FT <: AbstractFloat} <:
       AbstractSpectralDiscretization{FT}
    "Wavelength boundaries between each band in the discretization (m)"
    λ::NTuple{3, FT}
    "Banded irradiance curve for the solar radiation (W/m^2)"
    I::NTuple{3, FT}
    "Proportions of PAR radiation in each band"
    PAR_proportions::NTuple{3, FT}
end

"""
    function TwoBandSpectralDiscretization(λ::NTuple{2, FT}) where {FT}

Creates a two-band spectral discretization with the given wavelength boundaries.
Wavelength boundaries default to the PAR and NIR ranges.
"""
function TwoBandSpectralDiscretization(
    λ::NTuple{3, FT} = (400e-9, 700e-9, 2500e-9),
) where {FT <: AbstractFloat}
    irradiance_curve = ntuple(_ -> zero(FT), 3)
    PAR_prop = ntuple(_ -> zero(FT), 3)
    PAR_bounds = (400e-9, 700e-9)
    sum = 0
    for i in 1:3
        # Portion of incident radiation divided into this band
        irradiance_curve[i] = SolarIrradianceArea(λ[i], λ[i + 1])
        sum += irradiance_curve[i]
        # Portion of the band which is PAR
        if λ[i] >= PAR_bounds[1] && λ[i + 1] <= PAR_bounds[2]
            PAR_prop[i] = 1
        elseif λ[i] >= PAR_bounds[2] || λ[i + 1] <= PAR_bounds[1]
            PAR_prop[i] = 0
        elseif λ[i] <= PAR_bounds[1] && λ[i + 1] <= PAR_bounds[2]
            PAR_trap = SolarIrradianceArea(PAR_bounds[1], λ[i + 1])
            PAR_prop[i] = FT(PAR_trap / irradiance_curve[i])
        else
            PAR_trap = SolarIrradianceArea(λ[i], PAR_bounds[2])
            PAR_prop[i] = FT(PAR_trap / irradiance_curve[i])
        end
    end
    irradiance_curve = FT.(irradiance_curve ./ sum)
    return TwoBandSpectralDiscretization{FT}(λ, irradiance_curve, PAR_prop)
end

"""
    HyperspectralDiscretization{FT <: AbstractFloat}

A type used to represent the discretization of the shortwave solar irradiance
spectrum into 16 bands to match with ClimaAtmos.
"""
struct HyperspectralDiscretization{FT <: AbstractFloat} <:
       AbstractSpectralDiscretization{FT}
    "Wavelength boundaries between each band in the discretization (m)"
    λ::NTuple{17, FT}
    "Banded irradiance curve for the solar radiation (W/m^2)"
    I::NTuple{17, FT}
    "Proportion of PAR in each band"
    PAR_proportions::NTuple{17, FT}
end

"""
    function HyperspectralDiscretization(λ::NTuple{17, FT}) where {FT}

Creates a hyperspectral discretization with the given wavelength boundaries.
Wavelength boundaries default to even spacing over the full shortwave spectrum.
"""
function HyperspectralDiscretization(
    λ::NTuple{17, FT} = ntuple(i -> 100e-9 + (i - 1) * 150e-9, 17),
) where {FT}
    irradiance_curve = ntuple(_ -> zero(FT), 17)
    PAR_prop = ntuple(_ -> zero(FT), 17)
    PAR_bounds = (400e-9, 700e-9)
    sum = 0
    for i in 1:17
        # Portion of incident radiation divided into this band
        irradiance_curve[i] = SolarIrradianceArea(λ[i], λ[i + 1])
        sum += irradiance_curve[i]
        # Portion of the band which is PAR
        if λ[i] >= PAR_bounds[1] && λ[i + 1] <= PAR_bounds[2]
            PAR_prop[i] = 1
        elseif λ[i] >= PAR_bounds[2] || λ[i + 1] <= PAR_bounds[1]
            PAR_prop[i] = 0
        elseif λ[i] <= PAR_bounds[1] && λ[i + 1] <= PAR_bounds[2]
            PAR_trap = SolarIrradianceArea(PAR_bounds[1], λ[i + 1])
            PAR_prop[i] = FT(PAR_trap / irradiance_curve[i])
        else
            PAR_trap = SolarIrradianceArea(λ[i], PAR_bounds[2])
            PAR_prop[i] = FT(PAR_trap / irradiance_curve[i])
        end
    end
    irradiance_curve = FT.(irradiance_curve ./ sum)
    return HyperspectralDiscretization{FT}(λ, irradiance_curve, PAR_prop)
end
