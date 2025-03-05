ClimaLand.jl Release Notes
========================

main
-------


v0.15.10
--------
- use GL vs GLL quadrature; speeds up simulation 2x globally
  PR[#1051](https://github.com/CliMA/ClimaLand.jl/pull/1051)
- Use MODIS LAI by default in experiments and longruns
  PR[#973](https://github.com/CliMA/ClimaLand.jl/pull/973)
- Revert PR993 changes to runoff
  PR[#1021](https://github.com/CliMA/ClimaLand.jl/pull/1021)
- Add ITime (see [ClimaUtilities documentation](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/) for more information)
  PR[#1030](https://github.com/CliMA/ClimaLand.jl/pull/1030)

v0.15.9
-------
- Remove `ImplicitEquationJacobian` and use
  `ClimaCore.MatrixFields.FieldMatrixWithSolver` instead
  PR[#996](https://github.com/CliMA/ClimaLand.jl/pull/996)
- Add global seasonal plots for long runs
  PR[#923](https://github.com/CliMA/ClimaLand.jl/pull/923)
- Various snow model changes
  PR[#965](https://github.com/CliMA/ClimaLand.jl/pull/965)
  PR[#989](https://github.com/CliMA/ClimaLand.jl/pull/989)
  PR[#988](https://github.com/CliMA/ClimaLand.jl/pull/988)

v0.15.8
-------
- Add `soilco2.C` to `:short` default diagnostics
  PR[#984](https://github.com/CliMA/ClimaLand.jl/pull/984)
- Add function to count NaNs in state variables
  PR[#970](https://github.com/CliMA/ClimaLand.jl/pull/970)
- Reduce allocations by updating functions to modify in-place
  PR[#967](https://github.com/CliMA/ClimaLand.jl/pull/967)
  PR[#978](https://github.com/CliMA/ClimaLand.jl/pull/978)
  PR[#981](https://github.com/CliMA/ClimaLand.jl/pull/981)
  PR[#980](https://github.com/CliMA/ClimaLand.jl/pull/980)
- Include freezing point depression in `EnergyHydrology`,
  and generate plots for paper.
  PR[#963](https://github.com/CliMA/ClimaLand.jl/pull/963)
- Rename `LandHydrologyModel` to `SoilSnowModel`
  PR[#968](https://github.com/CliMA/ClimaLand.jl/pull/968)


v0.15.7
-------
- Run unit tests on GPU, and update code for GPU compatibility
  (including workaround for ClimaCore type inference failure)
  PR[#739](https://github.com/CliMA/ClimaLand.jl/pull/739)
- Initialize communications contexts, enabling experiments
  to run with MPI.
  PR[#954](https://github.com/CliMA/ClimaLand.jl/pull/954)

v0.15.6
-------
- Add progress log for snow land longrun
  PR[#943](https://github.com/CliMA/ClimaLand.jl/pull/943)
- Add tutorial to check the analytic solution of phase change
  in soil
  PR[#940](https://github.com/CliMA/ClimaLand.jl/pull/940)
- Update soil water bc and snow conductivity to match paper.
  Infiltration/runoff are now calculated from precip, snow melt,
  and evaporation, and infiltration is the boundary condition.
  PR[#931](https://github.com/CliMA/ClimaLand.jl/pull/931)
- Don't include snow depth in surface height. This fixes a bug
  that was causing NaNs when snow depth was >10m.
  PR[#935](https://github.com/CliMA/ClimaLand.jl/pull/935)
- Add snowy land benchmark
  PR[#932](https://github.com/CliMA/ClimaLand.jl/pull/932)
- Use ClimaArtifacts Lehmann 2008 dataset for evaporation experiment
  PR[#929](https://github.com/CliMA/ClimaLand.jl/pull/929)
- Use 2008 ERA forcing instead of 2021 for longruns, benchmarks,
  and other forced runs
  PR[#920](https://github.com/CliMA/ClimaLand.jl/pull/920)
- Update deprecated function calls to reduce test warnings
  PR[#925](https://github.com/CliMA/ClimaLand.jl/pull/925)
- Use the TOPMODEL artifact from ClimaArtifacts
  PR[#928](https://github.com/CliMA/ClimaLand.jl/pull/928)

v0.15.5
-------
- Integrated land model with snow, soil, canopy
  PR[#834](https://github.com/CliMA/ClimaLand.jl/pull/834)

v0.15.4
-------
- The foliage clumping index of the radiative transfer model now varies spatially, using MODIS data
  PR[#863](https://github.com/CliMA/ClimaLand.jl/pull/863)
- All outputs directories use activelink, described [here](https://clima.github.io/ClimaUtilities.jl/dev/outputpathgenerator/)
  PR[#804](https://github.com/CliMA/ClimaLand.jl/pull/804)
- Add functions to save simulations to checkpoints and read them back.
  PR[#853](https://github.com/CliMA/ClimaLand.jl/pull/853)
- Add `start_date` to output NetCDF files.
  PR[#853](https://github.com/CliMA/ClimaLand.jl/pull/853)

v0.15.3
-------
- It is now possible to call `ClimaComms.context` and `ClimaComms.device` Models
  and Domains. This provides a unambiguous way to determine the context and
  device for general simulations.
  PR[#852](https://github.com/CliMA/ClimaLand.jl/pull/852)
- `prescribed_analytic_forcing` is now available to simplify setting up analytic experiments.
  PR[#870](https://github.com/CliMA/ClimaLand.jl/pull/870)

v0.15.2
--------
- Boundary fluxes are non-allocating
PR[#819](https://github.com/CliMA/ClimaLand.jl/pull/819)
- Artifacts for the bucket model are now automatically downloaded. PR [#820](https://github.com/CliMA/ClimaLand.jl/pull/820)

v0.15.1
--------
- Add a tutorial of a single site calibration of a perfect model PR[#621](https://github.com/CliMA/ClimaLand.jl/pull/621)
- Add a longrun simulation for the bucket PR[#807](https://github.com/CliMA/ClimaLand.jl/pull/807)
- Add ground heat flux to snow-soil model PR[#796](https://github.com/CliMA/ClimaLand.jl/pull/796)
- Add snow-soil model PR [#779](https://github.com/CliMA/ClimaLand.jl/pull/779)
- Step canopy temperature implicitly. PR [#675](https://github.com/CliMA/ClimaLand.jl/pull/675)

v0.15.0
--------
- Add regional simulation example PR [#757](https://github.com/CliMA/ClimaLand.jl/pull/757)
- Reduced number of dependencies, leading to faster instantiation and import time,
- Improved compatibility of ClimaLand with older versions of packages.
  PR[#749](https://github.com/CliMA/ClimaLand.jl/pull/749)
  PR[#748](https://github.com/CliMA/ClimaLand.jl/pull/748)
- Added more inputs varying in space (MedlynConductance and Vcmax).
  PR[#759](https://github.com/CliMA/ClimaLand.jl/pull/759)
- Added a regional run example.
  PR[#757](https://github.com/CliMA/ClimaLand.jl/pull/757)
- ![breaking change][badge-💥breaking] Extend photosynthesis mechanism parameter to support fields.
PR[#774](https://github.com/CliMA/ClimaLand.jl/pull/774)
  - C3/C4 structs are removed. Now C3 is represented by a float of 1.0 and C4 by a float of 0.0

v0.14.3
--------
- Add support for regional simulations, box runs centered around a given
  longitude and latitude.
  PR [#710](https://github.com/CliMA/ClimaLand.jl/pull/710)
- Add support for SoilCanopyModel diagnostics
  PR [#699](https://github.com/CliMA/ClimaLand.jl/pull/699)

v0.14.1
--------
- Add simple model for single-column surface runoff
  PR[#702](https://github.com/CliMA/ClimaLand.jl/pull/702)

v0.14.0
--------
- Use the soil parameters in creating the biogeochemistry SoilMet driver for consistency.
  PR[#690](https://github.com/CliMA/ClimaLand.jl/pull/690)
- ![][badge-💥breaking] Generalize our forcing ``drivers" to include prescribed soil organic carbon
  PR[#692](https://github.com/CliMA/ClimaLand.jl/pull/692)
- Add a long run with diagnostics
  PR[#688](https://github.com/CliMA/ClimaLand.jl/pull/688)

v0.13.0
--------
- NOTE: the breaking PR below was merged by accident in v0.12.5
- ![][badge-💥breaking] rename update_jacobian to compute_jacobian
  and tendency_jacobian to jacobian
  PR[#685](https://github.com/CliMA/ClimaLand.jl/pull/685)

v0.12.5
--------
- Turn of dss
  PR[#650](https://github.com/CliMA/ClimaLand.jl/pull/650)
- Add ClimaCoupler downstream test
  PR [#680](https://github.com/CliMA/ClimaLand.jl/pull/680)
- Adds ClimaLand.Diagnostics
  PR [#628](https://github.com/CliMA/ClimaLand.jl/pull/628)
- Improve Performance of soil canopy model
  PR [#666](https://github.com/CliMA/ClimaLand.jl/pull/666)
  PR [#677](https://github.com/CliMA/ClimaLand.jl/pull/677)
- Add base global soil canopy run to benchmark and experiments
  PR [#591](https://github.com/CliMA/ClimaLand.jl/pull/591)
  PR [#669](https://github.com/CliMA/ClimaLand.jl/pull/669)
- updates Jacobian for soil energy
  PR[#678](https://github.com/CliMA/ClimaLand.jl/pull/678)

v0.12.4
--------
- Fix various canopy flux bugs
  PR [#641](https://github.com/CliMA/ClimaLand.jl/pull/641)
- Adds Richards to benchmark
  PR [#648](https://github.com/CliMA/ClimaLand.jl/pull/648)
- Reduce allocations in update runoffs
  PR [#664](https://github.com/CliMA/ClimaLand.jl/pull/664)
- Store z in cache instead of model
  PR [#658](https://github.com/CliMA/ClimaLand.jl/pull/658)

v0.12.3
--------
- Add benchmark pipeline
  PR [#592](https://github.com/CliMA/ClimaLand.jl/pull/592)
  PR [#642](https://github.com/CliMA/ClimaLand.jl/pull/642)
- Removed ArtifactWrappers
  PR [#627](https://github.com/CliMA/ClimaLand.jl/pull/627)
  PR [#540](https://github.com/CliMA/ClimaLand.jl/pull/640)
- Run RichardsModel on GPU
  PR [#638](https://github.com/CliMA/ClimaLand.jl/pull/638)

v0.12.2
--------
- Update implicit solver interface
  PR [#542](https://github.com/CliMA/ClimaLand.jl/pull/542)
- Add ClimaLand.Artifacts
  PR [#624](https://github.com/CliMA/ClimaLand.jl/pull/624)

v0.12.1
--------
- Add `regridder_type` option to albedo constructors.
  PR [#662](https://github.com/CliMA/ClimaLand.jl/pull/622)
- ![][badge-✨feature] Add simple snow model.
  PR [#147](https://github.com/CliMA/ClimaLand.jl/pull/147)
- ![][badge-✨feature] Implemented a simple sublimation model, which is now
  included in the soil model. PR [#373](https://github.com/CliMA/ClimaLand.jl/pull/373)

v0.12.0
--------
- Updated to ClimaComms 0.6 and ClimaCore 0.14. Now, `CUDA` and `MPI` are no
  longer automatically installed.

v0.11.2
--------
-  Add profiling of soil/canopy model to buildkite pipeline. This was the
   first global run of this model, and we also fixed dss! for tuple-valued fields,
   changed how we compute the zenith angle to allow for 2d fields, and changed
   instances of `sum` to a column integral.
   PR [#561](https://github.com/CliMA/ClimaLand.jl/pull/561)
- ![][badge-✨feature] Use
  [ClimaUtilities](https://github.com/CliMA/ClimaUtilities.jl) for `Space` and
  `Time` `VaryingInputs`. This adds support to non-conservative MPI/GPU
  compatible regridding of NetCDF input. PR
  [#560](https://github.com/CliMA/ClimaLand.jl/pull/560)
- Moved neural snow to extension, reducing latency by a factor 2. PR
  [#567](https://github.com/CliMA/ClimaLand.jl/pull/567)

v0.11.1
-------
- ![][badge-✨feature] Add option to profile albedo job. PR
  [#505](https://github.com/CliMA/ClimaLand.jl/pull/551)
- ![][badge-✨feature] ClimaLandSimulations: better plots (legend and style tweaks, added cumulative ET and P), unit tests, start and end time as an optional argument. PR [#494](https://github.com/CliMA/ClimaLand.jl/pull/494)
- ![][badge-🐛bugfix] ClimaLandSimulations: installation readme. PR [#494](https://github.com/CliMA/ClimaLand.jl/pull/494)

v0.11.0
-------
- Update SurfaceFluxes compat to include 0.11: PR [#548](https://github.com/CliMA/ClimaLand.jl/pull/548)
- Started changelog: PR [#547](https://github.com/CliMA/ClimaLand.jl/pull/547)
- ![][badge-🐛bugfix] Use ClimaCore v0.13.2, which includes remapping allocation fix for CPU case: PR [#546](https://github.com/CliMA/ClimaLand.jl/pull/546)
- ![][badge-💥breaking] Refactor BucketModelParameters, move some parameters to ClimaParams: PR [#507](https://github.com/CliMA/ClimaLand.jl/pull/507)
- ![][badge-✨feature] Add TOPMODEL runoff parameterization: PR [#511](https://github.com/CliMA/ClimaLand.jl/pull/511), Issue [#266](https://github.com/CliMA/ClimaLand.jl/issues/266)
- Add web dashboard for Fluxnet simulations: PR [#478](https://github.com/CliMA/ClimaLand.jl/pull/478)
- Use depot for buildkite pipeline: PR [#537](https://github.com/CliMA/ClimaLand.jl/pull/537)
- ![][badge-✨feature] Refactor EnergyHydrologyParameters, move some parameters to ClimaParams: PR [#505](https://github.com/CliMA/ClimaLand.jl/pull/505)
- Update to ClimaCore v0.13 (note: introduced CPU global bucket bug): PR [#536](https://github.com/CliMA/ClimaLand.jl/pull/536)
- Add ClimaLand logos: PR [#525](https://github.com/CliMA/ClimaLand.jl/pull/525)
- ![][badge-✨feature] Add infrastructure to use PFTs, add experiment running Ozark with PFTs: PR [#493](https://github.com/CliMA/ClimaLand.jl/pull/493)
- ![][badge-💥breaking] Refactor boundary condition types to unify types/structure: PR [#535](https://github.com/CliMA/ClimaLand.jl/pull/535)
- ![][badge-💥breaking] Update to ClimaParams v0.10, free Insolation and Thermodynamics, update SurfaceFluxes: PR [#508](https://github.com/CliMA/ClimaLand.jl/pull/508)
- ![][badge-✨feature] Add PrescribedPrecipitation driver to be used in global simulations: PR [#533](https://github.com/CliMA/ClimaLand.jl/pull/533)
- Switch to use new-central, pin Insolation (0.9.1) and Thermodynamics (0.12.3) versions: PR [#526](https://github.com/CliMA/ClimaLand.jl/pull/526)
- ![][badge-💥breaking] Refactor albedo parameterizations to have two albedo models, PrescribedBaregroundAlbedo and PrescribedSurfaceAlbedo: PR [#513](https://github.com/CliMA/ClimaLand.jl/pull/513)
- Add documentation of standalone models: PR [#474](https://github.com/CliMA/ClimaLand.jl/pull/474)

<!--
Contributors are welcome to begin the description of changelog items with badge(s) below. Here is a brief description of when to use badges for a particular pull request / set of changes:
 - 💥breaking - breaking changes. For example: removing deprecated functions/types, removing support for functionality, API changes, breaking changes in compats.
 - ✨feature - new feature added. For example: adding support for a new parameterization.
 - 🐛bugfix - bugfix. For example: fixing incorrect logic, resulting in incorrect results, or fixing code that otherwise might give a `MethodError`.
 - 🔥behavioralΔ - behavioral changes. For example: a new model is used, yielding more accurate results.
 - 🤖precisionΔ - machine-precision changes. For example, swapping the order of summed arguments can result in machine-precision changes.
 - 🚀performance - performance improvements. For example: improving type inference, reducing allocations, or code hoisting.
-->

[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
[badge-✨feature]: https://img.shields.io/badge/feature/enhancement-blue.svg
[badge-🐛bugfix]: https://img.shields.io/badge/🐛bugfix-purple.svg
[badge-🔥behavioralΔ]: https://img.shields.io/badge/🔥behavioralΔ-orange.svg
[badge-🤖precisionΔ]: https://img.shields.io/badge/🤖precisionΔ-black.svg
[badge-🚀performance]: https://img.shields.io/badge/🚀performance-green.svg
