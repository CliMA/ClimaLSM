import SciMLBase
import ClimaComms
ClimaComms.@import_required_backends
import ClimaTimeSteppers as CTS
using ClimaCore
using ClimaLand
using ClimaLand.Domains: Column
using ClimaLand.Soil
using ClimaLand.Soil.DisorderedKinetics
using Dates

import ClimaLand.Parameters as LP

# Define simulation times
t0 = Float64(0);
tf = Float64(100);
dt = Float64(0.01);

FT = Float64;

model_params = DisorderedKineticsModelParameters{FT}(
    mu = FT(-1.0),
    sigma = FT(2.0));
zmax = FT(0);
zmin = FT(-1);
nelems = 1;
# we use a vertical column with one layer to in the future interface with other variables like temperature and moisture that are depth dependent
lsm_domain = Column(; zlim = (zmin, zmax), nelements = nelems);

# Make biogeochemistry model args
NPP = PrescribedSOCInputs{FT}(TimeVaryingInput((t) -> 500));
model_sources = (LitterInput{FT}(),);

model = DisorderedKineticsModel{FT}(;
    parameters=model_params,
    npools=100,
    domain=lsm_domain,
    sources=model_sources,
    drivers=NPP
);

Y, p, coords = initialize(model);
set_initial_cache! = make_set_initial_cache(model);




function init_model!(Y, p, model)
    N = NTuple{model.npools, FT}
    function set_k(k::N) where {N}
        k = collect(exp.(range(model.parameters.mu-8*model.parameters.sigma,model.parameters.mu+8*model.parameters.sigma,length=model.npools)));
        return ntuple(x->k[x],model.npools)
    end
    p.soilC.ks .= set_k.(p.soilC.ks);
end


init_model!(Y, p, model);

set_initial_cache!(p, Y, t0);
disordered_kinetics_exp_tendency! = make_exp_tendency(model);

timestepper = CTS.RK4();
ode_algo = CTS.ExplicitAlgorithm(timestepper);

saveat = collect(t0:FT(10 * dt):tf);
sv = (;
    t = Array{Float64}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
);
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat);
# updateat = deepcopy(saveat)
# drivers = ClimaLand.get_drivers(model)
# updatefunc = ClimaLand.make_update_drivers(drivers)
# driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
# cb = SciMLBase.CallbackSet(driver_cb, saving_cb)

prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(T_exp! = disordered_kinetics_exp_tendency!),
    Y,
    (t0, tf),
    p,
)
# sol = SciMLBase.solve(prob, ode_algo; dt = dt, callback = cb)
sol = SciMLBase.solve(prob, ode_algo; dt = dt, callback = saving_cb);

# Check that simulation still has correct float type
@assert eltype(sol.u[end].soilC) == FT;


