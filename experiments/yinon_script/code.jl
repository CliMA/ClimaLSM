# parameters: mu and sigma (mean and std distributions)
# auxiliary: k is an array of decay rate (constant)
# state variable: u (array of C with different decay rate)
# input: input of carbon per area per unit time
# we prescribe input for now, but it should be given by another
# module prognostically, e.g., litter fall etc.

using QuadGK
using DifferentialEquations
using StaticArrays
using Distributions
using DataFrames
using CSV
using BenchmarkTools
# integral, error = quadgk(x -> cos(200x), 0, 1)

# function dc_kt_dt(u, p, t)#, fixed=true)
#     k,mu,sigma,input = p
#     if fixed==true
#         return SA[input*pdf(LogNormal(mu,sigma),k) - k*u]
#     else
#         ind = min(int(t+1),length(input))
#         return SA[input[ind]*pdf(LogNormal(mu,sigma),k) - k*u]
#     end


# end

function dc_kt_dt(u, p, t)#, fixed=true)
    k,mu,sigma,input,fixed = p
    if fixed== true
        SA[input.*pdf(LogNormal(mu,sigma),k) .- k.*u[1]]
    else
        ind = min(floor(Int,t+1),length(input))
        return SA[input[ind]*pdf(LogNormal(mu,sigma),k) - k*u[1]]
    end

    # SA[p[4].*pdf(LogNormal(p[2],p[3]),p[1]) .- p[1].*u[1]]
end
# prob = ODEProblem(dc_kt_dt, SA[0.], (0.,100.),[0.1,1.,2.,0.25,true])
# solve(prob)


function solve_ode(k,tau,age,input,fixed=true)
    sigma = sqrt(log(age/tau))
    mu = - log(sqrt(tau^3/age));
    ps = (k,mu,sigma,input,fixed)
    tspan = (0.,100.)
    u0 = SA[0.]
    prob = ODEProblem(dc_kt_dt, u0,tspan , ps)
    solve(prob,saveat=1)
end
function run_diskin(tau,age,input,fixed=true,discrete=false)
    if discrete==true
        krange = exp.(range(-10,stop=10,length=100000));
        dk = diff(vcat(0,krange));
        res = sum(reduce(hcat,[solve_ode(k,tau,age,input,fixed).u.*d for (k,d) in zip(krange,dk)]),dims=2);
    else
        res, error = quadgk(x -> solve_ode(x,tau,age,input,fixed), 0, Inf)
    end
    res
end
NPP = Array(CSV.read("experiments/yinon_script/test_NPP.csv",DataFrame));

cont = run_diskin(17,1000,0.25,true,false);
@btime noncont = run_diskin(17,1000,0.25,true,true);
cont'./noncont
