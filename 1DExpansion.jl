# To set up environment run
#
# using Pkg
# Pkg.activate(".")
# Pkg.add(["ITensors","ITensorMPS","Compat","OrdinaryDiffEqTsit5",
#        "Graphs","NamedGraphs","SpecialFunctions",
#        "Observers","KrylovKit","Plots","DataFrames","HDF5"])
#

using ITensors
using ITensorMPS
using KrylovKit: exponentiate
using OrdinaryDiffEqTsit5: ODEProblem, Tsit5, solve
using Graphs, NamedGraphs
using SpecialFunctions: erf
using Observers
using Plots
using DataFrames
using HDF5
using LinearAlgebra

"""
    well(numsites;V,L,s)

Creates a potential well, of dept `V`, width `L`, whose edge has sharpness `s`.
"""
function well(numsites;V,L,s)
    if s==0
        return [j>L ? V : 0 for j in 1:numsites]
    end
    return [(V/2)*(1+tanh((j-L)/s)) for j in 1:numsites]
end

"""
    tdvpexp(;num,trap,U,time_step,T,maxn=5)

First uses `dmrg` to find the ground-state of the BHM in the given trap, with `num` particles, interaction `U`, and maximum number of particles per site `maxn`.  It then uses `tdvp` to calculate the free expansion when the trap is turned off.

Optional arguments and their defaults:
- `g=1` or `t` -- hopping strength -- can also be specified with either symbol
- `η` -- alternative way to specify interactions `U`
- `verbose=false` -- verbosity setting of main loop
- `outputlevel=0` -- verbosity setting of `dmrg` and `tdvp`
- `redirectstdout` -- a string which given will redirect the output of `dmrg` and `tdvp` to that file.  Really useful if you use `outputlevel=1` or `outputlevel=2` inside a notebook
- `dg=0` or `dt` -- amplitude of oscillations of hopping
- `dU=0` or `dη` -- amplitude of oscillations of interactions
- `ω=0` -- frequency of oscillations of hopping/interactions
- `tdvpopts=(;updater=krylov_updater,updater_kwargs=(;tol=1e-8,eager=true),cutoff=1e-6,maxdim=200,nsite=2,outputlevel=outputlevel)
- `dmrgopts=(;nsweeps=100,maxdim=100,cutoff=1E-6,outputlevel=outputlevel)`
"""
function tdvpexp(;num,trap,g=1,dg=0,U=0,dU=0,ω=0,
        t=nothing,
        dt=nothing,
        η=nothing, dη=nothing,
        time_step,T,
        maxn=5,etol=1e-8,verbose=false,
        outputlevel=0,
        tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=200,
            nsite=2,outputlevel=outputlevel
            ),
        dmrgopts=(;nsweeps=100,maxdim=100,cutoff=1E-6,outputlevel=outputlevel),
        redirectstdout=nothing
    )
    # enable calling with different notations
        if t!=nothing ; g=t ; end
        if dt!=nothing ; dg=dt ; end
        if η!=nothing ; U=η ; end
        if dη!=nothing ; dg=dη ; end
    if verbose
        println("finding ground state with $num particles in a system of length $(length(trap))")
        println("max occupation per site $maxn, hopping g=$g, interactions U=$U")
        if redirectstdout != nothing
            println("redirecting stdout to $redirectstdout")
        end
        flush(stdout)
    end
    tick=time()
    if redirectstdout==nothing
        gs=findgs(num,trap;t=g,U,maxn,etol,dmrgopts)
    else
        open(redirectstdout,"w") do io
            redirect_stdout(io) do
                gs=findgs(num,trap;t=g,U,maxn,etol,dmrgopts)
            end
        end
    end
    ψ=gs.ψ
    E=gs.E
    tock=time()
    if verbose
        println("Found ground state in wall time $(tock-tick) s.  Energy=$E")
        println()
        println("Calculating time evolution")
        if dg==0 && dU==0
            println("Expanding with time independent Hamiltonian")
        else
            println("Expanding with oscillating Hamiltonian,dg=$dg, dU=$dU")
        end
        if redirectstdout != nothing
            println("redirecting stdout to $redirectstdout")
        end
        flush(stdout)
    end
    if redirectstdout==nothing
        ex=expand(ψ;g,η=U,T,time_step,tdvpopts)
    else
        open(redirectstdout,"a") do io
            redirect_stdout(io) do
                ex=expand(ψ;g,η=U,T,time_step,tdvpopts)
            end
        end
    end
    tick=time()
    if verbose
        println("completed time evolution in wall time $(tick-tock) s")
    end
    return ex
end

"""
    prodtdvpexp(;num,U,time_step,T,maxn=5)

Starts with a product state where num particles are at the left.  It then uses `tdvp` to calculate the free expansion.

Optional arguments and their defaults:
- `g=1` or `t` -- hopping strength -- can also be specified with either symbol
- `η` -- alternative way to specify interactions `U`
- `verbose=false` -- verbosity setting of main loop
- `outputlevel=0` -- verbosity setting of `dmrg` and `tdvp`
- `redirectstdout` -- a string which given will redirect the output of `dmrg` and `tdvp` to that file.  Really useful if you use `outputlevel=1` or `outputlevel=2` inside a notebook
- `dg=0` or `dt` -- amplitude of oscillations of hopping
- `dU=0` or `dη` -- amplitude of oscillations of interactions
- `ω=0` -- frequency of oscillations of hopping/interactions
- `tdvpopts=(;updater=krylov_updater,updater_kwargs=(;tol=1e-8,eager=true),cutoff=1e-6,maxdim=200,nsite=2,outputlevel=outputlevel)
- `dmrgopts=(;nsweeps=100,maxdim=100,cutoff=1E-6,outputlevel=outputlevel)`
"""
function prodtdvpexp(;num,Lx=2*num,g=1,dg=0,U=0,dU=0,ω=0,
        t=nothing,
        dt=nothing,
        η=nothing, dη=nothing,
        time_step,T,
        maxn=5,etol=1e-8,verbose=false,
        outputlevel=0,maxdim=200,
        tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=maxdim,
            nsite=2,outputlevel=outputlevel
            ),
        dmrgopts=(;nsweeps=100,maxdim=100,cutoff=1E-6,outputlevel=outputlevel),
        redirectstdout=nothing
    )
    # enable calling with different notations
        if t!=nothing ; g=t ; end
        if dt!=nothing ; dg=dt ; end
        if η!=nothing ; U=η ; end
        if dη!=nothing ; dg=dη ; end
    if verbose
        println("starting with product state of $num particles in a system of length $Lx")
        println("max occupation per site $maxn")
        flush(stdout)
    end
    ψ,sites=Grid1D(Lx; d=maxn+1, Pn=num)
    tock=time()
    if verbose
        println()
        println("Calculating time evolution")
        if dg==0 && dU==0
            println("Expanding with time independent Hamiltonian")
        else
            println("Expanding with oscillating Hamiltonian,dg=$dg, dU=$dU")
        end
        if redirectstdout != nothing
            println("redirecting stdout to $redirectstdout")
        end
        flush(stdout)
    end
    if redirectstdout==nothing
        ex=expand(ψ;g,η=U,T,time_step,tdvpopts)
    else
        open(redirectstdout,"w") do io
            redirect_stdout(io) do
                ex=rawexpand(ψ;g,η=U,T,time_step,tdvpopts)
            end
        end
    end
    tick=time()
    if verbose
        println("completed time evolution in wall time $(tick-tock) s")
    end
    return ex
end

"""
    saveocs(data,filename)
    saveoccs(data,filename)

Saves the output of the tdvp run in a `hdf5` file that can be openned in Python
"""
saveocs(wrapper,filename)=saveocs(wrapper.obs,filename)
function saveocs(obs::DataFrame,filename)
    h5open(filename, "w") do file
        file["times"] = obs.times
        file["sites"] = collect(osites(obs))
        file["densities"] = odensities(obs)
        file["probs"] = stack(obs.probs)
    end
    return filename
end

# keep spelling errors loose
saveoccs=saveocs

odensities(wrapper)=odensities(wrapper.obs)
odensities(obs::DataFrame)=stack([density(p) for p in obs[:,"probs"]])

otimes(wrapper)=otimes(wrapper.obs)
otimes(obs::DataFrame)=obs.times

osites(wrapper)=osites(wrapper.obs)
osites(obs::DataFrame)=1:size(obs[1,"probs"],1)

density(probs)=[sum((j-1)*p for (j,p) in enumerate(row)) for row in eachrow(probs)]

"""
    findgs(num,trap;t=1,U=0,maxn=2,etol=1e-8,
        dmrgopts=(nsweeps=100,maxdim=100,cutoff=1E-6,outputlevel=0))

Runs dmrg to find the ground state of `num` particles in a trap given by the vector of number `trap`.  To get verbose output change
`outputlevel` in `dmrgopts` from `0` to either `1` or `2`.
"""
function findgs(num,trap;t=1,U=0,maxn=2,etol=1e-8,
        dmrgopts=(nsweeps=100,maxdim=100,cutoff=1E-6,outputlevel=0))
    d=maxn+1
    Lx=length(trap)
    ψ,sites=Grid1D(Lx; d, Pn=num)
    Hop=BoseHubbardModel(t, U, Lx,V=trap)
    HMPO=MPO(Hop,sites)
    observer=EnergyObserver(etol)
    E,ψ=dmrg(HMPO,ψ;dmrgopts...,observer)
    return (;ψ,E,observer)
end

"""expand(ψ₀;g,η,time_step,T)

Runs TDVP to look at free-expansion of wavefunction `ψ₀`.  Here `g=t` is the hopping matrix element at `η=U` is the interaction strength.

Optional named arguments (with defaults):
    dg=0,dη=0,ω=0
    tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=200,
            nsite=2,outputlevel=1
            )

These allow you to make the hopping and the interaction vary sinusoidally:

    g(t)=g+dg sin(ω t)
    η(t)=η+dη sin(ω t)
"""
function expand(ψ₀;g,dg=0,η,dη=0,ω=0,time_step,T,
    tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=200,
            nsite=2,outputlevel=1
            ))
    Lx=length(ψ₀)
    sites=siteinds(ψ₀)
    K=MPO(Hopping(Lx),sites)
    U=MPO(Interactions(Lx),sites)
    H=TimeDependentSum([t->-im*(g+dg*sin(ω*t)),t->-im*(η+dη*sin(ω*t))],[K,U])
    obs=p_observer()
    println("calling tdvp")
    ψ = tdvp(H,T,ψ₀;tdvpopts...,time_step,(step_observer!)=obs)
    return (;ψ,obs)
end     

## Utilities for data analysis
function fwd(density)
    m=maximum(d)
    a=linextrap(d,3*m/4)
    b=linextrap(d,m/4)
    return b-a
end   

"""rawexpand(ψ₀;g,η,time_step,T)

Runs TDVP to look at free-expansion of wavefunction `ψ₀`.  Here `g=t` is the hopping matrix element at `η=U` is the interaction strength.

Same as `expand` -- but uses a different KE operator

Optional named arguments (with defaults):
    dg=0,dη=0,ω=0
    tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=200,
            nsite=2,outputlevel=1
            )

These allow you to make the hopping and the interaction vary sinusoidally:

    g(t)=g+dg sin(ω t)
    η(t)=η+dη sin(ω t)
"""
function rawexpand(ψ₀;g,dg=0,η,dη=0,ω=0,time_step,T,
    tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=200,
            nsite=2,outputlevel=1
            ))
    Lx=length(ψ₀)
    sites=siteinds(ψ₀)
    K=MPO(rawHopping(Lx),sites)
    U=MPO(Interactions(Lx),sites)
    H=TimeDependentSum([t->-im*(g+dg*sin(ω*t)),t->-im*(η+dη*sin(ω*t))],[K,U])
    obs=p_observer()
    println("calling tdvp")
    ψ = tdvp(H,T,ψ₀;tdvpopts...,time_step,(step_observer!)=obs)
    return (;ψ,obs)
end     

## Utilities for data analysis
function fwd(d)
    m,j=findmax(d)
    a=linextrap(d,3*m/4;start=j)
    b=linextrap(d,m/4;start=j)
    return b-a
end   

"""
    fw(probs)

gives the width of the density to go from 1/4 to 3/4 of its maximum value
"""
function fw(probs)
    d=density(probs)
    m=maximum(d)
    a=linextrap(d,3*m/4)
    b=linextrap(d,m/4)
    return b-a
end    

"""
    hwhm(probs)

Gives the half-width-half max width of the cloud"""
function hwhm(probs)
    d=density(probs)
    m=maximum(d)
    j=findfirst(x->x<m/2,d)
    dj=d[j]
    i=j-1
    di=d[i]
    return ((m/2)*(j-i)+i*dj-j*di)/(dj-di)
end

"""
    width(probs)

Given a Lxn matrix probs, calculates the first spatial moment
"""
width(probs)=sum((j[2]-1)*(j[1]-1)*p for (j,p) in pairs(probs))/
    sum((j[2]-1)*p for (j,p) in pairs(probs))

function linextrap(vals,target;start=1)
    values=vals[start:end]
    j=findfirst(x->x<target,values)
    vj=values[j]
    i=j-1
    vi=values[i]
    return (target*(j-i)+i*vj-j*vi)/(vj-vi)
end


### Functions which are stolen from ITensorTDVP.jl
### Easier to just define them in main name-space
### than to futz with exporting, etc...
###
### I don't bother documenting these as they are stolen
###
# TimeDependentSum
struct TimeDependentSum{C,T}
    coefficients::C
    terms::T
end
coefficients(e::TimeDependentSum) = e.coefficients
terms(e::TimeDependentSum) = e.terms
Base.copy(e::TimeDependentSum) = TimeDependentSum(coefficients(e), copy.(terms(e)))
function Base.:*(c::Number, e::TimeDependentSum)
    return TimeDependentSum(map(f -> (t -> c * f(t)), coefficients(e)), terms(e))
end
Base.:*(e::TimeDependentSum, c::Number) = c * e
(e::TimeDependentSum)(t::Number) = ScaledSum(map(f -> f(t), coefficients(e)), terms(e))

ITensorMPS.reduced_operator(op::TimeDependentSum) =
    TimeDependentSum(coefficients(op), ITensorMPS.reduced_operator.(terms(op)))
ITensorMPS.set_nsite!(op::TimeDependentSum, nsite) =
    (foreach(t -> ITensorMPS.set_nsite!(t, nsite), terms(op)); op)
ITensorMPS.position!(op::TimeDependentSum, state, pos) =
    (foreach(t -> ITensorMPS.position!(t, state, pos), terms(op)); op)

# ── ScaledSum ─────────────────────────────────────────────────────────────────

struct ScaledSum{C,T}
    coefficients::C
    terms::T
end
coefficients(e::ScaledSum) = e.coefficients
terms(e::ScaledSum) = e.terms

function _apply(e::ScaledSum, x)
    return mapreduce(+, zip(coefficients(e), terms(e))) do (c, H)
        c * H(x)
    end
end
(e::ScaledSum)(x) = _apply(e, x)
(e::ScaledSum)(x::ITensor) = permute(_apply(e, x), inds(x))

# ── Qudit operators ───────────────────────────────────────────────────────────

function ITensors.op(::OpName"ad", ::SiteType"Qudit", d::Int)
    m = zeros(d, d)
    for i in 1:d-1; m[i+1, i] = √i; end
    return m
end
function ITensors.op(::OpName"a", ::SiteType"Qudit", d::Int)
    m = zeros(d, d)
    for i in 1:d-1; m[i, i+1] = √i; end
    return m
end
function ITensors.op(::OpName"adadaa", ::SiteType"Qudit", d::Int)
    m = zeros(d, d)
    for i in 1:d; m[i, i] = Float64((i-1)*(i-2)); end
    return m
end
function ITensors.op(::OpName"n", ::SiteType"Qudit", d::Int)
    m = zeros(d, d)
    for i in 1:d; m[i, i] = Float64(i-1); end
    return m
end
function ITensors.op(::OpName"nd", ::SiteType"Qudit", d::Int)
    m = zeros(d, d); m[3, 3] = 1.0; return m
end
####
####

### functions which are used to set up MPS
"""
    Grid1D(Lx::Int; d=3, Pn=Lx)

Returns a tuple (ψ, sites)
-- `ψ` is a MPS representing a Bose-Hubbard state with `Lx` sites, max occupation of `nmax=d-1`, and `Pn` particles, where those particles are 1 per site on the left-most sites
-- sites is the list of index objects for the sites.  (Redundant, can get it via `siteinds(ψ)`)
"""
function Grid1D(Lx::Int; d=3, Pn=Lx)
    Nsites = Lx 
    sites = siteinds("Qudit", Nsites; dim=d, conserve_number=true, qnname_number="n")
    bond = Index(QN("n", 0) => 1, tags="Link")
    bonds = [settags(bond, "Link"*string(i)) for i in 1:Nsites]
    ψ = MPS(sites)
    ψ[1] = onehot(sites[1] => 2, bonds[1] => 1); Pn -= 1
    for i in 2:Nsites-1
        ψ[i] = Pn == 0 ? onehot(sites[i] => 1, dag(bonds[i-1]) => 1, bonds[i] => 1) :
                         onehot(sites[i] => 2, dag(bonds[i-1]) => 1, bonds[i] => 1)
        Pn != 0 && (Pn -= 1)
    end
    ψ[Nsites] = Pn == 0 ? onehot(sites[Nsites] => 1, dag(bonds[Nsites-1]) => 1) :
                           onehot(sites[Nsites] => 2, dag(bonds[Nsites-1]) => 1)
    return ψ, sites
end

## OpSum's
"""
    BoseHubbardModel(t, U, len;V=nothing)

Gives an OpSum object representing the Bose-Hubbard Hamiltonian.  The optional named argument `V` is a list that is a local potential

I chose to put a potential dip on the first site so that in the non-interacting case there is an anti-node at the left.
"""
function BoseHubbardModel(t, U, len;V=nothing)
    H = OpSum()
    if !iszero(t)
        H += -t, "n", 1
        for j in 1:len-1
            H += -t, "ad", j, "a", j+1
            H += -t, "ad", j+1, "a", j
        end
    end
    for vertex in 1:len
        V!=nothing && !iszero(V[vertex]) && (H += (V[vertex], "n", vertex))
        !iszero(U) && (H += (U/2, "adadaa", vertex))
    end
    return H
end

"""
    Hopping(len)

Gives an OpSum object representing the Kinetic Energy operator for the Bose Hubard Model.

I chose to put a potential dip on the first site so that in the non-interacting case there is an anti-node at the left.
"""
function Hopping(len)
    H = OpSum()
    H += -1, "n", 1
    for j in 1:len-1
            H +=  -1,"ad", j, "a", j+1
            H +=  -1,"ad", j+1, "a", j
    end
    return H
end

"""
    rawHopping(len)

Gives an OpSum object representing the Kinetic Energy operator for the Bose Hubard Model.

"""
function rawHopping(len)
    H = OpSum()
    for j in 1:len-1
            H +=  -1,"ad", j, "a", j+1
            H +=  -1,"ad", j+1, "a", j
    end
    return H
end

"""
    Interactions(len)

Gives an OpSum object representing the Interaction Energy operator for the Bose Hubard Model.
"""
function Interactions(len)
    H = OpSum()
    for vertex in 1:len
        H += 1/2, "adadaa", vertex
    end
    return H
end

### Updaters for TDVP
### -- Stolen from tdvp.jl
function ode_updater(operator, init; internal_kwargs, alg=Tsit5(), kwargs...)
    (; current_time, time_step) = (; current_time=zero(Bool), internal_kwargs...)
    time_span = typeof(time_step).((current_time, current_time + time_step))
    init_vec, to_itensor = to_vec(init)
    f(x::ITensor, ::Any, t) = operator(t)(x)
    f(v::AbstractArray, p, t) = to_vec(f(to_itensor(v), p, t))[1]
    sol = solve(ODEProblem(f, init_vec, time_span), alg; kwargs...)
    return to_itensor(sol.u[end]), (;)
end

function krylov_updater(operator, init; internal_kwargs, kwargs...)
    (; current_time, time_step) = (; current_time=zero(Bool), internal_kwargs...)
    state, info = exponentiate(operator(current_time), time_step, init; kwargs...)
    return state, (; info)
end

"""
    p_observer()

generates an `Observer` object which is used in the TDVP function to record the site occupations as a function of time
"""
p_observer() = observer(
    "steps"  => (; sweep)        -> sweep,
    "times"  => (; current_time) -> current_time,
    "probs" => (; state)        -> probs(state),
)

### My utility functions
"""
    probs(psi::MPS; sites = 1:length(psi))

Given a MPS `psi`, returns a 2D array of the probabilities of each local state.  Optional named argument `sites` selects which sites to include.

For example, suppose you have a spin-1/2 chain in an antiferromagnetic state.  This would then return the matrix
1 0
0 1
1 0
0 1
"""
function probs(psi::MPS; sites = 1:length(psi))
    psi = copy(psi)
    N = length(psi)
    ElT = ITensorMPS.scalartype(psi)
    s = siteinds(psi)

    site_range = (sites isa AbstractRange) ? sites : collect(sites)
    Ns = length(site_range)
    start_site = first(site_range)

    psi = orthogonalize(psi, start_site)
    norm2_psi = norm(psi)^2
    iszero(norm2_psi) && error("MPS has zero norm in function `probs`")

    maxdim = maximum(dim(s) for s in siteinds(psi))
    result= zeros(Float64,Ns,maxdim)
    
    for (entry, j) in enumerate(site_range)
        psi = orthogonalize(psi, j)
        ind=siteind(psi,j)
        A=psi[j]
        for (n,sv) in enumerate(ind)
            pA=(A*dag(onehot(sv)))
            prob=norm(pA)^2/norm2_psi
            result[entry,n]=prob
        end
    end

    return result
end

###
### Finding Ground State
###

"""
    EnergyObserver(energy_tol=0.0)

Creates an `Observer` that can be used by `ITensors` `dmrg` function.  Every sweep it checks the energy.  If the energy difference drops below `energy_tol` it stops the algorithm.

This observer also stores `last_energy`, `sweep`, and `energylist`.  The last can be used to see how the energy evolves from iteration to iteration.
"""
mutable struct EnergyObserver <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64
    sweep::Int64
    energylist :: Vector{Float64}

   EnergyObserver(energy_tol=0.0) = new(energy_tol,1000.0,0,[])
end

function ITensorMPS.checkdone!(o::EnergyObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  out=kwargs[:outputlevel]
  o.sweep= sw
  push!(o.energylist,energy)
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    if out>0
        println("Stopping DMRG after sweep $sw")
    end
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end



        