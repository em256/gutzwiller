using ITensors
using ITensorMPS
using Compat: @compat
using KrylovKit: exponentiate
using OrdinaryDiffEqTsit5: ODEProblem, Tsit5, solve
using Graphs, NamedGraphs
using SpecialFunctions: erf
using Observers: observer

# ── TimeDependentSum ──────────────────────────────────────────────────────────
# Ported from archived ITensorTDVP.jl. Interface methods are extended into the
# ITensorMPS namespace so alternating_update picks them up over the identity fallback.

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

# ── Lattice ───────────────────────────────────────────────────────────────────

function Generate_Mapping(Lx::Int, Ly::Int)
    mapping = Graphs.grid((Ly, Lx))
    index_map = [(lx, ly) for lx in 1:Lx for ly in 1:Ly]
    return mapping, index_map
end

function Generate_Lattice(Lx::Int, Ly::Int; d=3, Pn=Lx*Ly)
    Nsites = Lx * Ly
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

# ── Hamiltonians ──────────────────────────────────────────────────────────────

function BoseHubbardModel(t::Float64, mu::Vector{Float64}, U::Float64, mapping)
    H = OpSum()
    if !iszero(t)
        for edge in edges(mapping)
            H += t, "ad", src(edge), "a", dst(edge)
            H += t, "ad", dst(edge), "a", src(edge)
        end
    end
    for vertex in vertices(mapping)
        !iszero(mu[vertex]) && (H += mu[vertex], "n", vertex)
        H += U/2, "adadaa", vertex
    end
    return H
end

function Phonon_MPO(A::Float64, k::Int, Lx::Int, index_map, mapping)
    P = OpSum()
    if !iszero(A)
        for vertex in vertices(mapping)
            x = index_map[vertex][1]
            P += A * cos(π*k*(x-1)/(Lx-1)), "n", vertex
        end
    end
    return P
end

function Higgs_MPO(A::Float64, mapping)
    K = OpSum()
    if !iszero(A)
        for edge in edges(mapping)
            K += A, "ad", dst(edge), "a", src(edge)
            K += A, "ad", src(edge), "a", dst(edge)
        end
    end
    return K
end

# ── Updaters ──────────────────────────────────────────────────────────────────

function ode_updater(operator, init; internal_kwargs, alg=Tsit5(), kwargs...)
    @compat (; current_time, time_step) = (; current_time=zero(Bool), internal_kwargs...)
    time_span = typeof(time_step).((current_time, current_time + time_step))
    init_vec, to_itensor = to_vec(init)
    f(x::ITensor, ::Any, t) = operator(t)(x)
    f(v::AbstractArray, p, t) = to_vec(f(to_itensor(v), p, t))[1]
    sol = solve(ODEProblem(f, init_vec, time_span), alg; kwargs...)
    return to_itensor(sol.u[end]), (;)
end

function krylov_updater(operator, init; internal_kwargs, kwargs...)
    @compat (; current_time, time_step) = (; current_time=zero(Bool), internal_kwargs...)
    state, info = exponentiate(operator(current_time), time_step, init; kwargs...)
    return state, (; info)
end

# ── Envelope functions ────────────────────────────────────────────────────────

function gaussian_ramp_up(t; t_up)
    σ = t_up / 6
    return 0.5 * (1 + erf((t - 3σ) / (√2 * σ)))
end
gaussian_pulse(t; τ₀, τₚ) = 0.5 * (erf(t - τ₀) + erf(τₚ - (t - τ₀)))

# ── Shared helpers ────────────────────────────────────────────────────────────

make_observer() = observer(
    "steps"  => (; sweep)        -> sweep,
    "times"  => (; current_time) -> current_time,
    "states" => (; state)        -> state,
)

function _run_tdvp(H_td, ψ₀, time_stop, time_step;
                   updater=krylov_updater, tol=1e-8,
                   cutoff=1e-6, maxdim=200, nsite=2, outputlevel=2)
    obs = make_observer()
    ψ = tdvp(H_td, time_stop, ψ₀;
        updater,
        updater_kwargs=(; tol, eager=true),
        time_step,
        cutoff,
        (step_observer!)=obs,
        maxdim,
        nsite,
        outputlevel,
    )
    return ψ, obs
end

function _bhm_mpos(g::Float64, W::Vector{Float64}, η::Float64,
                   Lx::Int, Ly::Int, mapping, sites)
    Ht = MPO(BoseHubbardModel(g,   zeros(Lx*Ly), 0.0, mapping), sites)
    Hμ = MPO(BoseHubbardModel(0.0, W,            0.0, mapping), sites)
    Hη = MPO(BoseHubbardModel(0.0, zeros(Lx*Ly), η,   mapping), sites)
    return Ht, Hμ, Hη
end

# ── Evolution functions ───────────────────────────────────────────────────────

function TimeEvolveTDVP_BHM(Lx::Int, Ly::Int, g::Float64, η::Float64, W::Array{Float64},
                              ψ₀::MPS, time_step::Float64, time_stop::Float64;
                              ramp_type="Gaussian", kwargs...)
    mapping, _ = Generate_Mapping(Lx, Ly)
    Ht, Hμ, Hη = _bhm_mpos(g, W, η, Lx, Ly, mapping, siteinds(ψ₀))
    ramp_fn = ramp_type == "Gaussian" ?
        (let t_up=time_stop; t -> gaussian_ramp_up(t; t_up) end) :
        (let t_up=time_stop; t -> t / t_up end)
    H_td = -im * 2π * TimeDependentSum([ramp_fn, t -> 1.0, t -> 1.0], [Ht, Hμ, Hη])
    return _run_tdvp(H_td, ψ₀, time_stop, time_step; kwargs...)
end

function TimeEvolveTDVP_FE(Lx::Int, Ly::Int, g::Float64, η::Float64, W::Array{Float64},
                             ψ₀::MPS, time_step::Float64, time_stop::Float64; kwargs...)
    mapping, _ = Generate_Mapping(Lx, Ly)
    Ht, Hμ, Hη = _bhm_mpos(g, W, η, Lx, Ly, mapping, siteinds(ψ₀))
    H_td = -im * 2π * TimeDependentSum([t -> 1.0, t -> 1.0, t -> 1.0], [Ht, Hμ, Hη])
    return _run_tdvp(H_td, ψ₀, time_stop, time_step; kwargs...)
end

function TimeEvolveTDVP_Spectroscopy(Lx::Int, Ly::Int, g::Float64, dg::Float64, f::Float64,
                                      η::Float64, W::Array{Float64}, ψ₀::MPS,
                                      time_step::Float64, time_stop::Float64; kwargs...)
    mapping, _ = Generate_Mapping(Lx, Ly)
    Ht, Hμ, Hη = _bhm_mpos(g, W, η, Lx, Ly, mapping, siteinds(ψ₀))
    p = dg / g
    drive_fn = let fd=f, p=p; t -> 1 + p * sin(sin(2π * fd * t)) end
    H_td = -im * 2π * TimeDependentSum([drive_fn, t -> 1.0, t -> 1.0], [Ht, Hμ, Hη])
    return _run_tdvp(H_td, ψ₀, time_stop, time_step; kwargs...)
end

function Phonon_Impulse_Spectroscopy(Lx::Int, Ly::Int, g::Float64, A::Float64,
                                      τ_pulse::Float64, τ_offset::Float64, η::Float64,
                                      W::Array{Float64}, ψ₀::MPS,
                                      time_step::Float64, time_stop::Float64; kwargs...)
    mapping, index_map = Generate_Mapping(Lx, Ly)
    H0 = MPO(BoseHubbardModel(g, W, η, mapping), siteinds(ψ₀))
    Ht = MPO(Phonon_MPO(A, 1, Lx, index_map, mapping), siteinds(ψ₀))
    pulse_fn = let τ₀=τ_offset, τₚ=τ_pulse; t -> gaussian_pulse(t; τ₀, τₚ) end
    H_td = -im * 2π * TimeDependentSum([t -> 1.0, pulse_fn], [H0, Ht])
    return _run_tdvp(H_td, ψ₀, time_stop, time_step; kwargs...)
end

function Higgs_Impulse_Spectroscopy(Lx::Int, Ly::Int, g::Float64, A::Float64,
                                     τ_pulse::Float64, τ_offset::Float64, η::Float64,
                                     W::Array{Float64}, ψ₀::MPS,
                                     time_step::Float64, time_stop::Float64; kwargs...)
    mapping, _ = Generate_Mapping(Lx, Ly)
    H0 = MPO(BoseHubbardModel(g, W, η, mapping), siteinds(ψ₀))
    Ht = MPO(Higgs_MPO(A, mapping), siteinds(ψ₀))
    pulse_fn = let τ₀=τ_offset, τₚ=τ_pulse; t -> gaussian_pulse(t; τ₀, τₚ) end
    H_td = -im * 2π * TimeDependentSum([t -> 1.0, pulse_fn], [H0, Ht])
    return _run_tdvp(H_td, ψ₀, time_stop, time_step; kwargs...)
end
