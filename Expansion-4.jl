# See Gutz2dExpansion.ipynb for the description of how this works.
# This is a visualization of the expansion of a 2D Bose-Hubbard system using the Gutzwiller approximation. 
#
# Sample usage: Gutz2DExpansionProduction4.ipynb
#
# Edited on July 28 2026 
#
# Version History:
# Notebooks: Gutz2DExpansion.ipynb, Gutz2DExpansion2.ipynb  (version 3 was a dead-end where I was trying things that did not work)
# Prior sample running of earlier versions: tstvz.ipynb
# Scripts: ExpansionVisualization-1.jl, ExpansionVisualization-2.jl, ExpansionVisualization-3.jl
#
# Current version supercedes all previous, so you will not likely need to look back at those.


using OffsetArrays
using LaTeXStrings
using LinearAlgebra
using Random
using GLMakie, Makie.Colors
using Observables

abstract type Gutz <:AbstractArray{ComplexF64,3} end

mutable struct Gutz2D <:Gutz
    data :: OffsetArray{ComplexF64, 3, Array{ComplexF64, 3}}
end

Gutz2D(Lx,Ly,maxN)=Gutz2D(OffsetArray(zeros(ComplexF64,2Lx+1,2Ly+1,maxN+1),(-Lx-1,-Ly-1,-1)))

Lx(f::Gutz2D)=axes(f.data,1)[end]
Lxmin(f::Gutz2D)=axes(f.data,1)[begin]
Ly(f::Gutz2D)=axes(f.data,2)[end]
Lymin(f::Gutz2D)=axes(f.data,2)[begin]
maxN(f::Gutz2D)=axes(f.data,3)[end]
minN(f::Gutz2D)=axes(f.data,3)[begin]

sites(f::Gutz2D)=CartesianIndices(axes(f.data)[1:2])
xv(f::Gutz2D)=sites(f)

Base.size(f::Gutz2D)=size(f.data)

@doc raw"""
    nv(f)

gives the values of the $n$ coordinates in $f(j, n)$
"""
nv(f::Gutz2D)=axes(f.data,3)
rightnv(f)=Iterators.drop(nv(f),1)


function Base.show(io::IO,f::Gutz2D)
    print(io,"G2D(x=$(Lxmin(f)):$(Lx(f)),y=$(Lymin(f)):$(Ly(f));maxN=$(maxN(f)))")
end

# function Base.show(io::IO,mime::MIME,f::Gutz2D)
#     println("generic ",mime)
#     print(io,"G2D")
#     #print(io,"G2D(x=$(Lxmin(f)):$(Lx(f)),y=$(Lymin(f)):$(Ly(f));maxN=$(maxN(f)))")
# end

function Base.show(io::IO,::MIME"text/plain",f::Gutz2D)
    print(io,"G2D(x=$(Lxmin(f)):$(Lx(f)),y=$(Lymin(f)):$(Ly(f));maxN=$(maxN(f)))")
end

# function Base.show(io::IO,::MIME"text/plain",f::Gutz2D)
#     println("custom markdown")
#     print(io,"G2D")
#     #print(io,"G2D(x=$(Lxmin(f)):$(Lx(f)),y=$(Lymin(f)):$(Ly(f));maxN=$(maxN(f)))")
# end

#pass-through functions for manipulating data
Base.getindex(f::Gutz,inds...)=f.data[inds...]
Base.setindex!(f::Gutz, args...)=setindex!(f.data,args...)

#slicing
sitewf(f::Gutz,siteinds::Int...)= f.data[siteinds...,:]
nslice(f::Gutz2D,n::Int)=f.data[:,:,n]

function psi(f::Gutz)
    nvals = rightnv(f) 
    xvals = xv(f) 
    result=[sum(sqrt(n)*f[j,n]*conj(f[j,n-1]) for n in nvals) for j in xvals]
end

@doc raw"""
    psi!(target,f)
Calculates the order parameter $\psi_j=\sum_n \sqrt{n} f_{j,n-1}^* f_{jn}$, and places
it in `target`.  
"""
function psi!(target,f)
    nvals = rightnv(f) 
    xvals=xv(f) 
    for j in xvals
        target[j]=sum(sqrt(n)*f[j,n]*conj(f[j,n-1]) for n in nvals)
    end
    target
end

@doc raw"""
    density(f)

calculates the density $\rho_j=\sum_n n |f_{j,n}|^2$.  It assumes that $f$ is a `OffsetArray` where the allowed values of `n` are
set by the values of the keys.

see density!(target,f).
"""
function density(f)
    nvals = nv(f) 
    xvals= xv(f) 
    result=[sum(n*abs(f[j,n])^2 for n in nvals) for j in xvals]
end

@doc raw"""
    density!(target,f)

calculates the density $\rho_j=\sum_n n |f_{j,n}|^2$, placing it in `target`.  
It assumes that $f$ is a `OffsetArray` where the allowed values of `n` are
set by the values of the keys.

see density(f).
"""
function density!(target,f)
    nvals = nv(f) 
    xvals= xv(f) 
    for j in xvals
        target[j]=sum(n*abs(f[j,n])^2 for n in nvals)
    end
    target
end

@doc raw"""
    normalize!(f)

rescales $f$ so that $\sum_n |f_{jn}|^2=1$ by taking $f_{jn}\to f_{jn}/\sqrt{\sum_m |f_{jm}|^2}$.

also see `standardize!` and `nproj!`
"""
function normalize!(f)
    nvals = nv(f) #axes(f,2)
    xvals= xv(f) #axes(f,1)
    for x in xvals
        n=norm(f[x,:])
        if n!=0
            f[x,:]/=n
        end
    end
    f
end

# Makes it so that the component labeled nbar has phase zero
@doc raw"""
    standardize!(f,n̄)

Rescales $f$ so that $\sum_n |f_{jn}|^2=1$ by taking $f_{jn}\to f_{jn}/\sqrt{\sum_m |f_{jm}|^2}$.
Also shifts the phase so that $f_{jn̄}$ is real.

also see `normalize!` and `nproj!`. 
"""
function standardize!(f, nbar)
    nvals = nv(f)
    xvals = xv(f)
    for x in xvals
        normsq = 0.0
        for s in nvals
            normsq += abs2(f[x, s])
        end
        n = sqrt(normsq)
        n == 0 && continue

        f1 = f[x, nbar]
        scale = abs(f1) > 0 ? n * (f1 / abs(f1)) : n
        invscale = inv(scale)
        for s in nvals
            f[x, s] *= invscale
        end
    end
    return f
end

Base.axes(f::Gutz2D)=axes(f.data)
Base.axes(f::Gutz2D,d::Int)=axes(f.data,d)
Base.similar(f::Gutz2D)=Gutz2D(similar(f.data))
Base.similar(f::Gutz2D,args...)=Gutz2D(similar(f.data,args...))
Base.:(+)(f::Gutz2D,g::Gutz2D)=Gutz2D(f.data+g.data)
Base.:(-)(f::Gutz2D,g::Gutz2D)=Gutz2D(f.data-g.data)
Base.:(*)(f::Gutz2D,v::Number)=Gutz2D(f.data*v)
Base.:(*)(v::Number,f::Gutz2D)=Gutz2D(v*f.data)
Base.:(/)(f::Gutz2D,v)=Gutz2D(f.data/v)
Base.length(f::Gutz2D)=length(f.data)


function setuniform!(f,vec)
    xvals =xv(f)
    for x in xvals
        f[x,:]=vec
    end
    return f
end


function setreal!(f)
    xvals =xv(f)
    for x in xvals
        f[x,:]=abs.(f[x,:])
    end
    return f
end

function setdisk!(f,radius,vec,default=nothing)
    xvals=xv(f)
    for x in xvals
        r= norm(x.I)
        if r<=radius
            f[x,:]=vec
        elseif !(default isa Nothing)
            f[x,:]=default
        end
    end
    return f
end

#rotate f so that the average occupation is given by targn
@doc raw"""
    nproj!(f,targn)
    nproj!(f,targn;nbar=nbar)

Projects $f$ into the physical space, where $\sum_n |f_{jn}|^2=1$
and the averag number of particles per site is `targn`.

If `nbar` is specified, then it also does a gauge transformation
so that the coefficient of $f_{j\bar n}$ is real, and
so that the order parameter is real on the central site.
"""
function nproj!(f,targn;threshold=1e-12,nbar=nothing)
    nvals = nv(f)
    xvals = xv(f)
    A = 0.0
    B = 0.0
    C = 0.0

    for x in xvals
        F0 = 0.0
        F1 = 0.0
        F2 = 0.0
        F3 = 0.0
        for n in nvals
            p = abs2(f[x,n])
            dn = n - targn
            F0 += p
            F1 += dn * p
            F2 += dn * dn * p
            F3 += dn * dn * dn * p
        end
        invF0 = inv(F0)
        A += F1 * invF0
        B += (F1 * F1 * invF0 - F2) / sqrt(F0)
        C += F3 - 3 * F2 * F1 * invF0 + 2 * F1 * F1 * F1 * invF0 * invF0
    end

    beta = 0.0
    if abs(B) > threshold
        beta = A / (2 * B) + (A * A * C) / ((2 * B)^3)
    end

    for x in xvals
        alpha = 1.0
        if abs(beta) > threshold
            F0 = 0.0
            F1 = 0.0
            F2 = 0.0
            for n in nvals
                p = abs2(f[x,n])
                dn = n - targn
                F0 += p
                F1 += dn * p
                F2 += dn * dn * p
            end
            discriminant = beta^2 * F1^2 + (1 - beta^2 * F2) * F0
            if discriminant < 0
                continue
            end
            alpha = (-beta * F1 + sqrt(discriminant)) / F0
        end

        normsq = 0.0
        for n in nvals
            c = alpha + beta * (n - targn)
            f[x,n] *= c
            normsq += abs2(f[x,n])
        end

        phasefactor = sqrt(normsq)
        if !(nbar isa Nothing)
            f1 = f[x,nbar]
            if abs(f1) > threshold
                phasefactor *= f1 / abs(f1)
            end
        end
        invphase = inv(phasefactor)
        for n in nvals
            f[x,n] *= invphase
        end
    end

    if !(nbar isa Nothing)
        setphase!(f,nbar)
    end
    return f
end

function Nproj!(f,targN;threshold=1e-12,nbar=nothing)
    numsites=length(xv(f))
    nproj!(f,targN/numsites;threshold,nbar)
end
    
# This is actually i df/dt, the way that I wrote it
function dfdt(f,p,t)
    g=p.g(t) #scalar g
    eta=p.eta(t) #scalar eta
    delta = p.delta(t) # list of deltas
    result=similar(f)
    _dfdt!(result,f,g,delta,eta) # function barrier
end

function dfdt!(result,f,p,t,scratch=nothing)
    g=p.g(t) #scalar g
    eta=p.eta(t) #scalar eta
    delta = p.delta(t) # list of deltas
    #@show g eta typeof(delta)
    _dfdt!(result,f,g,delta,eta,scratch) # function barrier
end

function _dfdt!(result,f,g,delta,eta,scratch)
    if scratch===nothing
        ψ=psi(f) 
    else
        ψ=scratch
        psi!(ψ,f)
    end
    ax=axes(ψ,1)
    ay=axes(ψ,2)
    xmin=ax[begin]
    xmax=ax[end]
    ymin=ay[begin]
    ymax=ay[end]
    nvals=nv(f)
    rnvals=Iterators.drop(nvals,1)
    for x in xmin:xmax,n in nvals
        result[x,ymin,n]=0.
        result[x,ymax,n]=0.
    end
    for y in ymin:ymax, n in nvals
        result[xmin,y,n]=0.
        result[xmax,y,n]=0.
    end
    for x in xmin+1:xmax-1, y in ymin+1:ymax-1
        gpsi=g*ψ[x-1,y]
        gpsi+=g*ψ[x+1,y]
        gpsi+=g*ψ[x,y-1]
        gpsi+=g*ψ[x,y+1]
        for n in nvals
            result[x,y,n]=f[x,y,n]*(eta*n*(n-1)/2+delta[x,y]*n)
        end
        for n in rnvals
            result[x,y,n]-= gpsi*sqrt(n)*f[x,y,n-1]
            result[x,y,n-1]-= conj(gpsi)*sqrt(n)*f[x,y,n]
        end
    end
    return result
end 

function rk4step(x,p,dxdt,t,deltat;prefactor=1.0,scratch=nothing)
    k1=dxdt(x,p,t,scratch)
    k2=dxdt(x+prefactor*k1*(deltat/2),p,t+deltat/2,scratch)
    k3=dxdt(x+prefactor*k2*(deltat/2),p,t+deltat/2,scratch)
    k4=dxdt(x+prefactor*k3*deltat,p,t+deltat,scratch)
    return prefactor * (k1+2*k2+2*k3+k4)*(deltat/6)
end

function rk4step!(x,p,dxdt!,t,deltat;result,k1,k2,k3,k4,prefactor=1.,temp,scratch=nothing)
    dxdt!(k1,x,p,t,scratch)
    temp .= x .+ (prefactor*deltat/2).*k1
    dxdt!(k2,temp,p,t+deltat/2,scratch)
    temp .= x .+ (prefactor*deltat/2).*k2
    dxdt!(k3,temp,p,t+deltat/2,scratch)
    temp .= x .+ (prefactor*deltat).*k3
    dxdt!(k4,temp,p,t+deltat,scratch)
    result .= (k1 .+ 2 .*k2 .+ 2 .*k3 .+k4).*(prefactor*deltat/6)
end

function gstep(f,p,dfdt,t,dt,targN,nbar)
    f=f+rk4step(f,p,dfdt,t,dt)
    Nproj!(f,targN;threshold=1e-12)
    return f
end

mutable struct OscillatingScalar{T}
    base :: T
    amp :: T
    ω :: Float64
end

(f :: OscillatingScalar)(t) = f.base+f.amp*sin(f.ω*t)

function relaxstep!(f :: Gutz2D,p,N,dt,df,k1,k2,k3,k4,f2,scratch=nothing)
    rk4step!(f,p,dfdt!,0.,dt;result=df,k1,k2,k3,k4,temp=f2,prefactor=-1.,scratch=scratch)
    f .+=df
    Nproj!(f,N)
    return f
end

function expandstep!(f ::Gutz2D,p,N,t,dt,df,k1,k2,k3,k4,f2,scratch=nothing)
     rk4step!(f,p,dfdt!,t,dt;result=df,k1,k2,k3,k4,prefactor=-1.0im,temp=f2,scratch=scratch)
     f .+=  df
     standardize!(f,0)
     Nproj!(f,N)
     return f
end

function oldsimpexpandstep!(f ::Gutz2D,p,N,t,dt,df,k1,k2,k3,k4,f2,scratch=nothing)
     rk4step!(f,p,dfdt!,t,dt;result=df,k1,k2,k3,k4,prefactor=-1.0im,temp=f2,scratch=scratch)
     f .+=  df
     return f
end


function reshape(f :: Gutz2D,lx::Int,ly::Int;default=nothing)
    mN=maxN(f)
    oldminLx=Lxmin(f)
    oldmaxLx=Lx(f)
    oldminLy=Lymin(f)
    oldmaxLy=Ly(f)
    newdata=OffsetArray(zeros(ComplexF64,2lx+1,2ly+1,mN+1),(-lx-1,-ly-1,-1))
    for x in -lx:lx, y in -ly:ly
        if oldminLx<=x<=oldmaxLx && oldminLy<=y<=oldmaxLy
            newdata[x,y,:]=f[x,y,:]
        elseif !(default isa Nothing)
            newdata[x,y,:]=default
        end
    end
    return Gutz2D(newdata)
end

function reshape!(f :: Gutz2D,lx::Int,ly::Int;default=nothing)
    g=reshape(f,lx,ly;default)
    f.data=g.data
    return f
end

mutable struct CBilliard{T}
    R ::Float64
    V ::T
end

CBilliard(R::Number,V::T) where T=CBilliard{T}(Float64(R),V)

Base.getindex(U::CBilliard,x,y)= x^2+y^2<=U.R^2 ? zero(U.V) : U.V

mutable struct Static{T}
    data :: T
end

# Make it so that if you evaluate at time t
# it just returns the same object
(U::Static)(t)=U

Base.getindex(U::Static,x,y) = getindex(U.data,x,y)


function harmonicpotential(L,R)
    OffsetArray(
        [(x^2+y^2)/(2R^2) for x in -L:L, 
                y in -L:L],
        (-L-1,-L-1))
end

function billiard(x,y,R,s,V)
    if s==0
        return x^2+y^2<=R^2 ? 0.0 : V
    end
    w=x^2+y^2-R^2
    return (tanh(w/s^2)+1)*V/2
end

function squarebilliard(x,y,R,s,V)
    if s==0
        return x^2<=R^2 && y^2<=R^2 ? 0.0 : V
    end
    w=max(x^2,y^2)-R^2
    return (tanh(w/s^2)+1)*V/2
end

billiardpotential(L,R,s,V) =
    OffsetArray([billiard(x,y,R,s,V) 
        for x in -L:L, y in -L:L],(-L-1,-L-1))

squarebilliardpotential(L,R,s,V) =
    OffsetArray([squarebilliard(x,y,R,s,V) 
        for x in -L:L, y in -L:L],(-L-1,-L-1))

zeropotential(L)=OffsetArray(zeros(2L+1,2L+1),(-L-1,-L-1))

mutable struct CSquareBilliard{T}
    R ::Float64
    V ::T
end

CSquareBilliard(R::Number,V::T) where T=CSquareBilliard{T}(Float64(R),V)

Base.getindex(U::CSquareBilliard,x,y)= x^2<=U.R^2 && y^2<=U.R^2 ? zero(U.V) : U.V

# Make it so that if you evaluate at time t
# it just returns the same object
(U::CSquareBilliard)(t)=U

mutable struct CScalar{T}
    V ::T
end

(U::CScalar)(t)=U.V

Base.getindex(U::CScalar,ind...)=U.V

mutable struct CAScalar{T}
    V ::T
end 

(U::CAScalar)(t)=U
Base.getindex(U::CAScalar,ind...)=U.V

mutable struct CHarmonic
    R ::Float64
end

Base.getindex(U::CHarmonic,x,y)=(x^2+y^2)/(2*U.R^2)

(U::CHarmonic)(t)=U

function validate_float(str)
    val = tryparse(Float64, str)
    if isnothing(val)
        return false
    else
        return true
    end
end

function validate_int(str)
    val = tryparse(Int, str)
    if isnothing(val)
        return false
    else
        return true
    end
end

function validate_posfloat(str)
    val = tryparse(Float64, str)
    if isnothing(val) || val<=0
        return false
    else
        return true
    end
end

function reset!(f:: Gutz2D,R,u0,v0=nothing)
    setdisk!(f,R,u0,v0)
    normalize!(f)
end
        
function expansionviz(;mN=5,V=10,namedopts...)
    # Make Figure
#mN=5
#V=10
 fig=Figure()
    display(fig)
    ax_main=Axis(fig[1,1][1,1])
    ax_slice=Axis(fig[1,1][2,1], yticklabelcolor = :blue,
    ylabel="ρ")
    ax_pot=Axis(fig[1,1][2,1], yticklabelcolor = :red, yaxisposition = :right,ylabel="V")
    hidespines!(ax_pot)
    hidexdecorations!(ax_pot)
    ax_integrated=Axis(fig[1,1][3,1], ylabel="column integrated ρ")
    controlpanel=fig[1,2][1,1]
    sliderpanel=fig[1,2][2,1]
    buttonpanel=fig[1,2][3,1]
    # Entry boxes
    Label(controlpanel[1,1],text="Main")
    maincontrols=controlpanel[2,1]
    Label(maincontrols[1,1],text="g=")
    Label(maincontrols[1,2],text="1")
    Label(maincontrols[1,3],text="η=")
    etabox=Textbox(maincontrols[1,4],reset_on_defocus=true,
        validator=validate_float,width=200,stored_string="1")
    η=@lift tryparse(Float64,$(etabox.stored_string))
    #Label(controlpanel[4,1],text="N=")
    #Nbox=Textbox(controlpanel[4,2],reset_on_defocus=true,
    #    validator=validate_float,width=200,stored_string="100")
    #N=@lift tryparse(Float64,$(Nbox.stored_string))
    secondcontrols=controlpanel[3,1]
    Label(secondcontrols[1,1],text="u0=")
    uboxes=Textbox[]
    for j in 0:mN
        box=Textbox(secondcontrols[1,2][1,j+1],
        reset_on_defocus=true,
        validator=validate_float,width=60,stored_string=
        j==1 ? "1" : "0")
        push!(uboxes,box)
    end
    ustrings = [u.stored_string for u in uboxes]
    u0 = lift((args...) -> [tryparse(Float64,u) for u in args], ustrings...)
    #u0 = @lift map(u -> tryparse(Float64, $(u.stored_string)), uboxes)
    #u0=@lift([tryparse(Float64,$u.stored_string) for u in uboxes])
    thirdcontrol=controlpanel[4,1]
    Label(thirdcontrol[1,1],text="L=")
    Lbox=Textbox(thirdcontrol[1,2],reset_on_defocus=true,
        validator=validate_int,width=200,stored_string="10")
    L=@lift tryparse(Int,$(Lbox.stored_string))
    N=Observable(0.)
    Nstring = lift(s->"$s",N)
    Label(thirdcontrol[1,3],text="N=")
    Nbox=Label(thirdcontrol[1,4],text=Nstring)
    Label(thirdcontrol[2,1],text="V=")
    Vbox=Textbox(thirdcontrol[2,2],reset_on_defocus=true,
        validator=validate_float,width=200,stored_string="10")
    V=@lift tryparse(Float64,$(Vbox.stored_string))
    oscpanel=controlpanel[5,1]
    Label(oscpanel[1,1][1,1],text="Oscillation")
    Label(oscpanel[1,1][1,2][1,1],text="ω=")
    omegabox=Textbox(oscpanel[1,1][1,2][1,2],reset_on_defocus=true,
        validator=validate_float,width=200,stored_string="1")
    ω=@lift tryparse(Float64,$(omegabox.stored_string))
    Label(oscpanel[2,1][1,1],text="δg=")
    dgbox=Textbox(oscpanel[2,1][1,2],reset_on_defocus=true,
        validator=validate_float,width=200,stored_string="0")
    dg=@lift tryparse(Float64,$(dgbox.stored_string))
    Label(oscpanel[2,1][1,3],text="δη=")
    detabox=Textbox(oscpanel[2,1][1,4],reset_on_defocus=true,
        validator=validate_float,width=200,stored_string="0")
    dη=@lift tryparse(Float64,$(detabox.stored_string))
    simpanel=controlpanel[6,1]
    Label(simpanel[1,1],text="Simulation")
    Label(simpanel[2,1][1,1],text="timestep=")
    dtbox=Textbox(simpanel[2,1][1,2],reset_on_defocus=true,
        validator=validate_posfloat,width=200,stored_string="0.01")
    dt=@lift tryparse(Float64,$(dtbox.stored_string))
    Label(simpanel[2,1][2,1],text="display timestep=")
    displaytbox=Textbox(simpanel[2,1][2,2],reset_on_defocus=true,
        validator=validate_posfloat,width=200,stored_string="0.01")
    displayt=@lift tryparse(Float64,$(displaytbox.stored_string))
    T=Observable("0.")
    Label(simpanel[2,1][3,1],text="t=")
    tbox=Label(simpanel[2,1][3,2],text=T)
    zero_time=Button(simpanel[2,1][3,3],label="zero")
    on(zero_time.clicks) do click 
        t[]=0.0
        T[]="0.0"
    end
    #
    # Sliders
    # Set it up as a slider grid, so we can add more sliders later
    sliders=SliderGrid(sliderpanel[1,1],
        (label="R",range=1:2L[],startvalue=L[],format="{:d}"),
    (label="s",range=0.0:(L[]/100):2L[],startvalue=0.0,format="{:.2f}")
    )
    Rslider=sliders.sliders[1]
    R=Rslider.value
    sslider=sliders.sliders[2]
    s=sslider.value
    on(L) do L
        Rslider.range=1:2*L
        sslider.range=0.0:L[]/100:2L
    end

    bp=@lift billiardpotential($L,$R,$s,$V)
    sbp=@lift squarebilliardpotential($L,$R,$s,$V)
    hp=@lift harmonicpotential($L,$R)
    zp=@lift zeropotential($L)
    
    #
    # Buttons
    reset_button=Button(buttonpanel[1,1][1,1];label="reset",tellwidth=false)
    Label(buttonpanel[1,2];text="autoscale")
autoscale_switch=Toggle(buttonpanel[1,3];active=true)
    as=autoscale_switch.active
    relax_button=Button(buttonpanel[2,1][1,1];label="box",tellwidth=false)
    square_button=Button(buttonpanel[2,1][1,2];label="square",tellwidth=false)
    harmonic_button=Button(buttonpanel[2,1][1,3];label="harmonic",tellwidth=false)
    expand_button=Button(buttonpanel[3,1];label="expand",tellwidth=false)
    pause_button=Button(buttonpanel[1,1][1,2];label="pause",tellwidth=false)
    recording=Observable(false)
    recordlabel=@lift $recording ? "pause record" : "      record"
    record_button=Button(buttonpanel[4,1][1,1];label=recordlabel)
    reset_record_button=Button(buttonpanel[4,1][1,2];label="reset record")
    wfs=timeseries(Float64[],Gutz2D[])
    on(record_button.clicks) do click
        recording[]=!(recording[])
    end
    on(reset_record_button.clicks) do click
        empty!(wfs.times)
        empty!(wfs.data)
    end
    # Setup
    f=Observable(Gutz2D(L[],L[],mN))
    v0=zeros(nv(f[]))
    v0[0]=1.
    df=Gutz2D(L[],L[],mN)
    k1=Gutz2D(L[],L[],mN)
    k2=Gutz2D(L[],L[],mN)
    k3=Gutz2D(L[],L[],mN)
    k4=Gutz2D(L[],L[],mN)
    f2=Gutz2D(L[],L[],mN)
    ψ=@lift psi($f)
    relaxp=@lift (g=CScalar(1.),eta=CScalar($η),
        #delta=CBilliard(Float64($R),$V)
        delta=Static($bp)
    )
    squarep=@lift (g=CScalar(1.),eta=CScalar($η),
        #delta=CSquareBilliard(Float64($R),$V)
        delta=Static($sbp)
        )    
    expandp=@lift (g=OscillatingScalar(1.,$dg,$ω),eta=OscillatingScalar($η,$dη,$ω),
        delta=Static($zp)
                    #delta=CAScalar(0.)
    )
    harmonicp=@lift (g=CScalar(1.),eta=CScalar($η),
        #delta=CHarmonic(Float64($R))
        delta=Static($hp)
    )
    isrunning_notifier = Condition()
    isrunning=Observable(false)
    relax=Observable(true)
    harmonic=Observable(false)
    square=Observable(false)

    pot=@lift if $relax
        $bp
    elseif $square
        $sbp
    elseif $harmonic
        $hp
    else
        $zp
    end
    
    
    # Figures
    #ρ=@lift density($f)
    # Alternatively could make a custom observable
    # that checks the size of ρ:  If it matches `f`
    # then we use `density!` and do an in-place replace
    # otherwise call `density`.  That will save a
    # lot of heap allocations, and speed hings up
    # signficantly
    # Here is my version -- will need debugging
    ρ=Observable(density(f[]))
    ρparent = @lift ($ρ).parent
    on(f) do f
        #println("f has been updated")
        #@show size(f) size(ρ[])
        if size(f)[1:2]==size(ρ[])
            #println("update ρ in place")
            density!(ρ[],f)
            notify(ρ)
        else
            #println("replace ρ")
            ρ[]=density(f)
        end
    end
    xvals= @lift [axes($ρ,1)...]
    yvals= @lift [axes($ρ,2)...]
    heatmap!(ax_main,xvals,yvals,ρparent)
    ρslice=@lift $ρ[0,:]
    scatter!(ax_slice,ρslice,color=:blue)
    potslice=@lift $pot[0,:]
    lines!(ax_pot,potslice,color=:red)
    ρintegrated=@lift sum($ρ,dims=1)[:]
    #println(typeof(ρintegrated))
    lines!(ax_integrated,ρintegrated)
    # Textbox hooks
    on(L) do L
        reshape!(f[],L[],L[];default=v0)
        normalize!(f[])
        reshape!(df,L[],L[])
        reshape!(k1,L[],L[])
        reshape!(k2,L[],L[])
        reshape!(k3,L[],L[])
        reshape!(k4,L[],L[])
        reshape!(f2,L[],L[])
        notify(f)
    end
    # Slider Hooks
    # Button Hooks
    t=Ref(0.)
    on(ρ) do ρ
        if as[]
            autolimits!(ax_slice)
            autolimits!(ax_main)
            autolimits!(ax_pot)
            autolimits!(ax_integrated)
        end
    end    
    on(reset_button.clicks) do clicks
        reset!(f[],R[],u0[],v0)
        notify(f)
        N[]=sum(ρ[])
        T[]=string(0.)
    end
    on(relax_button.clicks) do clicks
        relax[]=true
        harmonic[]=false
        square[]=false
        isrunning[]=true
        pot[]=bp[]
    end
    on(expand_button.clicks) do clicks
        relax[]=false
        harmonic[]=false
        square[]=false
        isrunning[]=true
        pot[]=zp[]
    end
    on(square_button.clicks) do clicks
        relax[]=false
        harmonic[]=false
        square[]=true
        isrunning[]=true
        pot[]=sbp[]
    end
    on(harmonic_button.clicks) do clicks
        relax[]=false
        harmonic[]=true
        square[]=false
        isrunning[]=true
        pot[]=hp[]
    end
    on(pause_button.clicks) do clicks
        isrunning[]=false
    end
    # Main Loop
    lastdisplay=Ref(0.)
    on(cond -> cond && notify(isrunning_notifier), isrunning)
    errormonitor(
        @async while true
            if isrunning[]
                isopen(fig.scene) || break # stops if window is closed
                if relax[]
                    relaxstep!(f[],relaxp[],N[],dt[],df,k1,k2,k3,k4,f2,ψ[])
                elseif square[]
                    relaxstep!(f[],squarep[],N[],dt[],df,k1,k2,k3,k4,f2,ψ[])
                elseif harmonic[]
                    relaxstep!(f[],harmonicp[],N[],dt[],df,k1,k2,k3,k4,f2,ψ[])
                else
                    expandstep!(f[],expandp[],N[],t[],dt[],df,k1,k2,k3,k4,f2,ψ[])
                end
                lastdisplay[]+=dt[]
                t[]+=dt[]
                if lastdisplay[]>displayt[]
                    notify(f)
                    lastdisplay[]=0.
                    T[]=string(round(t[],digits=3))
                    if recording[]
                        push!(wfs.times,t[])
                        push!(wfs.data,copy(f[]))
                    end
                    #isrunning[]=false
                    # Let Makie process UI/render events between simulation steps.
                    yield()
                end  
                #yield()
            else
                wait(isrunning_notifier)
            end
        end)
    return wfs
end

#expansionviz()


struct timeseries{T}
    times :: Vector{Float64}
    data :: Vector{T}
end

function Num(f::Gutz2D)
    N=0.
    nvals = nv(f) 
    xvals= xv(f)
    for x in xvals
        num=0.
        den=0.
        for n in nvals
            f2=abs2(f[x,n])
            num += n*f2
            den += f2
        end
        N+= num/den
    end
    return N
end

function squaredifference(f1::Gutz2D,f2::Gutz2D)
    dif=0.
    for xn in eachindex(f1)
        dif+=abs2(f1[xn]-f2[xn])
    end
    return dif
end

function energy(f,p,t,scratch=nothing)
    g=p.g(t) #scalar g
    eta=p.eta(t) #scalar eta
    delta = p.delta(t) 
    _energy(f,g,delta,eta,scratch)
end

function _energy(f,g,delta,eta,scratch=nothing)
    if scratch===nothing
        ψ=psi(f) 
    else
        ψ=scratch
        psi!(ψ,f)
    end 
    ax=axes(ψ,1)
    ay=axes(ψ,2)
    xmin=ax[begin]
    xmax=ax[end]
    ymin=ay[begin]
    ymax=ay[end]
    nvals=nv(f)
    rnvals=Iterators.drop(nvals,1)
    energy=0.
    for x in xmin+1:xmax-1, y in ymin+1:ymax-1
        energy-= 2*g*real(ψ[x,y]*conj(ψ[x+1,y]+ψ[x,y+1]))
        for n in rnvals
            energy += n*delta[x,y]*abs2(f[x,y,n])
            energy += (n*(n-1)/2)*eta*abs2(f[x,y,n])
        end
    end
    return energy
end


function initialize(L,R,mN,u0)
    f=Gutz2D(L,L,mN)
    v0=zeros(nv(f))
    v0[0]=1.
    reset!(f,R,u0,v0)
    return f
end


"""
typical values for `delta`

- `delta=CBilliard(R,V)` -- trap of height V and radius R
- `delta=CSquareBilliard(R,V)` -- square trap of height V and radius R (so diameter 2R)
- `delta=CHarmonic(R)` -- harmonic trap
"""
function relax!(f::Gutz2D;g=1.,η,delta,dt,maxiterations=1000,threshold=1e-6,
        storeE::Bool=false,storef::Bool=false,storeinterval::Int=1)
    p=(g=CScalar(Float64(g)),eta=CScalar(Float64(η)),delta=delta)
    N=Num(f)
    df=similar(f)
    k1=similar(f)
    k2=similar(f)
    k3=similar(f)
    k4=similar(f)
    f2=similar(f)
    ψ=psi(f)
    oldE=energy(f,p,0.,ψ)
    E=oldE
    Elist= storeE ?  timeseries([0.],[E]) : nothing 
    flist= storef ? timeseries([0.],[copy(f)]) : nothing 
    numit=0
    sit=0
    for j in 1:maxiterations
        numit=j
        sit+=1
        relaxstep!(f,p,N,dt,df,k1,k2,k3,k4,f2,ψ)
        E=energy(f,p,0.,ψ)
        if abs(E-oldE)<threshold
            break
        end
        oldE=E
        if sit >=storeinterval
            sit=0
            if storeE  
                push!(Elist.times,j*dt)
                push!(Elist.data,E)
            end
            if storef 
                push!(flist.times,j*dt)
                push!(flist.data,copy(f))
            end
        end
    end
    return (f=f,numit=numit,Elist=Elist,flist=flist)
end

function oldevolve!(f;dg,ω,dη,η,delta=CAScalar(0.),
    dt,iterations,storef::Bool=false,storeinterval::Int=1)
    p=(g=OscillatingScalar(1.,Float64(dg),Float64(ω)),
    eta=OscillatingScalar(Float64(η),Float64(dη),Float64(ω)),
                    delta=delta)
    N=Num(f)

    df=similar(f)
    k1=similar(f)
    k2=similar(f)
    k3=similar(f)
    k4=similar(f)
    f2=similar(f)
    ψ=psi(f)
    sit=0
    flist= storef ? timeseries([0.],[copy(f)]) : nothing
    for j in 1:iterations
        t=j*dt
        sit+=1
        simpexpandstep!(f,p,N,t,dt,df,k1,k2,k3,k4,f2,ψ)
        if storef && sit >=storeinterval
            push!(flist.times,j*dt)
            push!(flist.data,copy(f))
            sit=0
        end
    end
    return (f=f,flist=flist)
end

# --- 1) Trial step helper: src -> dest (no in-place commit to src) ---

function simpexpandstep_to!(
    dest::Gutz2D, src::Gutz2D, p, t, dt,
    df, k1, k2, k3, k4, f2, scratch=nothing
)
    rk4step!(src, p, dfdt!, t, dt;
        result=df, k1=k1, k2=k2, k3=k3, k4=k4,
        prefactor=-1.0im, temp=f2, scratch=scratch)

    dest .= src
    dest .+= df
    return dest
end

# Optional: keep old API working
function simpexpandstep!(f::Gutz2D, p, N, t, dt, df, k1, k2, k3, k4, f2, scratch=nothing)
    simpexpandstep_to!(f, f, p, t, dt, df, k1, k2, k3, k4, f2, scratch)
    return f
end


# --- 2) Error norm for RK4 step-doubling ---
# full-step vs two-half-steps error estimate: e ~= (half2 - full)/15

function step_error_norm(full::Gutz2D, half2::Gutz2D; abstol=1e-8, reltol=1e-5)
    s = 0.0
    n = 0
    @inbounds for i in eachindex(full)
        e = (half2[i] - full[i]) / 15
        sc = abstol + reltol * max(abs(full[i]), abs(half2[i]))
        r = abs(e) / sc
        s += r * r
        n += 1
    end
    return sqrt(s / max(n, 1))
end


# --- 3) Adaptive evolve skeleton ---
# Keep "iterations * dt" as total physical time for compatibility.

function evolve!(
    f; dg, ω, dη, η, delta=CAScalar(0.0),
    dt, iterations,
    storef::Bool=false, storeinterval::Int=1,
    adaptive::Bool=true,
    abstol::Float64=1e-8,
    reltol::Float64=1e-5,
    dtmin::Float64=1e-6,
    dtmax::Float64=0.1,
    safety::Float64=0.9,
    facmin::Float64=0.2,
    facmax::Float64=5.0,
    max_reject::Int=20
)
    p = (
        g   = OscillatingScalar(1.0, Float64(dg), Float64(ω)),
        eta = OscillatingScalar(Float64(η), Float64(dη), Float64(ω)),
        delta = delta
    )

    # Keep target time horizon compatible with old fixed-step meaning
    dt0 = Float64(dt)
    tfinal = iterations * dt0
    t = 0.0
    dt_try = dt0

    # Existing work buffers
    N = Num(f)  # currently unused in simpexpandstep!, keep for API compatibility
    df = similar(f)
    k1 = similar(f)
    k2 = similar(f)
    k3 = similar(f)
    k4 = similar(f)
    f2 = similar(f)
    ψ = psi(f)

    # Additional trial states for adaptive control
    f_full = similar(f)
    f_half = similar(f)
    f_mid  = similar(f)

    # Storage: time-based cadence (equivalent to old every 'storeinterval' fixed steps)
    flist = storef ? timeseries([0.0], [copy(f)]) : nothing
    store_dt = storeinterval * dt0
    next_store_t = store_dt

    accepted_steps = 0
    rejected_steps = 0
    min_dt_used = Inf
    max_dt_used = 0.0

    alltimes=[0.0]

    if !adaptive
        # Old fixed-step behavior skeleton
        for j in 1:iterations
            tj = j * dt0
            simpexpandstep!(f, p, N, tj, dt0, df, k1, k2, k3, k4, f2, ψ)
            if storef && (j % storeinterval == 0)
                push!(flist.times, tj)
                push!(flist.data, copy(f))
            end
            push!(alltimes, tj)
        end
        return (f=f, flist=flist, accepted_steps=iterations, rejected_steps=0,times=alltimes)
    end

    # Adaptive loop in physical time
    while t < tfinal
        dt_step = min(dt_try, tfinal - t)
        dt_step = max(dt_step, dtmin)

        local_accepted = false
        local_rejects = 0

        while !local_accepted
            # 1 full step
            simpexpandstep_to!(f_full, f, p, t, dt_step, df, k1, k2, k3, k4, f2, ψ)

            # 2 half steps
            h = dt_step / 2
            simpexpandstep_to!(f_mid,  f,     p, t,     h, df, k1, k2, k3, k4, f2, ψ)
            simpexpandstep_to!(f_half, f_mid, p, t + h, h, df, k1, k2, k3, k4, f2, ψ)

            err = step_error_norm(f_full, f_half; abstol=abstol, reltol=reltol)

            if err <= 1.0 || dt_step <= dtmin
                # Accept better estimate (two half steps)
                f .= f_half
                t += dt_step
                accepted_steps += 1
                push!(alltimes, t)
                min_dt_used = min(min_dt_used, dt_step)
                max_dt_used = max(max_dt_used, dt_step)
                local_accepted = true

                # Grow/shrink next trial dt
                fac = err == 0.0 ? facmax : clamp(safety * err^(-0.2), facmin, facmax)
                dt_try = clamp(dt_step * fac, dtmin, dtmax)

                # Time-based snapshotting
                if storef
                    if t + 1e-14 >= next_store_t
                        push!(flist.times, next_store_t)
                        push!(flist.data, copy(f))
                        next_store_t = t+store_dt
                    end
                end
            else
                # Reject and retry with smaller dt
                rejected_steps += 1
                local_rejects += 1
                fac = clamp(safety * err^(-0.2), facmin, 1.0)
                dt_step = max(dtmin, dt_step * fac)

                if local_rejects >= max_reject
                    error("adaptive evolve!: too many rejected steps near t=$t")
                end
            end
        end
    end

    return (
        f=f,
        flist=flist,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        min_dt=min_dt_used,
        max_dt=max_dt_used,
        times=alltimes
    )
end

function stats(f::Gutz2D)
    N=0.
    X=0.
    XX=0.
    Y=0.
    YY=0.
    XY=0.
    nvals = nv(f) 
    xvals= xv(f)
    for x in xvals
        xX=x[1]
        xY=x[2]
        num=0.
        den=0.
        for n in nvals
            f2=abs2(f[x,n])
            num += n*f2
            den += f2
        end
        n=num/den
        N+=n
        X+=n*xX
        Y+=n*xY
        XX+=n*xX^2
        YY+=n*xY^2
        XY+=n*xX*xY
    end
    return (N=N, X=X, Y=Y, XX=XX, YY=YY, XY=XY)
end

function sliderplot(ev ::timeseries{Gutz2D})
    f=Figure()
    ax=Axis(f[1,1])
    hm=heatmap!(ax,density(ev.data[1]).parent)
    slider=Slider(f[2,1][1,1], range=1:length(ev.times), startvalue=1)
    timelabel=Label(f[2,1][1,2], text="time = $(ev.times[1])",tellwidth=false)
    on(slider.value) do i
        hm[1] = density(ev.data[Int(i)]).parent
        timelabel.text = "time = $(ev.times[Int(i)])"
    end
    display(f)
end

intdens(f::Gutz2D)=intdens(density(f))
intdens(d::OffsetArray)=sum(d,dims=1)[:]

function shadow(w::timeseries{Gutz2D};skip=1)
    data=stack([intdens(f) for f in w.data[1:skip:end]])
end    

function shadowplot(w::timeseries{Gutz2D};skip=1,opts...)
    xvals=collect(axes(w.data[1])[1])
    yvals=w.times
    heatmap(xvals,yvals,shadow(w;skip);opts...)
end

function shadowplot!(ax::Axis,w::timeseries{Gutz2D};skip=1,opts...)
    xvals=collect(axes(w.data[1])[1])
    yvals=w.times
    heatmap!(ax,xvals,yvals,shadow(w;skip);opts...)
end

function rotshadowplot(w::timeseries{Gutz2D};skip=1,opts...)
    xvals=collect(axes(w.data[1])[1])
    yvals=w.times
    heatmap(yvals,xvals,transpose(shadow(w;skip));opts...)
end

function rotshadowplot!(ax::Axis,w::timeseries{Gutz2D};skip=1,opts...)
    xvals=collect(axes(w.data[1])[1])
    yvals=w.times
    heatmap!(ax,yvals,xvals,transpose(shadow(w;skip));opts...)
end