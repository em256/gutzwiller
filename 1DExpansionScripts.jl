function masterscript(shell_script::AbstractString,
                           julia_scripts::AbstractVector{<:AbstractString})
    shellquote(s) = "'" * replace(s, "'" => "'\"'\"'") * "'"

    open(shell_script, "w") do io
        println(io, "#!/usr/bin/env bash")
        println(io)

        for script in julia_scripts
            stdout_file = script * ".out"
            stderr_file = script * ".err"
            println(io,
                "nohup julia $(shellquote(script)) > $(shellquote(stdout_file)) " *
                "2> $(shellquote(stderr_file)) &")
        end
    end

    chmod(shell_script, 0o755)
    return shell_script
end

function expscript(root;U,num,len,T,V=30,time_step=0.1,maxn=num)
    filename=root*"$U.jl"
    open(filename,"w") do io
        println(io,
        """
using Pkg
Pkg.activate(".")

include("1DExpansion.jl")

U=$U
println("Starting loop with U=$U")
root="$root$U"
data=tdvpexp(;num=$num,
trap=well($(2*num);V=$V,L=$(num+0.5),s=0),
    U,time_step=$time_step,T=$T,maxn=$maxn,
    verbose=true,
    outputlevel=1,redirectstdout=root*".out");
saveocs(data,root*".hdf5")
println("*******************************************")
""")
    end
    return filename
end

function scaledexpmodscript(root;ωlist,dU,dg,U,num,len,T,V=30,time_step=0.1,maxn=num,dir=".",maxdim)
    wU=round(Int,U*10)
    wdg=round(Int,dg*100)
    wdU=round(Int,dU*100)
    filename=root*"sU$(wU)dU$(wdU)dg$(wdg).jl"
    open(filename,"w") do io
        println(io,
        """
using Pkg
Pkg.activate(".")

include("1DExpansion.jl")

if !(isdir(\"$dir\"))
    mkdir(\"$dir\")
end
            
U=$U
dU=$dU
dg=$dg
num=$num
println("Starting loop with U=$U")
root="$(dir)/$(root)sU$(wU)"
redirectstdout=root*".out"
open(redirectstdout,"w") do io
    redirect_stdout(io) do
        gs=findgs(num,well($(2*num);V=$V,L=$(num+0.5),s=0);
            t=1,U,maxn=$maxn,etol=1e-8,
            dmrgopts=(;nsweeps=100,maxdim=$maxdim,cutoff=1E-6,outputlevel=1))
        ψ=gs.ψ
        E=gs.E      
        for ω in $ωlist
            println("************************")
            println("calling expand with ω=\$ω")
            println("************************")
            ex=expand(ψ;g=1,η=U,T=$T,time_step=$time_step,
            tdvpopts=(;updater=krylov_updater,
            updater_kwargs=(;tol=1e-8,eager=true),
            cutoff=1e-6,maxdim=$maxdim,
            nsite=2,outputlevel=1
            ),dη=dU,dg,ω)
            wω=round(Int,100*ω)
            saveocs(ex,root*"w\$(wω).hdf5")
        end
    end
end
println("*******************************************")
""")
    end
    return filename
end

function prodexpscript(root;U,num,len,T,V=30,time_step=0.1,maxn=num,maxdim=500)
    filename=root*"$U.jl"
    open(filename,"w") do io
        println(io,
        """
using Pkg
Pkg.activate(".")

include("1DExpansion.jl")

U=$U
println("Starting loop with U=$U")
root="$root$U"
data=prodtdvpexp(;num=$num,
    U,time_step=$time_step,T=$T,maxn=$maxn,
    verbose=true,maxdim=$maxdim,
    outputlevel=1,redirectstdout=root*".out");
saveocs(data,root*".hdf5")
println("*******************************************")
""")
    end
    return filename
end

function wUexpscript(root;wU,num,len,T,V=30,time_step=0.1,maxn=num)
    U=0.1*wU
    filename=root*"0$wU.jl"
    open(filename,"w") do io
            println(io,
"""
using Pkg
Pkg.activate(".")

include("1DExpansion.jl")

U=$U
println("Starting loop with U=$U")
root="$(root)0$wU"
data=tdvpexp(;num=$num,
    trap=well($(2*num);V=$V,L=$(num+0.5),s=0),
    U,time_step=$time_step,T=$T,maxn=$maxn,
    verbose=true,
    outputlevel=1,redirectstdout=root*".out");
    saveocs(data,root*".hdf5")
println("*******************************************")
""")
    end
    return filename
end

function prodwUexpscript(root;wU,num,len,T,V=30,time_step=0.1,maxn=num,maxdim=500)
    U=0.1*wU
    filename=root*"0$wU.jl"
    open(filename,"w") do io
            println(io,
"""
using Pkg
Pkg.activate(".")

include("1DExpansion.jl")

U=$U
println("Starting loop with U=$U")
root="$(root)0$wU"
data=prodtdvpexp(;num=$num,
    U,time_step=$time_step,T=$T,maxn=$maxn,
    verbose=true,maxdim=$maxdim,
    outputlevel=1,redirectstdout=root*".out");
    saveocs(data,root*".hdf5")
println("*******************************************")
""")
    end
    return filename
end
            
