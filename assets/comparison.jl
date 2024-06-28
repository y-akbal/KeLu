using Plots
using NNlib
using Flux
using Zygote
using Distributions


@inline function KeLu(x; a = 3.5)
    return  ifelse(x < -a, 0, ifelse(x>a, x, @fastmath 0.5*x*(1+x/a+(1/pi)*sin(x*pi/a))))
end

@inline function KeLu(x::AbstractMatrix{T}) where T<:Float32
    return KeLu.(x)
end




function d(f::Function, x::Real)
    return Zygote.gradient(x->f(x), x)[1]
end


function d__(f::Function, x::Real)
    f_d(x) = Zygote.gradient(x->f(x), x)[1]
    return Zygote.gradient(x->f_d(x), x)[1]
end


x = -3:0.01:3
plot(x, x -> KeLu(x), label = "KeLü")
plot!(x, gelu.(x), label = "Gelu")
#plot!(x, sigmoid.(x), label = "Sigmoid")
plot!(x, relu.(x), label = "Relu")
#plot!(x, swish.(x), label = "Swish")
#plot!(x, hardswish.(x), label = "Hard Swish")
#plot!(size=(1400,1400))
png("comparison.png")



x = -5:0.01:5
plot(x, d.(KeLu,x), label = "KeLü First Derivative")
plot!(x, d.(gelu,x), label = "Gelu First Derivative")
plot!(x, d.(swish, x), label = "Swish First Derivative")
#plot!(x, d.(tanh, x), label = "Relu First Derivative")
png("comparison_first_derivatives.png")


x = -5:0.01:5
plot(x, d__.(KeLu,x), label = "KeLü Second Derivative")
plot!(x, d__.(gelu,x), label = "Gelu Second Derivative")
plot!(x, d__.(swish, x), label = "Swish Second Derivative")
plot!(x, d__.(tanh, x), label = "Relu Second Derivative")
png("comparison_second_derivatives.png")