using ChainRules:NoTangent
using Zygote
using Flux
using BenchmarkTools


@inline function KeLu(x; a = 3.5f0)
    pi_ = oftype(a, pi)
    inv_a = inv(a)
    return  ifelse(x < -a, 0.f0, ifelse(x>a, x, @fastmath  oftype(a, 0.5)*x*(one(a)+x*inv_a +(inv(pi_))*sin(x*pi_*inv_a))))
end



@inline function Kelu_derivative(x; a = 3.5f)
    pi_ = oftype(a, pi)
    inv_a = inv(a)
    return ifelse(x < -a, 0.f0, ifelse(x>a, one(x), @fastmath  oftype(a, 0.5)*(one(a)+2*x*inv_a +(inv(pi_))*(sin(x*pi_*inv_a)+inv_a*cos(x*pi_*inv_a)))))
end


Zygote.@adjoint KeLu(x) = KeLu.(x), a -> (Kelu_derivative.(x)*a,)


