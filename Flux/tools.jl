
using Parameters

abstract type  LossReg end

@with_kw mutable struct loss_reg <: LossReg
    loc_accuracy::Float32 = 0.f0
    n::Int = 0
end


function reset!(acc::loss_reg)
    acc.loc_accuracy = 0.f0
    acc.n = 0
end

function update!(acc::loss_reg, value::Float32)
    acc.loc_accuracy += value
    acc.n += 1
end

function get_avg(acc::loss_reg)
    return acc.loc_accuracy/acc.n 
end