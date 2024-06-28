using Flux, MLDatasets, NNlib
using CUDA, cuDNN
using Flux: onehotbatch, onecold, Optimiser
using ProgressBars
using MLUtils: DataLoader
using StatsBase
using BSON: @save,@load
using NNlib
using Wandb, Dates, Logging
using Parameters
using Random
using CUDA
using Augmentor



include("tools.jl")
include("KeLu.jl")

@kwdef struct training_args
    Wandb_Name::String = "GeLu_0.1label_smoothing"
    project_name::String = "KeLu_vs_GeLu"
    in_channel::Int = 3
    η::Float64 = 3e-4
    patch_size::Int = 2
    kernel_size::Int = 8
    embedding_dim::Int = 512
    depth::Int = 16
    use_cuda::Bool = true
    CudaDevice::Int = 0
    n_epochs::Int = 25
    num_classes::Int = 10
    seed::Int = 0
    batch_size::Int = 16
end
 
function ConvMixer(in_channels::Int64, kernel_size::Int64, patch_size::Int64, dim::Int64, depth::Int64, N_classes::Int64; activation::Function)
    model = Chain(
            Conv((patch_size, patch_size), in_channels=>dim, activation; stride=patch_size),
            BatchNorm(dim),
            [
                Chain(
                    SkipConnection(Chain(Conv((kernel_size,kernel_size), dim=>dim,  activation; pad=SamePad(), groups=dim), BatchNorm(dim)), +),
                    Chain(Conv((1,1), dim=>dim, activation), BatchNorm(dim))
                ) 
                for i in 1:depth
            ]...,
            AdaptiveMeanPool((1,1)),
            Flux.flatten,
            Dense(dim,N_classes)
        )
    return model
end

function get_statistics(dataset::DataType)
    data_set = dataset(:train)[:][1]
    return mean(data_set, dims = [1, 2, 4]), std(data_set, dims = [1,2, 4])
end

  

function get_data(batchsize::Int64; dataset = MLDatasets.CIFAR10)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

    xtrain, ytrain = dataset(:train)[:]
    xtest, ytest = dataset(:test)[:]
    
    # Normalize -- these dudes may be recalculated for each run--
    m, s = dataset |> get_statistics
    xtrain = @. (xtrain - m)/s
    xtest = @. (xtest - m)/s

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true, parallel = true, buffer= true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize, parallel = true, buffer= true)
    @info "Dataset preprocessing is done!!!"
    return train_loader, test_loader
end


function train(args::training_args)

    ## Extract params from args
    η = args.η
    in_channel = args.in_channel
    patch_size = args.patch_size
    kernel_size = args.kernel_size
    embedding_dim = args.embedding_dim
    depth = args.depth
    use_cuda = args.use_cuda
    cuda_device = args.CudaDevice
    num_classes = args.num_classes
    rng_seed = args.seed
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    project_name = args.project_name
    Wandb_Name = args.Wandb_Name

    
    train_loader, test_loader = get_data(batch_size)

    if use_cuda
        device = gpu
        CUDA.device!(cuda_device)
        @info "Training on GPU:$cuda_device"
        
    else
        device = cpu
        @info "Training on CPU"
    end
    
    model = begin
        Random.seed!(rng_seed)
        ConvMixer(in_channel, kernel_size, 
        patch_size, embedding_dim, 
        depth, num_classes, 
        activation = gelu) |> device    
    end
    

    opt = Optimiser(
            WeightDecay(1f-10), 
            ClipNorm(2.0),
            ADAM(η)
            )
    opt_state = Flux.setup(opt, model)

    # Start a new run, tracking hyperparameters in config
    lg = WandbLogger(project = project_name, name = Wandb_Name*"-$(now())", config = Dict("architecture" => "CNN", "dataset" => "CIFAR-10"))
    # Use LoggingExtras.jl to log to multiple loggers together
    global_logger(lg)
    # -- #
    train_loss = loss_reg()
    val_loss = loss_reg()

    pl = FlipX(0.5) |> SplitChannels() |> PermuteDims((2, 3, 1))


    for epoch in ProgressBar(1:n_epochs)
 
        for (x,y) in  train_loader
            x,y = map(device, [x,y])
            y = Flux.label_smoothing(y, 0.1f0)
            loss, grads = Flux.withgradient(model) do model
                Flux.logitcrossentropy(model(x), y)                
            end
            update!(train_loss, loss |> cpu)
            Flux.update!(opt_state, model, grads[1])
        end

        acc = 0.f0
        m = 0
        for (x,y) in test_loader
            x,y = map(device, [x,y])
            z = model(x)
            temp_validation_loss = Flux.logitcrossentropy(model(x), y) 
            update!(val_loss, temp_validation_loss |> cpu)
            acc += sum(onecold(z).==onecold(y)) |> cpu
            m += size(x)[end]
        end

        
        
        #logging
        Wandb.log(lg, Dict("loss" => get_avg(train_loss), "acc"=>acc/m, "validation_loss" => get_avg(val_loss)))
        map(reset!, [train_loss, val_loss])
    end
    close(lg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = training_args()
    train(args)
end