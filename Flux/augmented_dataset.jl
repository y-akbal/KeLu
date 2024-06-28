using Augmentor
using Images
# create five example observations where each observation is
# made up of two conceptually linked 3x3 arrays
# create output arrays of appropriate shape
# transform the batch of images

function augment(batch)
    img = colorview(RGB, batch)
    outs = similar(batch)
    augmentbatch!(outs, img, FlipY(0.2)*FlipX(0.5)*ElasticDistortion(15,15,0.1)*CropRatio(0.1) |> SplitChannels())
    return outs
end


"""
x = randn(31, 33,3,100)
y = randn(3, 31, 33, 100)
permutedims!(y, x,(3,1,2,4))
permute

permute(x, (3,1,2,4))
augment(randn(31, 33,3,100)) |> size

@show names(Threads)
f(x) = x^2
x = Threads.@spawn f(2)

fetch(x)

"""