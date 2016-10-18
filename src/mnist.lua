local nn = require 'nn'
local optim = require 'optim'

-- 1. Net
local net = nn.Sequential()

net:add(nn.SpatialConvolution(1, 8, 5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.SpatialConvolution(8, 16, 5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.Reshape(400))
net:add(nn.Linear(400, 100))

net:add(nn.Linear(100, 10))
net:add(nn.LogSoftMax())

-- 2. Criterion
local criterion = nn.ClassNLLCriterion()

-- 3. Confusion matrix
local confusion = optim.ConfusionMatrix(10, torch.range(1, 10))

-- 3. Exports
return {
   net = net,
   criterion = criterion,
   confusion = confusion,
}
