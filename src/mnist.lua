local nn = require 'nn'

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

net:add(nn.LogSoftMax())

-- 2. Criterion
local criterion = nn.ClassNLLCriterion()

-- 3. Exports
exports = {
   net = net,
   criterion = criterion,
}
return exports
