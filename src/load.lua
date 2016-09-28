require 'nn'
require 'optim'

local train_data_path = '../data/train_32x32.t7'
local test_data_path = '../data/test_32x32.t7'

local train_data = torch.load(train_data_path, 'ascii')
local test_data = torch.load(test_data_path, 'ascii')

-- 1. The net
local derpNet = nn.Sequential()

derpNet:add(nn.SpatialConvolution(1, 8, 5, 5))
derpNet:add(nn.ReLU())
derpNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))

derpNet:add(nn.SpatialConvolution(8, 16, 5, 5))
derpNet:add(nn.ReLU())
derpNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))

derpNet:add(nn.Reshape(400))
derpNet:add(nn.Linear(400, 100))

derpNet:add(nn.LogSoftMax())

-- 2. The criterion
local criterion = nn.ClassNLLCriterion()

-- 3. The trainer
local params, gradParams = derpNet:getParameters()
local optimState = {learningRate = 1e-3}

-- 4. The training
local nEpochs = 4
local batchSize = 64

for epoch = 1, nEpochs do

   local shuffle = torch.randperm(train_data['data']:size(1))

   local batch = 1

   for batchOffset = 1, train_data['data']:size(1), batchSize do

      local batchInputs = torch.Tensor(batchSize, 1, 32, 32)
      local batchResponse = torch.Tensor(batchSize)

      for i = 1, batchSize do
         local ind = shuffle[batchOffset + i]
         batchInputs[i]:copy(train_data['data'][ind])
         batchResponse[i] = train_data['labels'][ind]
      end

      local function evaluateBatch(params)
         gradParams:zero()
         local batchEstimate = derpNet:forward(batchInputs)
         local batchLoss = criterion:forward(batchEstimate, batchResponse)
         local nablaLoss = criterion:backward(batchEstimate, batchResponse)
         derpNet:backward(batchInputs, nablaLoss)
         print('Finished epoch: ' .. epoch .. ', batch: ' ..
                  batch .. ', with loss: ' .. batchLoss)
         return batchLoss, gradParams
      end

      optim.sgd(evaluateBatch, params, optimState)

      batch = batch + 1
      batchOffset = batchOffset + batchSize

   end

end
