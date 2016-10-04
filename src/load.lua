require 'nn'
require 'optim'
w_init = require 'weight-init'

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

-- local herpNet = w_init(derpNet, 'kaiming_caffe')
-- local herpNet = w_init(derpNet, 'xavier_caffe')
local herpNet = derpNet

-- 2. The criterion
local criterion = nn.ClassNLLCriterion()

-- 3. The trainer
local params, gradParams = herpNet:getParameters()
local optimState = {learningRate = 1e-3}

-- 4. The training
local nEpochs = 12
local batchSize = 1000
local trainSize = train_data['data']:size(1)
assert(trainSize % batchSize == 0,
       'Use a batch size that cleanly divides training size.')
local nBatches = trainSize / batchSize
local batchInputs = torch.Tensor(batchSize, 1, 32, 32)
local batchResponse = torch.Tensor(batchSize)

for epoch = 1, nEpochs do

   local shuffle = torch.randperm(train_data['data']:size(1))

   for batch = 1, nBatches do

      for i = 1, batchSize do
         local case = (batch - 1) * batchSize + i
         local shuffled = shuffle[case]
         batchInputs[i]:copy(train_data['data'][shuffled])
         batchResponse[i] = train_data['labels'][shuffled]
      end

      local function evaluateBatch(params)
         gradParams:zero()
         local batchEstimate = herpNet:forward(batchInputs)
         local batchLoss = criterion:forward(batchEstimate, batchResponse)
         local nablaLoss = criterion:backward(batchEstimate, batchResponse)
         herpNet:backward(batchInputs, nablaLoss)
         print('Finished epoch: ' .. epoch .. ', batch: ' ..
                  batch .. ', with loss: ' .. batchLoss)
         return batchLoss, gradParams
      end

      optim.sgd(evaluateBatch, params, optimState)

   end

end
