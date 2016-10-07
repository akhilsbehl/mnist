require 'cunn'
require 'cutorch'
require 'nn'
require 'optim'

local use_cuda = true

local function tablep (this)
   return type(this) == 'table'
end

local function cudablep (this)
   if getmetatable(this) then
      return this.cuda ~= nil
   end
end

local function cudafy (this)
   return cudablep(this) and this:cuda() or this
end

local function map (fun, over)
   local mapped = {}
   for key, elem in pairs(over) do
      mapped[key] = fun(elem)
   end
   return mapped
end

local function localize (this, iterate)
   if use_cuda then
      return (iterate and tablep(this)) and
         map(cudafy, this) or
         cudafy(this)
   else
      return this
   end
end

local train_data_path = '../data/train_32x32.t7'
local test_data_path = '../data/test_32x32.t7'

local train_data = localize(torch.load(train_data_path, 'ascii'), 'iterate')
local test_data = localize(torch.load(test_data_path, 'ascii'), 'iterate')

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

local herpNet = localize(derpNet)

-- 2. The criterion
local criterion = localize(nn.ClassNLLCriterion())

-- 3. The trainer
local params, gradParams = herpNet:getParameters()
local optimState = {learningRate = 1e-3}

-- 4. The training
local nEpochs = 1
local batchSize = 1000
local trainSize = train_data['data']:size(1)
assert(trainSize % batchSize == 0,
       'Use a batch size that cleanly divides training size.')
local nBatches = trainSize / batchSize
local batchInputs = localize(torch.Tensor(batchSize, 1, 32, 32))
local batchResponse = localize(torch.Tensor(batchSize))

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
