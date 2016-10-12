require 'optim'
utils = require 'utils'

--[[

TODO:
   1. Weight initialization
   2. Early stopping
   3. Dropout
   4. Cross validation
   5. Layer tweaking

--]]

cmd = torch.CmdLine()
cmd:text()
cmd:text('Experimenting with MNIST to learn NNs.')
cmd:text('Example:')
cmd:text('$> th mnist.lua --cuda true --batch_size 1000')
cmd:text('Options:')
cmd:option('--cuda', false, 'Use CUDA?')
cmd:option('--dataset', 'mnist', 'Dataset to load.')
cmd:option('--modelpath', 'mnist.lua', 'Lua file to load the model from.')
cmd:option('--nepochs', 3, 'Number of epochs to run training for.')
cmd:option('--batchsize', 1000, 'Number of instances in an SGD mini batch.')
cmd:option('--sgdparams', {}, 'Params to be passed on to the SGD optimizer.')
cmd:option('--silent', false, 'Avoid printing to stdout?')
cmd:text()

options = cmd:parse(arg or {})

if options.cuda then
  require 'cunn'
  require 'cutorch'
end

local function localize (this, iterate)
   if options.cuda then
      return (iterate and utils.tablep(this)) and
         utils.map(utils.cudafy, this) or
         utils.cudafy(this)
   else
      return this
   end
end

-- 1. The net
local model = dofile(options.modelpath)
local net = localize(model.net)
local criterion = localize(model.criterion)

-- 2. The data
local dataset = utils.load_data(options.dataset, 0.25)
local train = localize(dataset.train, 'iterate')
local validation = localize(dataset.validation, 'iterate')
local test = localize(dataset.test, 'iterate')

-- 3. The trainer
local params, grad_params = net:getParameters()
local optim_state = {learningRate = 1e-3}

-- 4. The training
local n_epochs = 1
local batch_size = 1000
local train_size = train_data['data']:size(1)
assert(train_size % batch_size == 0,
       'Use a batch size that cleanly divides training size.')
local n_batches = train_size / batch_size
local batchInputs = localize(torch.Tensor(batch_size, 1, 32, 32))
local batchResponse = localize(torch.Tensor(batch_size))

for epoch = 1, n_epochs do

   local shuffle = torch.randperm(train_data['data']:size(1))

   for batch = 1, n_batches do

      for i = 1, batch_size do
         local case = (batch - 1) * batch_size + i
         local shuffled = shuffle[case]
         batchInputs[i]:copy(train_data['data'][shuffled])
         batchResponse[i] = train_data['labels'][shuffled]
      end

      local function evaluate_batch(params)
         grad_params:zero()
         local batchEstimate = net:forward(batchInputs)
         local batchLoss = criterion:forward(batchEstimate, batchResponse)
         local nablaLoss = criterion:backward(batchEstimate, batchResponse)
         net:backward(batchInputs, nablaLoss)
         print('Finished epoch: ' .. epoch .. ', batch: ' ..
                  batch .. ', with loss: ' .. batchLoss)
         return batchLoss, grad_params
      end

      optim.sgd(evaluate_batch, params, optim_state)

   end

end
