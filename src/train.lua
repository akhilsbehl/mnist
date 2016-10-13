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
cmd:text('$> th mnist.lua --cuda true --batchsize 1000')
cmd:text('Options:')
cmd:option('--cuda', false, 'Use CUDA?')
cmd:option('--dataset', 'mnist', 'Dataset to load.')
cmd:option('--modelpath', 'mnist.lua', 'Lua file to load the model from.')
cmd:option('--epochs', 3, 'Number of epochs to run training for.')
cmd:option('--batchsize', 128, 'Number of instances in an SGD mini batch.')
cmd:option('--optparams', '{}', 'Params to be passed on to the optimizer.')
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
local dataset = utils.load_data(options.dataset, 'train')
local train, validation = utils.make_validation(dataset, 0.25)
train = localize(train, 'iterate')
validation = localize(validation, 'iterate')

-- 3. The training
local params, grad_params = net:getParameters()
local optim_state = utils.eval_string(options.optparams)

-- Let's waste some of the data if it so happens. This is fine when
-- batch sizes are small.
local batch_size = options.batchsize
local batches = math.floor(train['records']:size(1) / batch_size)

local batch_records_storage = train.records:size()
batch_records_storage[1] = batch_size
local batch_records = localize(torch.Tensor(batch_records_storage))

local batch_labels_storage = train.labels:size()
batch_labels_storage[1] = batch_size
local batch_labels = localize(torch.Tensor(batch_labels_storage))

for epoch = 1, options.epochs do

   local shuffle = torch.randperm(train['records']:size(1))

   for batch = 1, batches do

      for i = 1, batch_size do
         local case = (batch - 1) * batch_size + i
         local shuffled = shuffle[case]
         batch_records[i]:copy(train['records'][shuffled])
         batch_labels[i] = train['labels'][shuffled]
      end

      local function evaluate_batch(params)
         grad_params:zero()
         local batch_estimate = net:forward(batch_records)
         local batch_loss = criterion:forward(batch_estimate, batch_labels)
         local nabla_loss = criterion:backward(batch_estimate, batch_labels)
         net:backward(batch_records, nabla_loss)
         print(epoch .. ',' .. batch .. ',' .. batch_loss)
         return batch_loss, grad_params
      end

      optim.sgd(evaluate_batch, params, optim_state)

   end

end
