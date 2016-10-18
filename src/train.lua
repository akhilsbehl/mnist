-- require('mobdebug').start()
local optim = require 'optim'
local utils = require 'utils'

--[[

TODO:
   1. Weight initialization
   2. Dropout
   3. Layer tweaking

--]]

cmd = torch.CmdLine()
cmd:text()
cmd:text('Usage: th train.lua --cuda --batchsize 1000')
cmd:text()
cmd:text('Options:')
cmd:text()
cmd:text('Utilities:')
cmd:option('--cuda', false, 'Use CUDA?')
cmd:option('--help', false, 'Print this help message.')
cmd:option('--printstep', 1, 'Print output every step batches.')
cmd:option('--silent', false, 'Suppress logging to stdout?')
cmd:text('Problem selection:')
cmd:option('--dataset', 'mnist', 'Dataset to load.')
cmd:option('--model', './mnist.lua', 'Lua file to load the model from.')
cmd:text('Model hyper-parameters:')
cmd:option('--batchsize', 100, 'Number of instances in an SGD mini batch.')
cmd:option('--earlystop', 10, 'Number of unimproved epochs to stop after.')
cmd:option('--maxepochs', 100, 'Maximum number of epochs to run training for.')
cmd:text('Optimizer parameters:')
cmd:option('--optparams', '{}', 'Params to be passed on to the optimizer.')
cmd:text('Cross-validation:')
cmd:option('--kfolds', 5, 'Proportion of train to use for validation.')
cmd:text()
cmd:silent()

local options = cmd:parse(arg or {})

if options.help then
   table.print(options)
   os.exit()
end

assert(options.kfolds > 1)

if options.cuda then
  require 'cunn'
end

local localize = utils.localizer(options.cuda)

-- 1. The net
local model = dofile(options.model)
local net = localize(model.net)
local criterion = localize(model.criterion)
local confusion = localize(model.confusion)

-- 2. The data
local dataset = utils.load_data(options.dataset, 'train')
dataset = localize(dataset, 'iterate')
dataset_size = dataset['labels']:size(1)

-- 3. The training

local params, grad_params = net:getParameters()
local validation_size, train_size =
   utils.size_validation(dataset_size, 1 / options.kfolds)
local batch_size = options.batchsize

assert(train_size % batch_size == 0)
assert(validation_size % batch_size == 0)
assert(dataset_size % options.kfolds == 0)

local train_batches = train_size / batch_size
local validation_batches = validation_size / batch_size

local batch_records = localize(
   utils.make_batch_container(
      dataset['records'], batch_size))
local batch_labels = localize(
   utils.make_batch_container(
      dataset['labels'], batch_size))

local fold_indices = localize(
   utils.compute_fold_indices(
      dataset, options.kfolds, 'shuffle'),
   'iterate')

-- Make sure that any string literals which are a set of params are
-- evaluated before cat_options is called.
options.optparams = utils.eval_literal(options.optparams)
local iterdir = '../data/iters/' ..
   utils.cat_options(options) ..
   ',' .. os.date('%Y-%m-%d-%H:%M:%S')
paths.mkdir(iterdir)

local info = {
   iterdir = iterdir,
   train_log = io.open(iterdir .. '/training-loss.csv', 'w'),
   options = options,
}

for k = 1, options.kfolds do

   -- Reinitalize the network per fold. Remember to deal with this
   -- when changing init schemes.
   for i, _ in ipairs(net.modules) do
      net.modules[i]:reset()
   end

   local train, validation = utils.kth_fold(dataset, fold_indices, k)

   local best_iter = {
      stop_after = options.earlystop,
      epoch = 0,
      accuracy = 0,
   }

   for epoch = 1, options.maxepochs do

      -- With validation_ratio = 0, essentially shuffles the data.
      local shuffled, _ = utils.make_validation(train, 0)

      for batch = 1, train_batches do
         local o = (batch - 1) * batch_size  -- o := batch offset
         batch_labels:copy(shuffled['labels'][{ {o + 1, o + batch_size} }])
         batch_records:copy(shuffled['records'][{ {o + 1, o + batch_size} }])
         local function learn_batch(params)
            grad_params:zero()
            local batch_estimates = net:forward(batch_records)
            local batch_loss = criterion:forward(batch_estimates, batch_labels)
            local nabla_loss = criterion:backward(batch_estimates, batch_labels)
            net:backward(batch_records, nabla_loss)
            info.timestamp = os.date('%Y-%m-%d %H:%M:%S')
            info.fold = k
            info.epoch = epoch
            info.batch = batch
            info.loss = batch_loss
            utils.communicate(info)
            return batch_loss, grad_params
         end
         optim.sgd(learn_batch, params, options.optparams)
      end

      confusion:zero()
      for batch = 1, validation_batches do
         local o = (batch - 1) * batch_size  -- o := batch offset
         batch_labels:copy(validation['labels'][{ {o + 1, o + batch_size} }])
         batch_records:copy(validation['records'][{ {o + 1, o + batch_size} }])
         local batch_estimates = net:forward(batch_records)
         local _, estimated_labels = torch.max(batch_estimates, 2)
         confusion:batchAdd(estimated_labels, batch_labels)
      end
      confusion:updateValids()
      decision, best_iter = utils.stop_early(info, confusion, best_iter)
      if decision then break end

   end

end
