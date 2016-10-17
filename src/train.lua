require('mobdebug').start()
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
cmd:text('Usage: th train.lua --cuda true --batchsize 1000')
cmd:text('Options:')
cmd:option('--help', false, 'Print this help message.')
cmd:option('--cuda', false, 'Use CUDA?')
cmd:option('--dataset', 'mnist', 'Dataset to load.')
cmd:option('--modelpath', 'mnist.lua', 'Lua file to load the model from.')
cmd:option('--maxepochs', 100, 'Maximum number of epochs to run training for.')
cmd:option('--earlystop', 10, 'Number of unimproved epochs to stop after.')
cmd:option('--kfolds', 5, 'Proportion of train to use for validation.')
cmd:option('--batchsize', 100, 'Number of instances in an SGD mini batch.')
cmd:option('--optparams', '{}', 'Params to be passed on to the optimizer.')
cmd:option('--silent', false, 'Suppress logging to stdout?')
cmd:option('--printstep', 1, 'Print output every step batches.')
cmd:option('--skiplog', false, 'Avoid logging the run?')
cmd:text()

options = cmd:parse(arg or {})

if options.help then
   table.print(options)
   os.exit()
end

assert(options.kfolds > 1)

-- Needed for logging, so let's deal with this first.
local optim_state = utils.eval_literal(options.optparams)
if not options.skiplog then
   local logpath = '../logs/'      ..
      options.dataset .. '-'       ..
      os.date('%Y-%m-%d-%H-%M-%S') ..
      '.log'
   logfile = assert(io.open(logpath, 'w'))
   for param, value in pairs(options) do
      logfile:write('# ' .. param .. '=' .. tostring(value) .. '\n')
   end
   for param, value in pairs(optim_state) do
      logfile:write('# ' .. param .. '=' .. tostring(value) .. '\n')
   end
end

if options.cuda then
  require 'cunn'
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

      for batch = 1, train_batches do
         local o = (batch - 1) * batch_size  -- o := batch offset
         batch_labels:copy(train['labels'][{ {o + 1, o + batch_size} }])
         batch_records:copy(train['records'][{ {o + 1, o + batch_size} }])
         local function learn_batch(params)
            grad_params:zero()
            local batch_estimates = net:forward(batch_records)
            local batch_loss = criterion:forward(batch_estimates, batch_labels)
            local nabla_loss = criterion:backward(batch_estimates, batch_labels)
            net:backward(batch_records, nabla_loss)
            local info = {
               timestamp = os.date('%Y-%m-%d %H:%M:%S'),
               fold = k,
               epoch = epoch,
               batch = batch,
               loss = batch_loss,
            }
            utils.communicate(
               info, logfile, options.skiplog,
               options.silent, options.printstep)
            return batch_loss, grad_params
         end
         optim.sgd(learn_batch, params, optim_state)
      end

      confusion:zero()
      for batch = 1, validation_batches do
         local o = (batch - 1) * batch_size  -- o := batch offset
         batch_labels:copy(validation['labels'][{ {o + 1, o + batch_size} }])
         batch_records:copy(validation['records'][{ {o + 1, o + batch_size} }])
         local batch_estimates = net:forward(batch_records)
         confusion:batchAdd(batch_estimates, batch_labels)
      end
      confusion:updateValids()
      print('Total accuracy of classifier at completion of fold ' .. k ..
               ', epoch ' .. epoch .. ' = ' ..
               confusion.totalValid * 100 .. '.')
      print('Mean accuracy across classes at completion of fold ' .. k ..
               ', epoch ' .. epoch .. ' = ' ..
               confusion.averageValid * 100 .. '.')
      if confusion.totalValid > best_iter.accuracy then
         best_iter.stop_after = options.earlystop
         best_iter.epoch = epoch
         best_iter.accuracy = confusion.totalValid
         print('Best model so far. Saving to disk.')
         -- torch.save(utils.make_model_save_path(), net)
      else
         best_iter.stop_after = best_iter.stop_after - 1
         if best_iter.stop_after == 0 then
            print('Stopping early at epoch ' .. epoch .. '!')
            break
         end
      end

   end

end
