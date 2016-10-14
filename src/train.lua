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
cmd:option('--epochs', 100, 'Number of epochs to run training for.')
cmd:option('--valprop', 0.2, 'Proportion of train to use for validation.')
cmd:option('--batchsize', 100, 'Number of instances in an SGD mini batch.')
cmd:option('--optparams', '{}', 'Params to be passed on to the optimizer.')
cmd:option('--silent', false, 'Suppress logging to stdout?')
cmd:option('--printstep', 1, 'Print output every step batches.')
cmd:option('--skiplog', false, 'Avoid logging the run?')
cmd:text()

options = cmd:parse(arg or {})

if options.help then
   require 'os'
   table.print(options)
   os.exit()
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

-- 3. The training
local params, grad_params = net:getParameters()
local optim_state = utils.eval_string(options.optparams)

local validation_size = math.ceil(dataset['records']:size(1) * options.valprop)
local train_size = dataset['records']:size(1) - validation_size
local batch_size = options.batchsize

-- Let's waste some of the data if it so happens. This is fine when
-- batch sizes are small.
local train_batches = math.floor(train_size / batch_size)
local overflow_train = train_size % batch_size
if overflow_train ~= 0 then
   print(overflow_train .. ' of ' .. train_size ..
            ' training records will be unused per epoch.')
end

local validation_batches = math.floor(validation_size / batch_size)
local overflow_validation = validation_size % batch_size
if overflow_validation ~= 0 then
   print(overflow_validation .. ' of ' .. validation_size ..
            ' validation records will be unused per epoch.')
end

local batch_records_storage = dataset['records']:size()
batch_records_storage[1] = batch_size
local batch_records = localize(torch.Tensor(batch_records_storage))

local batch_labels_storage = dataset['labels']:size()
batch_labels_storage[1] = batch_size
local batch_labels = localize(torch.Tensor(batch_labels_storage))

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

for epoch = 1, options.epochs do

   local train, validation = utils.make_validation(dataset, options.valprop)

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

   print('Total accuracy of classifier at completion of epoch ' .. epoch ..
            ' = ' .. confusion.averageValid * 100 .. '.')
   print('Mean accuracy across classes at completion of epoch ' .. epoch ..
            ' = ' .. confusion.totalValid * 100 .. '.')

end
