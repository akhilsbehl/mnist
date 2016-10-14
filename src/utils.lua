-- 1. Definitions

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

-- Use with caution!
local function eval_string(s)
   return loadstring('return ' .. s)()
end

local function load_mnist_part(part)
   local path = '../data/mnist/' .. part .. '_32x32.t7'
   local data = torch.load(path, 'ascii')
   data['records'] = data['data']
   data.data = nil
   return data
end

local function load_mnist(part)
   if part == 'test' or part == 'train' then
      return load_mnist_part(part)
   else
      return {
         train = load_mnist_part('train'),
         test = load_mnist_part('test'),
      }
   end
end

local function load_data(set, part)
   local data = nil
   if set == 'mnist' then
      data = load_mnist(part)
   -- elseif set == 'cifar10' then  -- Add more here.
   --    data = load_cifar10(part)
   else
      error('Unknown dataset!')
   end
   return data
end

local function make_validation(train_full, validation_ratio)
   local train_full_size = train_full['labels']:size(1)
   local validation_size = math.ceil(validation_ratio * train_full_size)
   local train_size = train_full_size - validation_size
   local shuffle = torch.randperm(train_full_size):long()
   local validation_indices = shuffle[{ {1, validation_size} }]
   local train_indices = shuffle[{ {validation_size + 1, train_full_size} }]
   train, validation = {}, {}
   for set, data in pairs(train_full) do
      train[set] = data:index(1, train_indices)
      validation[set] = data:index(1, validation_indices)
   end
   return train, validation
end

local function makelog(info)
   return
      info.timestamp  .. ',' ..
      info.epoch      .. ',' ..
      info.batch      .. ',' ..
      info.loss      .. '\n'
end

local function makemsg(info)
   return
      '['.. info.timestamp .. '] ' ..
      'Finished ' ..
      'epoch = ' .. info.epoch ..
      ', ' ..
      'batch = ' .. info.batch ..
      ', ' ..
      'with ' ..
      'loss = ' .. info.loss ..
      '.\n'
end

local function communicate(info, logfile, skiplog, silent, printstep)
   if not skiplog then
      logfile:write(makelog(info))
   end
   if (not silent) and (info.batch % printstep == 0) then
      io.stdout:write(makemsg(info))
   end
end

-- 2. Exports

return {
   tablep = tablep,
   cudablep = cudablep,
   cudafy = cudafy,
   map = map,
   eval_string = eval_string,
   load_data = load_data,
   make_validation = make_validation,
   communicate = communicate,
}
