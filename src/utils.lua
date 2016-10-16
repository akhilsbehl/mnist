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
local function eval_literal(s)
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

local function size_validation(train_full_size, validation_ratio)
   local validation_size = math.ceil(validation_ratio * train_full_size)
   local train_size = train_full_size - validation_size
   return validation_size, train_size
end

local function make_validation(train_full, validation_ratio)
   validation_size, train_size =
      size_validation(train_full['labels']:size(1), validation_ratio)
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
      info.fold       .. ',' ..
      info.epoch      .. ',' ..
      info.batch      .. ',' ..
      info.loss       .. '\n'
end

local function makemsg(info)
   return
      '['.. info.timestamp .. '] ' ..
      'Finished ' ..
      'fold = ' .. info.fold ..
      ', ' ..
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
   if (not silent) then
      if printstep == 1 or
      (printstep > 1 and info.batch % printstep == 1) then
         io.stdout:write(makemsg(info))
      end
   end
end

local function make_batch_container(dataset, batch_size, dim)
   dim = dim or 1
   local batch_storage = dataset:size()
   batch_storage[dim] = batch_size
   local batch_container = torch.Tensor(batch_storage)
   return batch_container
end

local function compute_fold_indices(dataset, kfolds, shuffle)
   local indices = shuffle and
      torch.randperm(dataset_size) or
      torch.range(1, dataset_size)
   local fold_size = dataset_size / kfolds
   local fold_indices = {}
   for k = 1, kfolds do
      offset = (k - 1) * fold_size
      fold_indices[k] = indices[{ {offset + 1, offset + fold_size} }]
   end
   return fold_indices
end

local function kth_fold_indices(fold_indices, k)
   local validation_indices = fold_indices[k]
   local train_indices = fold_indices[k == 1 and 2 or 1]:clone()
   for i = 2, #fold_indices do
      if i ~= k then
         train_indices = train_indices:cat(fold_indices[i], 1)
      end
   end
   return validation_indices, train_indices
end

local function kth_fold(dataset, fold_indices, k)
   local validation_indices, train_indices =
      kth_fold_indices(fold_indices, k)
   local train, validation = {}, {}
   for set, data in pairs(dataset) do
      train[set] = data:index(1, train_indices)
      validation[set] = data:index(1, validation_indices)
   end
   return train, validation
end

-- 2. Exports

return {
   communicate = communicate,
   compute_fold_indices = compute_fold_indices,
   cudablep = cudablep,
   cudafy = cudafy,
   eval_literal = eval_literal,
   kth_fold = kth_fold,
   kth_fold_indices = kth_fold_indices,
   load_data = load_data,
   make_batch_container = make_batch_container,
   make_validation = make_validation,
   map = map,
   size_validation = size_validation,
   tablep = tablep,
}
