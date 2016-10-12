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

local function load_data(dataset, validation_ratio)
   local loaders = {
      mnist = load_mnist,
      -- cifar10 = load_cifar10,
   }
   local train_, test = loaders[dataset]()
   local train, validation = make_validation(train_, validation_ratio)
   return {
      train = train,
      validation = validation,
      test = test,
   }

end

local function load_mnist()
   local train = torch.load(data_dir .. '/train.t7', 'ascii')
   train['records'] = train['data']
   train.data = nil
   local test = torch.load(data_dir .. '/test.t7', 'ascii')
   test['records'] = test['data']
   test.data = nil
   return train, test
end

local function make_validation(train, validation_ratio)
   local train_size = train['labels']:size(1)
   local validation_size = math.ceil(validation_ratio * train_size)
   local validation_indices =
      torch.randperm(train_size)[{ {1, validation_size} }]

-- 2. Exports

local exports = {
   tablep = tablep,
   cudablep = cudablep,
   cudafy = cudafy,
   map = map,
   load_data = load_data,
}
return exports
