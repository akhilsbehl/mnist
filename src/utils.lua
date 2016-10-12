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

local function load_data(data_dir)
end

-- 2. Exports

local exports = {
   tablep = tablep,
   cudablep = cudablep,
   cudafy = cudafy,
   map = map,
   load_data = load_data,
}
return exports
