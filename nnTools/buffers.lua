------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Tensor buffers class for convenient managment of
allocated memory. Allocates a buffer for each
field of provided input table ('data').

Fields in table 'shared' will share a common 
buffer. Groups are specified according to the
convention 'membername'='groupname' and should
be passed using table config.shared.

To do:
  - Implement buffer-specific fullsize / batchsize

Authored: 2015-11-04 (jwilson)
Modified: 2015-11-13
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local math  = require('math')

-------- Safeguard for isolated request
bot7 = bot7 or {}
bot7.nnTools = bot7.nnTools or {} 

---------------- Constants
local defaults = 
{ 
  verbose   = 0,     -- Print level
  fullsize  = 5120,  -- Actual size of buffer
  batchsize = 32,    -- Number of elements to return per call
  deepcopy  = false, -- Force copying of all input tensors
  inplace   = true,  -- Force reuse of existing memory
  
  shared = 
  {
    xe='x', xr='x', xv='x',
    ye='y', yr='y', yv='y',
  },
}

------------------------------------------------
--                                       buffers
------------------------------------------------
local title   = 'bot7.nnTools.buffers'
local buffers = torch.class(title)

function buffers:__init(config, data)
  self.config = utils.table.deepcopy(config or {})
  utils.table.update(self.config, defaults, true)
  self.buffers = {} -- Buffer Table
  self.groups  = {} -- Shared buffer groups
  if data then self:init(data) end
end

function buffers:init(data, config)
  local config = utils.table.update(self.config, config)

  -------- Shared buffer groups
  for key, group in pairs(config.shared) do
    if not self.groups[group] then 
      self.groups[group] = {key}
    else
      table.insert(self.groups[group], key)
    end
  end

  -------- Initialize buffers
  local TensorTypes, DefaultType = {}, nil
  if type(config.types) == 'table' then TensorTypes = config.types end
  if type(config.type)  == 'string' and config.type:find('Tensor') then
    DefaultType = config.type
  end
  

  local ttype, idx, temp
  for key, field in pairs(data) do
    if torch.isTensor(field) then
      -------- Determine Buffer type
      temp = TensorTypes[key]
      if type(temp) == 'string' and temp:find('Tensor') then
        ttype = TensorTypes[key]
      elseif DefaultType then
        ttype = DefaultType
      else
        ttype = field:type()
      end
      ---- Extract out parent name
      -- Assumed to be 'torch' or some alias thereof
      idx = ttype:find('%.')
      if idx then ttype = ttype:sub(idx+1, ttype:len()) end

      -------- Allocate buffer for field
      self:alloc(key, field:size(), ttype)
    end
  end
end

function buffers:__call__(key)
  if type(key) == 'string' and self.buffers[key] then
    local config = self.config
    local head = self.buffers[key].index
    local tail = math.min(head+config.batchsize-1, self.buffers[key].shape[1])
    return self.buffers[key].tensor:sub(head, tail)
  end
end

function buffers:alloc(key, shape, ttype)
  local config = self.config
  local ttype  = ttype or config.type or torch.getdefaulttensortype()
  local group  = nil
  if ttype:find('%.') then
    ttype = ttype:sub(7, ttype:len())
  end

  if self.buffers[key] then
    self.buffers[key].tensor:resize(shape)
    if self.buffers[key].tensor:type() ~= ttype then
      self.buffers[key].tensor = self.buffers[key].tensor:type(ttype)
      collectgarbage()
    end
  else
    group = config.shared[key]
    if group then 
      if not self.buffers[group] then self:alloc(group, shape) end
      self.buffers[key] = self.buffers[group]
    else
      self.buffers[key] = {shape=shape, dims=shape:size(), index=1, type=ttype}
      self.buffers[key].shape[1] = 0 -- allocate memory as needed
      self.buffers[key].tensor = torch[ttype](self.buffers[key].shape)

      if self.config.verbose > 0 then
        local info = utils.tensor.info(self.buffers[key].tensor)
        local msg  = string.format("> Allocated %s buffer "..
        "'%s' with shape %s", info.type, key, info.shape)
        print(msg)
      end
    end
  end
end

function buffers:set(key, tensor, idx, ttype)
  local config = self.config
  local buffer = self.buffers[key]

  -------- Existing Tensor
  if buffer then
    if ttype and ttype ~= buffer.type then
      print('Error: Re-casting of buffers not yet implemented; returning...')
      return
    end

    ---- Complete Tensor
    if not idx then
      buffer.shape[1] = tensor:size(1) 
      if buffer.type ~= tensor:type() then
        if config.inplace then
          buffer.tensor:resize(buffer.shape):copy(tensor)
        else
          buffer.tensor = tensor:type('torch.'..buffer.type)
        end
      elseif config.deepcopy then
        if config.inplace then
          buffer.tensor = buffer.tensor:resize(buffer.shape):copy(tensor)
        else
          buffers.tensor = tensor:clone()
        end
      else
        buffer.tensor = tensor
      end

    ---- Contiguous Subtensor
    elseif type(idx) == 'table' then
      buffer.shape[1] = idx[2]-idx[1]+1
      if buffer.type ~= tensor:type() then
        if config.inplace then
          buffer.tensor:resize(buffer.shape):copy(tensor:sub(idx[1], idx[2]))
        else
          buffer.tensor = tensor:sub(idx[1], idx[2]):type('torch.'..buffer.type)
        end
      elseif config.deepcopy then
        if config.inplace then
          buffer.tensor:resize(buffer.shape):copy(tensor:sub(idx[1], idx[2]))
        else
          buffer.tensor = tensor:sub(idx[1], idx[2]):clone()
        end
      else
        buffer.tensor = tensor:sub(idx[1], idx[2])
      end

    ---- Noncontiguous Subtensor
    elseif torch.isTensor(idx) then
      buffer.shape[1] = idx:nElement()
      if buffer.type ~= tensor:type() then
        if config.inplace then
          buffer.tensor:resize(buffer.shape):copy(tensor:index(1, idx))
        else
          buffer.tensor = tensor:index(1, idx):type('torch.'..buffer.type)
        end
      elseif config.inplace then -- torch.index() allocates, so deepcopy is irrelevant
        buffer.tensor:resize(buffer.shape):index(tensor, 1, idx)
      else
        buffer.tensor = tensor:index(1, idx)
      end

    ---- Error Case  
    else
      print('Error: buffers.set() encountered an '.. 
            'unrecognized index type; returning...')
      return
    end

    ---- If no error occurred, reset index to first position
    buffer.index = 1 

  -------- New Tensor
  elseif type(key) == 'string' and torch.isTensor(tensor) then
    self:alloc(key, tensor:size(), ttype)
    self:set(key, tensor, idx, ttype) -- kinda sloppy
  else
    print('Error: invalid arguments to buffers.set(); returning...')
    return
  end
end

function buffers:increment(keys, stepsize)
  local config = self.config
  local keys   = keys
  local group  = nil
  local flags  = {}

  local buffers  = self.buffers
  local stepsize = stepsize or config.batchsize

  -------- Increment specified buffers 
  if type(keys) == 'string' then keys = {keys} end
  for idx, key in pairs(keys) do
    assert(buffers[key].index <= buffers[key].shape[1])
    group = config.shared[key]
    if not group then
      buffers[key].index = math.min(buffers[key].index + stepsize, buffers[key].shape[1])
    elseif not flags[group] then -- avoid incrememting group 
      flags[group] = true        -- multiple times
      buffers[key].index = math.min(buffers[key].index + stepsize, buffers[key].shape[1])
      buffers[key].index = buffers[key].index + stepsize
    end
  end
end

function buffers:update(data, idx, patterns)
  local config   = self.config
  local patterns = patterns
  if type(patterns) == 'string' then
    patterns = {patterns}
  end
  
  -------- Iteratively update buffers
  local flag
  for key, field in pairs(data) do
    flag = torch.isTensor(field)

    ---- Check for string patters
    if flag and patterns then
      flag = false
      for idx, pattern in pairs(patterns) do
        flag = key:find(pattern) ~= nil
        if flag then break end
      end
    end

    ---- Update specified buffer
    if flag then self:set(key, field, idx) end
  end
  collectgarbage()
end


function buffers:has_key(key)
  return (self.buffers[key] ~= nil)
end

function buffers:index_reset(keys)
  if keys then
    local keys = keys
    if type(keys) == 'string' then keys = {keys} end
    for idx, key in pairs(keys) do
      self.buffers[key].index = 1
    end
  else
    for key, buffer in pairs(self.buffers) do
      buffer.index = 1
    end
  end
end

function buffers:fill(val, keys)
  assert(type(val) == 'number')
  if keys then
    local keys = keys
    if type(keys) == 'string' then keys = {keys} end
    for idx, key in pairs(keys) do
      self.buffers[key].tensor:fill(val)
    end
  else
    for key, buffer in pairs(self.buffers) do
      buffer.tensor:fill(val)
    end
  end
end
function buffers:zero(keys) self:fill(0.0, keys) end
function buffers:ones(keys) self:fill(1.0, keys) end

function buffers:type(ttype, keys)
  assert(type(ttype) == 'string')

  local ttype, keys  = ttype, keys
  local buffers = self.buffers
  local errMsg  = 
  {
    badType = 'Error: Invalid tensor type specified '..
              'in buffers.type(); returning...',
    badKey  = 'Error: Unrecognized key encountered '..
              'in buffers.type(); returning...',
  }

  -------- Type Checking
  if not ttype:find('Tensor') then
    print(errMsg.badType); return
  end

  local idx = ttype:find('%.')
  if idx then ttype = ttype:sub(idx+1, ttype:len()) end
  if not torch[ttype] then print(errMsg.badType); return end

  -------- Buffer Type Conversion 
  if not keys then
    for key, _ in pairs(self.buffers) do self:type(ttype, key) end
  elseif type(keys) == 'table' then
    for idx, key in pairs(keys) do self:type(ttype, key) end
  elseif type(keys) == 'string' then
    if not buffers[keys]
      then print(errMsg.badKey); return
    elseif buffers[keys].type ~= ttype then
      buffers[keys].type   = ttype
      buffers[keys].tensor = buffers[keys].tensor:type('torch.'..ttype)
    end
  else
    print(errMsg.badKey); return
  end

end
function buffers:cuda(keys) self:type('CudaTensor', keys) end
function buffers:byte(keys) self:type('ByteTensor', keys) end
function buffers:float(keys) self:type('FloatTensor', keys) end
function buffers:double(keys) self:type('DoubleTensor', keys) end
function buffers:typeAs(tensor, keys) self:type(tensor:type(), keys) end


return bot7.nnTools.buffers -- return the class, not its metatable!

