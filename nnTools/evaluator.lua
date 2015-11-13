------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple class for evaluating neural networks.

Expects data to be passed in as:
  ------------------------------
  | Field | Content            |
  ------------------------------
  |  xr   | Training Inputs    |
  |  yr   | Training Targets   |
  |  xe   | Test Inputs        |
  |  ye   | Test Targets       |
  |  xv   | Validation Inputs  |
  |  yv   | Validation Targets |
  ------------------------------

Authored: 2015-10-27 (jwilson)
Modified: 2015-11-13
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local nn    = require('nn')
local Buffers = require('bot7.nnTools.buffers')

---------------- Constants
local defaults =
{ 
  evalsize  = 1024, -- batchsize during evaluation
  max_eval  = math.huge,
  confusion = true,
  buffers   = {fullsize=5120},
}

-- Set inheritence for sub-tables
for key, field in pairs(defaults) do
  if type(field) == 'table' then
    setmetatable(field,{__index=defaults})
  end
end

------------------------------------------------
--                                     evaluator
------------------------------------------------
local evaluator = function(network, criterion, X, Y, config, cache)
  -------- Local Initialization
  local network   = network
  local criterion = criterion
  local data   = {x=X, y=Y}
  local cache  = cache or {}
  local config = utils.table.deepcopy(config or {})
  utils.table.update(config, defaults, true)

  -------- Establish Buffers
  local nInputs   = data.y:size(1)
  config.evalsize = math.min(config.evalsize, nInputs)
  local buffers   = cache.buffers or Buffers(config.buffers, data)
  buffers.config.batchsize = math.min(buffers.config.fullsize, config.evalsize)

  -------- Enforce GPU settings
  if config.gpu then
    require('cunn'); require('cutorch');
    network:cuda(); criterion:cuda(); buffers:cuda() -- Convert to GPU
    buffers.config.type = 'CudaTensor'
  end

  -------- Detect Target Type
  ---- Classification
  local ttype, yDim = data.y:type()
  if ttype == 'torch.LongTensor' 
  or ttype == 'torch.ByteTensor'
  or data.y:eq(torch.floor(data.y)):all() then
    yDim = data.y:max()
  ---- Regression
  else
    if data.y:dim() > 1 then yDim = data.y:size(2) end
    config.confusion = false -- not for regression
  end

  -------- Establish Confusion Matrix
  local confusion = cache.confusion
  if not confusion and config.confusion then
    confusion = optim.ConfusionMatrix(yDim)
  end

  -------- Evaluate a random subset
  if nInputs > config.max_eval then
    local idx = torch.randperm(N, 'torch.LongTensor'):sub(1, config.max_eval)
    data.x = data.x:index(1, idx)
    data.y = data.y:index(1, idx)
    nInputs = config.max_eval
  end

  -------- Evaluation Loop
  network:evaluate()
  if criterion.evaluate then criterion:evaluate() end

  local buffer_size = buffers.config.fullsize
  local batchsize   = buffers.config.batchsize
  local singleton   = (config.evalsize == 1)
  local loss, err   = 0

  local nBatches, tail
  for head = 1, nInputs, buffer_size do
    tail     = math.min(head+buffer_size-1, nInputs)
    nBatches = math.ceil((tail-head+1)/batchsize)
    idx      = {head, tail}
    buffers:update(data, idx, {'x', 'y'})

    for batch = 1, nBatches do
      local inputs  = buffers('x')
      local targets = buffers('y')
      outputs = network:forward(inputs)
      loss = loss + criterion:forward(outputs, targets)
      if confusion then
        if singleton then confusion:add(outputs,targets)
        else confusion:batchAdd(outputs, targets) end
      end
      buffers:increment({'x', 'y'})
    end
  end
  loss = loss/math.ceil(nInputs/batchsize)

  -------- Return recorded measures
  if confusion then
    confusion:updateValids()
    err = 1 - confusion.totalValid
    confusion:zero()
  end
  return loss, err
  
end

return evaluator
