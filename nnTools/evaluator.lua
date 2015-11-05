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
Modified: 2015-11-05
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local nn    = require('nn')

---------------- Constants
local defaults =
{
  batchsize = 1024,
  max_eval  = math.huge,
  confusion = true,
  gpu       = false,
}

------------------------------------------------
--                                     evaluator
------------------------------------------------
local evaluator = function(network, criterion, X, Y, config, confusion)
  -------- Local Initialization
  local network, criterion = network, criterion
  local config  = utils.table.deepcopy(config or {})
  utils.table.update(config, defaults, true)
  local network = network
  local X, xDim = X, X:size(2)
  local Y, yDim = Y, 1
  local nInputs = Y:size(1)
  local batchsize = math.min(config.batchsize, nInputs)

  -------- Detect Target Type
  ---- Classification
  local ttype = Y:type()
  if ttype == 'torch.LongTensor' 
  or ttype == 'torch.ByteTensor'
  or Y:eq(torch.floor(Y)):all() then
    yDim = Y:max()
  ---- Regression
  else
    if Y:dim() > 1 then yDim = Y:size(2) end
    config.confusion = false -- not for regression
  end

  -------- Establish Confusion Matrix
  local confusion  = confusion
  if not confusion and config.confusion then
    confusion = optim.ConfusionMatrix(yDim)
  end

  -------- Evaluate a random subset
  if nInputs > config.max_eval then
    local idx = torch.randperm(N, 'torch.LongTensor'):sub(1, config.max_eval)
    X = X:index(1, idx)
    Y = Y:index(1, idx)
    nInputs = config.max_eval
  end

  -------- Allocate Buffers (GPU only)
  local inputs, targets
  if config.gpu then
    inputs = torch.CudaTensor(batchsize, xDim)
    if yDim > 1 then
      targets = torch.CudaTensor(batchsize, yDim)
    else
      targets = torch.CudaTensor(batchsize)
    end
  end

  -------- Batch Generator
  local get_batch = function(tail, head)
    if config.gpu then
      local nRow = head-tail+1
      inputs:resize(nRow, xDim):copy(X:sub(tail, head))
      if yDim > 1 then
        targets:resize(nRow,yDim):copy(Y:sub(tail, head))
      else
        targets:resize(nRow):copy(Y:sub(tail, head))
      end
      return inputs, targets
    else
      return X:sub(tail, head), Y:sub(tail, head)
    end
  end

  -------- Evaluation Loop
  local loss, outputs = 0, nil
  local head, nRow    = 0, 0
  local singleton     = (batchsize == 1)
  network:evaluate()
  for tail = 1, nInputs, batchsize do
    head = math.min(head + batchsize, nInputs)
    inputs, targets = get_batch(tail, head)
    outputs = network:forward(inputs)
    loss = loss + criterion:forward(outputs, targets)
    if confusion then
      if singleton then confusion:add(outputs,targets)
      else confusion:batchAdd(outputs, targets) end
    end
  end
  loss = loss/math.ceil(nInputs/batchsize)

  -------- Return recorded measures
  if confusion then
    confusion:updateValids()
    local err = 1 - confusion.totalValid
    confusion:zero()
    return loss, err
  else
    return loss
  end
end

return evaluator
