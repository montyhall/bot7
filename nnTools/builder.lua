------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple class for constructing neural nets.

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

Authored: 2015-10-16 (jwilson)
Modified: 2015-11-05
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local nn    = require('nn')

------------------------------------------------
--                                       trainer
------------------------------------------------
local builder = function(config, data)
  ---------------- Default Settings
  local config = config or {}
  -------- Model
  local model =
  {
    problem    = 'classification',
    output     = 'LogSoftMax',
    edge       = 'Linear',
    activation = 'ReLU',

    xDim    = 1024,
    yDim    = 10,
    nLayers = 3,
    nHidden = 100,
    dropout = 0,

    gpu     = false,
  }
  utils.table.update(config, model, true)

  -------- GPU Dependencies
  if config.gpu then
    if not cunn then require('cunn') end
    if not cutorch then require('cutorch') end
  end

  -------- Data-specific
  if data then
    config['xDim'] = data.xr:size(2)
    if data.yr:dim() > 1 then 
      config['yDim'] = data.yr:size(2)
      config.problem = 'regression'
    elseif model.problem == 'classification' then
      config['yDim'] = data.yr:max()
    else
      config['yDim'] = 1
    end
  end

  ---------------- Auxiliary Structures
  -------- Per-layer hyperparameter structures
  local nLayers = config.nLayers

  ---- Input/Output sizes
  local dims    = config.nHidden
  if type(dims) == 'table' then 
    dims = torch.Tensor(dims)
  end
  if torch.isTensor(dims) then
    local N = dims:nElement()
    if N == nLayers then
      dims = torch.cat({torch.Tensor{config.xDim}, dims, torch.Tensor{config.yDim}}, 1)
    elseif N == nLayers-1 and dims[1] == config.xDim then
      dims = torch.cat({dims, torch.Tensor{config.yDim}}, 1)
    elseif N == nLayers-1 and dims[N] == config.yDim then
      dims = torch.cat({torch.Tensor{config.xDim}, dims}, 1)
    end
  else
    dims = torch.cat({torch.Tensor{config.xDim}, 
                      torch.Tensor(nLayers):fill(dims),
                      torch.Tensor{config.yDim}}, 1)
  end

  ---- Dropout rates
  local dropRate = config.dropout
  if type(dropRate) == 'table' then
    dropRate = torch.Tensor(dropRate)
  end
  if torch.isTensor(dropRate) then
    if dropRate:nElement() == nLayers then
      dropRate = torch.cat({torch.zeros(1), dropRate, torch.zeros(1)}, 1)
    else
      dropRate = torch.cat({dropRate, torch.zeros(1)}, 1)
    end
  else
    dropRate = torch.zeros(config.nLayers+2)
    dropRate:sub(2, config.nLayers+1):fill(config.dropout)
  end

  ---- Edge types
  local edge = {}
  if type(config.edge) == 'string' then
    for k = 1, nLayers+1 do edge[k] = config.edge end
  else
    edge = config.edge
  end

  ---- Activation functions
  local activation = {}
  if type(config.activation) == 'string' then
    for k = 1, nLayers+1 do activation[k] = config.activation end
  else
    activation = config.activation
  end

  ---------------- Model Construction
  -------- Input Layer 
  local network = nn.Sequential()
  network:add(nn.Reshape(dims[1]))
  if nn[edge[1]] ~= nil then
    network:add(nn[edge[1]](dims[1], dims[2]))
  end

  -------- Input/Hidden Layer(s)
  for layer = 2, nLayers+1 do
    ---- Activation Function
    if nn[activation[layer]] ~= nil then
      network:add(nn[activation[layer]]())
    end
    ---- Dropout Module
    local rate = dropRate[layer]
    if rate > 0 and rate < 1 then
      network:add(nn.Dropout(rate))
    end
    ---- Edge Module (Weights)
    if nn[edge[layer]] ~= nil then
      network:add(nn[edge[layer]](dims[layer], dims[layer+1]))
    end
  end

  -------- Output Layer
  if nn[config.output] ~= nil then
    network:add(nn[config.output]())
  end

  -------- Run model on GPU?
  if config.gpu then return network:cuda()
  else return network end
end

return builder