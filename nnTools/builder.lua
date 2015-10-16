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
Modified: 2015-10-26
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
    problem = 'classification',
    output  = 'LogSoftMax',
    nLayers = 3,
    nHidden = 100,
    dropout = 0,
    xDim    = 1024,
    yDim    = 10,
  }
  -- utils.tbl_update(config, model, true)
  setmetatable(config, {__index = model})

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
  -------- Per-layer hyperparameter tensors
  local nLayers = config.nLayers
  local dims    = config.nHidden
  if type(dims) == 'table' then 
    dims = torch.Tensor(dims)
  end
  if torch.isTensor(dims) then
    if dims:nElement() == nLayers then
      dims = torch.cat({dims, dims[nLayers], torch.Tensor{config.yDim}}, 1)
    else
      dims = torch.cat(dims, torch.Tensor{config.yDim}, 1)
    end
  else
    dims = torch.cat(torch.Tensor(nLayers+1):fill(dims), torch.Tensor{config.yDim}, 1)
  end

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

  ---------------- Model Construction
  -------- Input Layer 
  local network = nn.Sequential()
  network:add(nn.Linear(config.xDim, dims[1]))

  -------- Hidden Layers
  for layer = 1, config.nLayers+1 do
    network:add(nn.Tanh())
    local rate = dropRate[layer]
    if rate > 0 and rate < 1 then
      network:add(nn.Dropout(rate))
    end
    network:add(nn.Linear(dims[layer], dims[layer+1]))
  end

  -------- Output Layer
  if nn[config.output] ~= nil then
    network:add(nn[config.output]())
  end

  return network
end

return builder