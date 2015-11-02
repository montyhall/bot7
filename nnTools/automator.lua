------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Convenience script for automated tuning of
hyperparameters for a prespecified neural 
network architecture in Torch7.

Args:
  data    : target dataset(s)
  expt    : experiment design table. contains per module
              configuration tables. overwriting values 
              here should effect changes in the target
              modules (i.e. elements of expt should occur
              as pointers)
  hypers  : Table of hyperparameter objects; see 
              hyperparam.lua for details

Optional Table Arguments (targs):
  trainer : method for training a model from start to 
              finish. should return a tensor of measures
              the first k elements of which should be
              the k target measures.


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

Authored: 2015-10-15 (jwilson)
Modified: 2015-11-02
--]]

---------------- External Dependencies
local paths   = require('paths')
local math    = require('math')
local utils   = require('bot7.utils')
local bots    = require('bot7.bots')
local trainer = require('bot7.nnTools.trainer')
local builder = require('bot7.nnTools.builder')

---------------- Constants
local defaults = 
{
  bot = 
  {
    type     = 'bayesopt',
    budget   = 100,
    nInitial = 5,
  },
}

------------------------------------------------
--                                     automator
------------------------------------------------
local automator = function(data, expt, hypers, targs)
  local expt    = utils.tbl_update(expt, defaults, true)
  local targs   = targs or {}
  local trainer = targs.trainer or trainer
  local builder = targs.builder or builder
  local nHypers = utils.tbl_size(hypers)
  expt.xDim   = nHypers

  -------- Auxiliary method for extracting response value(s)
  local aux = {}
  aux.extract = function(vals, target)
  
    ---- Determine target measure(s)
    local target = target
    local ttype = type(target)
    if ttype == 'nil' then 
      target = {'err', 'loss'}
    elseif ttype == 'string' or ttype == 'number' then
      target = {target} 
    else 
      print('Error: nnTools.automator encountered an '..
            'unrecognized target type; returning...')
    end

    ---- Extract value(s)
    local vtype = type(vals)
    if vtype == 'table' then
      for idx, key in pairs(target) do
        if vals[key] then return aux.extract(vals[key]) end
      end
    end
    return utils.as_val(vals, 1)
  end
    

  local wrapper = function(vals)
    -------- Overwrite expt using values in hyp
    local env = {_G=_G, expt=expt, data=data, targs=targs} -- make visable
    local f, path, key, idx
    for k = 1, nHypers do
      key   = hypers[k].key
      idx   = string.find(key, '%.')
      stack = env[key:sub(1,idx-1)]
      key   = key:sub(idx+1, key:len())
      idx   = string.find(key, '%.')
      while idx do
        stack = stack[key:sub(1, idx-1)] or {}
        key   = key:sub(idx+1, key:len())
        idx   = string.find(key, '%.')
      end
      stack[key] = hypers[k]:warp(vals[k])
    end

    -------- Build & Train Neural Network
    local network  = builder(expt.model, data)
    local response = trainer(network, data, expt)
    return aux.extract(response, targs.target)
  end

  local bot = targs.bot or bots[expt.bot.type](expt, wrapper)
  return bot:run_experiment()
end

return automator