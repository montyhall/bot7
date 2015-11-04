------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Automated hyperparameter tuning for simple
neural networks.

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

For details regarding hyperparameters, please
see: hyperparam.lua
--]]

---------------- External Dependencies
local math    = require('math')
local bot7    = require('bot7')
local nnTools = require('bot7.nnTools')
local utils      = bot7.utils
local hyperparam = bot7.hyperparam

------------------------------------------------
--                              Argument Parsing
------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('================================ Options:')
cmd:option('-verbose',  0, 'specify print level')
cmd:option('-data',     'examples/data/iris_test30.t7', 'path to target dataset')
cmd:option('-expt',     '', 'path to experiment configuration file')
cmd:option('-hypers',   '', 'path to hyperparameter specification file')
cmd:text('================================')
local opt = cmd:parse(arg or {})
if opt.verbose then utils.ui.printArgs() end

------------------------------------------------
--                                        autoML
------------------------------------------------

---- Load dataset
local data = torch.load(opt.data)

-------- Default Experiment Settings
---- High-level settings
local expt = {yDim=data.yr:max(), verbose=opt.verbose, msg_freq=-1}

---- Model architecture
expt.model = {nLayers=2, nHidden=100}

---- Training schedule
expt.schedule = {nEpochs = 100}

---- Load expt config file (if provided)
if opt.expt ~= '' then
  utils.table.update(expt, paths.dofile(opt.expt))
end

---- Establish metatables
for key, val in pairs(expt) do
  if type(val) == 'table' then
    setmetatable(val, {__index = expt})
  end
end

---- Specify hyperparameters
local hypers 
if opt.hypers ~= '' then
  hypers = paths.dofile(opt.hypers)
else
  hypers = 
  {
    hyperparam('expt.optimizer.learningRate', 1e-3, 1e-1),
    hyperparam('expt.optimizer.learningRateDecay', 1e-7, 1e-3),
    hyperparam('expt.optimizer.weightDecay', 1e-7, 1e-2),
    hyperparam('expt.optimizer.momentum', 0, 1),
    hyperparam('expt.model.dropout', 0, 1),
    hyperparam('expt.schedule.corruption.degree', 0, 0.5),
  }
end

targs = {target = 'loss'} -- specify key for target measures

---- Run automator
nnTools.automator(data, expt, hypers, targs)





