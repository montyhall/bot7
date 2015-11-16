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
local math   = require('math')
local bot7   = require('bot7')
local autoML = require('bot7.nnTools.automator')
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
cmd:option('-gpu',      false, 'train neural network using GPU')
cmd:option('-target',   'loss', "specify target measure for optimization: {'loss', 'err'}")
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
local expt =
{ 
  verbose = opt.verbose,
  gpu = opt.gpu,
  yDim = data.yr:max(),
  batchsize = 32
}
---- User Interface
expt.ui = {msg_freq = math.huge} -- don't report progress

---- Model Architecture
expt.model = {nLayers=2, nHidden=100}

---- Training Schedule
expt.schedule = {nEpochs = 100}

---- Supplant default expt values (if provided)
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
  }
end

targs = {target = opt.target} -- specify key for target measures 

---- Run automator
autoML(data, expt, hypers, targs)





