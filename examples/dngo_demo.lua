------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple demo for DNGO bot.

Target Article:
"Scalable Bayesian Optimization Using Deep 
Neural Networks" (Snoek et. al 2015)


Authored: 2015-09-30 (jwilson)
Modified: 2015-11-08
--]]

---------------- External Dependencies
local paths = require('paths')
local bot7  = require('bot7')
local utils = bot7.utils
local benchmarks = require('bot7.benchmarks')

------------------------------------------------
--                                Initialization
------------------------------------------------

---------------- Argument Parsing
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-verbose',   3, 'specify which print level')
cmd:option('-bot',       'bo', 'specify which bot to use: {bo, rs}')
cmd:option('-benchmark', 'hartmann6', 'specify which function to optimize')
cmd:option('-msg_freq',  1, 'interval between progress reports (in terms of epochs)')
cmd:option('-nInitial',   2, 'specify number of initial candidates to sample at random')
cmd:option('-budget',    300, 'number of candidate nominations (queries)')
cmd:option('-nTrain',    0, 'number of training instances to pass to network')
cmd:option('-nTest',     0, 'number of test instances to pass to network')
cmd:option('-nValid',    0, 'number of validation instances to pass to network')
cmd:option('-xDim',      100, 'specify input dimensionality for experiment')
cmd:option('-yDim',      1,   'specify output dimensionality for experiment')
cmd:option('-noisy',     true, 'specify observations as noisy')
cmd:option('-batchsize', 32, 'specify batchsize for DNGO network')
cmd:option('-grid_size', 20000, 'specify size of candidate grid')
cmd:option('-grid_type', 'random', 'specify type of grid to employ')
cmd:option('-mins', '',  'specify minima for inputs (defaults to 0.0)')
cmd:option('-maxes',     '', 'specify maxima for inputs (defaults to 1.0)')
cmd:option('-score',      'ei', 'specify acquisition function to be used by bot; {ei, ucb}')

cmd:text()
opt = cmd:parse(arg or {})

---------------- Experiment Configuration

------ Establish xDim for experiment
if     (opt.benchmark == 'braninhoo') then
  opt.xDim = 2
elseif (opt.benchmark == 'hartmann3') then
  opt.xDim = 3
elseif (opt.benchmark == 'hartmann6') then
  opt.xDim = 6
end

local expt = 
{  
  func   = opt.benchmark, 
  xDim   = opt.xDim,
  yDim   = opt.yDim,
  budget = opt.budget,
  msg_freq = opt.msg_freq,
  verbose  = opt.verbose,
  batchsize = opt.batchsize,
}

expt.bot   = {type = opt.bot, nInitial = opt.nInitial}
expt.grid  = {type = opt.grid_type, size = opt.grid_size}
expt.model = {output=''} -- necessary for regression setting!

---- Network Initialization Settings
local init        = expt.init or {}
init['schedule']  = {nEpochs=100, corruption={degree=0.2}}
init['optimizer'] = 
{
  type              = 'sgd',
  learningRate      = 1e-2,
  learningRateDecay = 1e-3,
  weightDecay       = 1e-4,
  momentum          = 0.9,
}
expt['init'] = init
for key, field in pairs(expt.init) do
  if type(field) == 'table' then
    setmetatable(field, {__index=expt.init})
  end
end


---- Network Update Settings
local update = expt.update or utils.table.deepcopy(expt.init)
update.schedule.nEpochs = 100
update.optimizer.weightDecay  = 5e-4
update.optimizer.learningRate = 1e-2
update.optimizer.learningRateDecay = 5e-5
expt['update'] = update

---- Establish feasible hyperparameter ranges
if (opt.mins ~= '') then
  loadstring('expt.mins='..opt.mins)()
else
  expt.mins = torch.zeros(1, opt.xDim)
end

if (opt.maxes ~= '') then
  loadstring('opt.maxes='..opt.maxes)()
else
  expt.maxes = torch.ones(1, opt.xDim)
end

---- Choose acquistion function
expt['score'] = {}
if (opt.score == 'ei') then
  expt.score['type'] = 'expected_improvement'
elseif (opt.score == 'ucb' or opt.score == 'lcb') then
  expt.score['type'] = 'confidence_bound'
end

---- Set metatables
for key, field in pairs(expt) do
  if type(field) == 'table' then
    setmetatable(field, {__index=expt})
  end
end


------------------------------------------------
--                                     dngo_demo
------------------------------------------------
function synthesize()
  local config = utils.table.deepcopy(expt.grid)
  local func   = benchmarks[expt.func]
  local grid   = bot7.grids[config.type]()
  local data   = {}

  if opt.nTest > 0 then
    config.size = opt.nTrain
    data.xr = grid(config)
    data.yr = func(data.xr)
    if data.yr:dim() == 1 then
      data.yr:resize(opt.nTrain, 1)
    end
  end

  if opt.nTest > 0 then
    config.size = opt.nTest
    data.xe = grid(config)
    data.ye = func(data.xe)
    if data.ye:dim() == 1 then
      data.ye:resize(opt.nTest, 1)
    end
  end

  if opt.nValid > 0 then
    config.size = opt.nValid
    data.xv = grid(config)
    data.yv = func(data.xv)
    if data.yv:dim() == 1 then
      data.yv:resize(opt.nValid, 1)
    end
  end

  return data
end

data  = synthesize()
model = bot7.models.dngo(expt)
bot   = bot7.bots.bayesopt(expt, benchmarks[expt.func], {model=model})
bot:run_experiment()




