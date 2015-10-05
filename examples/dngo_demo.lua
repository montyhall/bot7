------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple demo for DNGO bot.

Target Article:
"Scalable Bayesian Optimization Using Deep 
Neural Networks" (Snoek et. al 2015)


Authored: 2015-09-30 (jwilson)
Modified: 2015-10-05
--]]

---------------- External Dependencies
local paths = require('paths')
local bot7  = require('bot7')
local utils = bot7.utils

------------------------------------------------
--                                Initialization
------------------------------------------------

---------------- Argument Parsing
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-verbose',   1, 'specify which print level')
cmd:option('-bot',       'bo', 'specify which bot to use: {bo, rs}')
cmd:option('-benchmark', 'hartmann6', 'specify which function to optimize')
cmd:option('-msg_freq',  1, 'interval between progress reports (in terms of epochs)')

cmd:option('-budget',    300, 'number of candidate nominations (queries)')
cmd:option('-nTrain',    0, 'number of training instances to pass to network')
cmd:option('-nTest',     0, 'number of test instances to pass to network')
cmd:option('-nValid',    0, 'number of validation instances to pass to network')
cmd:option('-xDim',      100, 'specify input dimensionality for experiment')
cmd:option('-yDim',      1,   'specify output dimensionality for experiment')
cmd:option('-noisy',     false, 'specify observations as noisy')
cmd:option('-grid_size', 20000, 'specify size of candidate grid')
cmd:option('-grid_type', 'random', 'specify type of grid to employ')
cmd:option('-mins', '',  'specify minima for inputs (defaults to 0.0)')
cmd:option('-maxes',     '', 'specify maxima for inputs (defaults to 1.0)')

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

------ Experiment Settings
local expt = 
{  
  bot    = opt.bot,
  func   = opt.benchmark,
  budget = opt.budget,
  xDim   = opt.xDim,
  yDim   = opt.yDim,
  model  = {noiseless = not opt.noisy},
  grid   = {size = opt.grid_size},
  nInitial = 2,
  msg_freq = opt.msg_freq,
}

-------- Grid Settings
local grid     = expt.grid or {}
grid['type']   = opt.grid_type or 'random'
grid['size']   = grid.size or 2e4
grid['dims']   = grid.dims or expt.xDim
grid['mins']   = grid.mins or torch.zeros(1, grid.dims)
grid['maxes']  = grid.maxes or torch.ones(1, grid.dims)
expt['grid']   = grid

-- ---------------- Training Schedule
-- local sched = expt.schedule or {}
-- sched['nEpochs']   = sched.nEpochs or 100
-- sched['batchsize'] = sched.batchsize or 32
-- sched['eval_freq'] = sched.eval_freq or 10
-- expt['schedule'] = sched

---------------- Network Update Settings
local update                = expt.update or {}
update['nEpochs']           = update.nEpochs or 100
update['batchsize']         = update.batchsize  or 32
update['criterion']         = update.criterion or 'MSECriterion'
update['momentum']          = update.momentum  or 9e-1
update['weightDecay']       = update.weightDecay  or 5e-4
update['learningRate']      = update.learningRate or 1e-1
update['learningRateDecay'] = update.learningRateDecay or 1e-5
expt['update']              = update

---------------- Network Initialization Settings
local init                = expt.init or {}
init['nEpochs']           = init.nEpochs or 100
init['batchsize']         = init.batchsize  or 32
init['criterion']         = init.criterion or 'MSECriterion'
init['momentum']          = init.momentum  or 9e-1
init['weightDecay']       = init.weightDecay  or 1e-3
init['learningRate']      = init.learningRate or 1e-1
init['learningRateDecay'] = init.learningRateDecay or 1e-6
expt['init']              = init

------------------------------------------------
--                                  bayesNN_demo
------------------------------------------------

function synthesize()
  ---- Load benchmark function
  paths.dofile('benchmarks/' .. expt.func .. '.lua')

  local config = utils.deepcopy(expt.grid)
  local func   = _G[expt.func]
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

function run_trainer(config, data, network, criterion)
  local config  = config or {}

  local model     = config.model or {}
  model['output'] = model.output or ''
  config['model'] = model
  config['optim'] = config.optim or expt.optim
  config['schedule'] = expt.schedule

  local trainer = bot7.bots.trainer(config, data)
  local network = trainer(data, network, criterion)

  -------- Remove Top Layer to get basis 
  network.modules[utils.tbl_size(network.modules)] = nil
  network.modules[utils.tbl_size(network.modules)] = nil
  return network
end

data    = synthesize()
-- network = run_trainer(nil, data)
-- model   = bot7.models.dngo(network, nil, nil, nil, data.xr, data.yr)

-- criterion = nn[optim.criterion](expt.yDim)
-- pred      = model:predict(data.xr, data.yr, data.xe)
-- loss      = criterion:forward(pred.mean, data.ye)
-- print(string.format('bayesNN loss: %.2e',loss))

model = bot7.models.dngo(expt)
bot   = bot7.bots.bayesopt(expt, _G[expt.func], {model=model})
-- bot = bot7.bots.bayesopt(expt, _G[expt.func], {model=model, observed=data.xr, responses=data.yr})
bot:run_experiment()




