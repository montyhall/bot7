------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Bayesian Optimization of benchmarking functions.

Authored: 2015-09-18 (jwilson)
Modified: 2016-02-24
--]]

---------------- External Dependencies
local paths = require('paths')
local bot7  = require('bot7')
local hyperparam = bot7.hyperparam
local benchmarks = require('bot7.benchmarks')

------------------------------------------------
--                                Initialization
------------------------------------------------

---------------- Argument Parsing
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-verbose',   1, 'specify which print level')
cmd:option('-bot',       'bo', 'specify which bot to use: {bo, rs}')
cmd:option('-benchmark', 'braninhoo', 'specify which function to optimize: '..
                                      '{braninhoo, hartmann3, hartmann6}')

cmd:option('-nInitial',   2, 'specify number of initial candidates to sample at random')
cmd:option('-budget',     100,'specify budget (#nominees) for experiment')
cmd:option('-xDim',       100,'specify input dimensionality for experiment')
cmd:option('-yDim',       1,'specify output dimensionality for experiment')
cmd:option('-noisy',      false, 'specify observations as noisy')
cmd:option('-grid_size',  20000, 'specify size of candidate grid')
cmd:option('-grid_type',  'sobol', 'specify type for candidate grid')
cmd:option('-mins', '',   'specify minima for inputs (defaults to 0.0)')
cmd:option('-maxes',      '', 'specify maxima for inputs (defaults to 1.0)')
cmd:option('-score',      'ei', 'specify acquisition function to be used by bot; {ei, ucb}')

cmd:text()
opt = cmd:parse(arg or {})

---- Manual override xDim for experiment
if     (opt.benchmark == 'braninhoo') then
  opt.xDim = 2
elseif (opt.benchmark == 'hartmann3') then
  opt.xDim = 3
elseif (opt.benchmark == 'hartmann6') then
  opt.xDim = 6
end

---------------- Experiment Configuration
local expt = 
{
  xDim   = opt.xDim,
  yDim   = opt.yDim,
  budget = opt.budget,
}

expt.model = {noiseless = not opt.noisy} 
expt.grid  = {type = opt.grid_type, size = opt.grid_size}
expt.bot   = {type = opt.bot, nInitial = opt.nInitial}

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
elseif (opt.score == 'ucb') then
  expt.score['type'] = 'confidence_bound'
end

---- Set metatables
for key, val in pairs(expt) do
  if type(val) == 'table' then
    setmetatable(val, {__index = expt})
  end
end

------------------------------------------------
--                                 run_benchmark
------------------------------------------------
function run_benchmark(expt)
  local hypers = {}
  for k = 1, opt.xDim do
    hypers[k] = hyperparam('x'..k, 0, 1)
  end

  -------- Initialize bot
  local bot
  if expt.bot.type == 'bo' then
    bot = bot7.bots.bayesopt(benchmarks[opt.benchmark], hypers, expt)
  elseif expt.bot.type == 'rs' then
    bot = bot7.bots.random_search(benchmarks[opt.benchmark], hypers, expt)
  else
    print('Error: Unrecognized bot specified; aborting...')
    return
  end

  -------- Perform experiment
  bot:run_experiment()
end

run_benchmark(expt)