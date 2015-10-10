------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Bayesian Optimization of benchmarking functions.

Authored: 2015-09-18 (jwilson)
Modified: 2015-10-10
--]]

---------------- External Dependencies
paths = require('paths')
bot7  = require('bot7')

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
cmd:option('-mins', '',   'specify minima for inputs (defaults to 0.0)')
cmd:option('-maxes',      '', 'specify maxima for inputs (defaults to 1.0)')
cmd:option('-score',      'ei', 'specify acquisition function to be used by bot; {ei, ucb}')

cmd:text()
opt = cmd:parse(arg or {})

---------------- Experiment Configuration
local expt = 
{  

  bot      = opt.bot,
  func     = opt.benchmark,
  nInitial = opt.nInitial,
  budget   = opt.budget,
  xDim     = opt.xDim,
  yDim     = opt.yDim,
  model    = {noiseless = not opt.noisy},
  grid     = {size = opt.grid_size}
}

---- Manual override xDim for experiment
if     (opt.benchmark == 'braninhoo') then
  opt.xDim = 2
elseif (opt.benchmark == 'hartmann3') then
  opt.xDim = 3
elseif (opt.benchmark == 'hartmann6') then
  opt.xDim = 6
end
expt.xDim, expt.grid['xDim'] = opt.xDim, opt.xDim

---- Establish minimia/maxima
if (opt.mins ~= '') then
  loadstring('opt.mins='..opt.mins)()
else
  opt.mins = torch.zeros(1, opt.xDim)
  expt.grid.mins = opt.mins
end

if (opt.maxes ~= '') then
  loadstring('opt.maxes='..opt.maxes)()
else
  opt.maxes = torch.ones(1, opt.xDim)
  expt.grid.maxes = opt.maxes
end

---- Choose acquistion function
expt['score'] = {}
if     (opt.score == 'ei') then
  expt.score['type'] = 'expected_improvement'
elseif (opt.score == 'ucb') then
  expt.score['type'] = 'confidence_bound'
end


------------------------------------------------
--                                 run_benchmark
------------------------------------------------
function run_benchmark(expt)

  -------- Load benchmark function
  paths.dofile('benchmarks/' .. expt.func .. '.lua')

  -------- Initialize bot
  local bot
  if expt.bot == 'bo' then
    bot = bot7.bots.bayesopt(expt, _G[expt.func])
  elseif expt.bot == 'rs' then
    bot = bot7.bots.random_search(expt, _G[expt.func])
  else
    print('Error: Unrecognized bot specified; aborting...')
    return
  end

  -------- Perform experiment
  bot:run_experiment()
end

run_benchmark(expt)