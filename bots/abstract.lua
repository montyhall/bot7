------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstact base class for bot7 bots.

Authored: 2015-09-18 (jwilson)
Modified: 2016-02-24
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local Grids = require('bot7.grids')

------------------------------------------------
--                                      abstract
------------------------------------------------
local title = 'bot7.bots.abstract'
local bot   = torch.class(title)

function bot:__init(objective, hypers, config, cache)
  -------- Establish settings
  local cache    = cache or {}
  self.hypers    = cache.hypers or hypers
  self.objective = cache.objective or objective
  self.config    = self:configure(cache.config or config)
  local config   = self.config

  -------- Generate Candidate Grid
  self.candidates = cache.candidates
  if not self.candidates then
    if (config.bot.verbose > 1) then
      print('> Generating candidate grid...')
    end
    self.candidates = Grids[config.grid.type](config.grid)()
  end
  self.responses  = cache.responses or nil
  self.observed   = cache.observed or nil
  if self.observed then
    self.nTrials = self.observed:size(1)
  else
    self.nTrials = 0
  end

  -------- Preallocate result struct
  self.best      = {}
  self.best['x'] = torch.Tensor(1, config.grid.dims)
  if self.responses then 
    self.best['y'], self.best['t'] = self.responses:min(1)
  else
    self.best['t'] = -torch.ones(1)
  end

end

function bot:configure(config)
  local config = utils.table.deepcopy(config)
  -- assert(config.xDim) -- must be provided

  ---------------- Default Settings
  -------- Bot
  local bot       = config.bot or {}
  bot['verbose']  = bot.verbose or 3
  bot['budget']   = bot.budget or 100
  bot['msg_freq'] = bot.msg_freq or 1
  bot['nInitial'] = bot.nInitial or 2
  bot['nSamples'] = bot.nSamples or 10
  bot['save']     = bot.save or false
  config['bot']   = bot

  -------- Acquisition Function
  local score     = config.score or {}
  score['type']   = score.type or 'expected_improvement'
  config['score'] = score


  -------- Grid
  local grid, idx = config.grid or {}, 0
  grid['type'] = grid.type or 'sobol'
  grid['size'] = grid.size or 2e4

  bot.name_width = 0
  if not grid.dims then
    grid.dims = 0
    for key, hyper in pairs(self.hypers) do
      grid.dims = grid.dims + hyper.size
      bot.name_width = math.max(bot.name_width, hyper.name:len())
    end
  end

  if not grid.mins then
    grid.mins, idx = torch.zeros(1, grid.dims), 1
    for key, hyper in pairs(self.hypers) do
      grid.mins:narrow(2, idx, hyper.size):copy(hyper.min)
      idx = idx + hyper.size
    end
  end

  if not grid.maxes then
    grid.maxes, idx = torch.zeros(1, grid.dims), 1
    for key, hyper in pairs(self.hypers) do
      grid.maxes:narrow(2, idx, hyper.size):copy(hyper.max)
      idx = idx + hyper.size
    end
  end
  config['grid'] = grid

  return config
end


function bot:run_trial()
  -------- Increment trial counter
  self.nTrials = self.nTrials + 1

  -------- Nominate candidate and mark as pending
  local idx    = self:nominate()
  self.pending, self.candidates = utils.tensor.steal(self.pending, self.candidates, idx)

  idx     = self.pending:size(1) -- update idx
  nominee = self.pending:select(1, idx)

  -------- Pass nominee to blackbox
  local y = self.objective(nominee)

  -------- Format response value (y)
  if not torch.isTensor(y) then
    if type(y) == 'number' then
      y = torch.Tensor{{y}}
    elseif type(y) == 'table' then
      y = torch.Tensor(y)
    end
  end
  if y:dim() == 1 then y:resize(y:nElement(), 1) end

  -------- Store result and mark nominee as observed
  if self.nTrials == 1 then
    self.responses = y
  else
    self.responses = self.responses:cat(y, 1)
  end

  self.observed, self.pending = 
    utils.tensor.steal(self.observed, self.pending, torch.LongTensor{idx})

  -------- Initialize model w/ values
  if self.model and self.nTrials == self.config.bot.nInitial then
    self.model:init(self.observed, self.responses)
  end

  return nominee, y
end


function bot:run_experiment()
  local x, y
  for t = 1, self.config.bot.budget do
    -------- Perform a single trial 
    x, y = self:run_trial()

    -------- Store results
    self:update_best(x,y)

    -------- Display results
    self:progress_report(t, x, y)
  end

  if self.config.save then self:save() end
end

function bot:update_best(x, y)
  if not self.best.y or self.best.y:gt(y):all() then
    self.best.t = self.nTrials
    self.best.x = x
    self.best.y = y
  end
end

function bot:progress_report(t, x, y)
  if t % self.config.bot.msg_freq == 0 then
    local config = self.config
    -------- Print Levels < 1 
    if config.bot.verbose < 1 then return end

    local best = self.best
    local msg = string.format('Trial: %d of %d', t, config.bot.budget)
    utils.ui.printSection(msg)

    -------- Print Level 1
    if y:nElement() == 1 then
       print(string.format('> Best response (#%d): %s', best.t, utils.tensor.string(best.y)))
       print(string.format('> Last response (#%d): %s', t, utils.tensor.string(y)))
    else
      print(string.format('> Best response (#%d):', best.t))
      print(utils.tensor.string(best.y)..'\n')
      print(string.format('> Last response (#%d):', t))
      print(utils.tensor.string(y))
    end

    if config.bot.verbose == 1 then return end

    -------- Print Levels 2 & 3
    local width = math.max(config.bot.name_width, 18)
    local msg = '| ' ..string.rep(' ', math.floor((width-14)/2)) 
                  .. 'Hyperparameter' .. string.rep(' ', math.ceil((width-14)/2))
                  .. ' | Best Seen |'

    if config.bot.verbose > 2 then msg = msg .. ' Previous' end
    msg = msg .. ' |'
    print('\n' .. msg .. '\n' .. string.rep('-', msg:len()))
    for idx, hyper in pairs(self.hypers) do
      msg = string.rep(' ', math.floor((width-hyper.name:len())/2))
            .. hyper.name ..
            string.rep(' ', math.ceil((width-hyper.name:len())/2))
      msg = string.format('%s | %.2e ', msg, hyper:warp(best.x[idx]))
      if config.bot.verbose > 2 then
        msg = string.format('%s | %.2e', msg, hyper:warp(x[idx]))
      end
      print('| '..msg .. ' |')
    end
  end
end

function bot:eval()
  print('Error: eval() method not implemented')
end

function bot:nominate()
  print('Error: nominate() method not implemented')
end



function bot:save()
  local results = {}
  results.best  = self.best
  results.x     = self.observed
  results.y     = self.responses
  torch.save('demo_'..self:class()..'.t7', results)
end

function bot:__call__()
  self:run_experiment()
end

function bot:class()
  return self:__tostring__()
end

function bot:__tostring__()
  return torch.type(self)
end
