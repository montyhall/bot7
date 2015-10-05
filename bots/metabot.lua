------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstact base class for bot7 bots.

Authored: 2015-09-18 (jwilson)
Modified: 2015-10-05
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local Grids = require('bot7.grids')

------------------------------------------------
--                                       metabot
------------------------------------------------
local title = 'bot7.bots.metabot'
local bot   = torch.class(title)

function bot:__init(config, objective, cache)
  -------- Establish settings
  local cache    = cache or {}
  self.objective = cache.objective or objective
  self.config    = self:configure(cache.config or config)
  local config   = self.config

  -------- Generate Candidate Grid
  local grid      = Grids[config.grid.type]()
  self.candidates = cache.candidates or grid(config.grid)
  self.responses  = cache.responses or nil
  self.observed   = cache.observed or nil
  if self.observed then
    self.nTrials = self.observed:size(1)
  else
    self.nTrials = 0
  end

  -------- Preallocate result struct
  self.best      = {}
  self.best['x'] = torch.Tensor(1, config.xDim)
  if self.responses then 
    self.best['y'], self.best['t'] = self.responses:min(1)
  else
    self.best['y'] = torch.Tensor(config.yDim):fill(math.huge)
    self.best['t'] = -torch.ones(1)
  end

end

function bot:configure(config)
  local config = utils.deepcopy(config)
  assert(config.xDim) -- must be provided

  ---------------- Default Settings
  -------- Bot
  config['verbose']  = config.verbose or 3
  config['budget']   = config.budget or 100
  config['msg_freq'] = config.msg_freq or 1
  config['nInitial'] = config.nInitial or 2
  config['nSamples'] = config.nSamples or 10
  config['save']     = config.save or false

  local score        = config.score or {}
  score['type']      = score.type or 'expected_improvement'
  config['score']    = score

  -------- Grid
  local grid     = config.grid or {}
  grid['type']   = grid.type or 'random'
  grid['size']   = grid.size or 2e4
  grid['dims']   = grid.dims or config.xDim
  grid['mins']   = grid.mins or torch.zeros(1, grid.dims)
  grid['maxes']  = grid.maxes or torch.ones(1, grid.dims)
  config['grid'] = grid

  return config
end


function bot:run_trial()
  -------- Increment trial counter
  self.nTrials = self.nTrials + 1

  -------- Nominate candidate and mark as pending
  local idx    = self:nominate()
  self.pending, self.candidates = utils.steal(self.pending, self.candidates, idx)

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
    utils.steal(self.observed, self.pending, torch.LongTensor{idx})

  -------- Initialize model w/ values
  if self.model and self.nTrials == self.config.nInitial then
    self.model:init(self.observed, self.responses)
  end

  return nominee, y
end


function bot:run_experiment()
  local x, y
  for t = 1, self.config.budget do
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
  if self.best.y:gt(y):all() then
    self.best.t = self.nTrials
    self.best.x = x
    self.best.y = y
  end
end

function bot:progress_report(t, x, y)
  if t % self.config.msg_freq == 0 then
    local config = self.config
    -------- Print Levels < 1 
    if config.verbose < 1 then return end

    local msg = string.format('Trial: %d of %d', t, config.budget)
    print('================================================')
    print(string.rep(' ', 48-msg:len()).. msg)
    print('================================================')

    local nHyp = x:nElement()
          xCol = math.ceil(math.sqrt(nHyp))
          xRow = math.ceil(nHyp/ xCol)
    local nObs = y:nElement()
          yCol = math.ceil(math.sqrt(nObs))
          yRow = math.ceil(nObs/ yCol)

    local y, best_y = y, self.best.y
    if yCol*yRow > nObs then
      local pad = torch.Tensor(yCol*yRow - nObs):fill(0/0)
      y, best_y = torch.cat(y, pad, 1), torch.cat(best_y, pad, 1)
    else
      y, best_y = y:clone(), best_y:clone()
    end
    y:resize(yCol, yRow); best_y:resize(yCol, yRow)

    -------- Print Level 1
    if config.verbose == 1 then
      print(string.format('> Best response (#%d):', utils.as_val(self.best.t)))
      print(utils.tnsr2str(best_y) ..'\n')
      print('> Most recent:')
      print(utils.tnsr2str(y))
      return
    end

    -------- Print Levels 2
    local x, best_x = x, self.best.x
    if xCol*xRow > nHyp then
      local pad = torch.Tensor(xCol*xRow - nHyp):fill(0/0)
      x, best_x = torch.cat(x, pad, 1), torch.cat(best_x, pad, 1)
    else
      x, best_x = x:clone(), best_x:clone()
    end
    x:resize(xCol, xRow)
    best_x:resize(xCol, xRow)

    print(string.format('> Best seen (#%d):', utils.as_val(self.best.t)))
    print('Response:\n' .. utils.tnsr2str(best_y) .. '\n')
    print('Hypers:\n' .. utils.tnsr2str(best_x))

    print('\n> Most recent:')
    print('Response:\n' .. utils.tnsr2str(y) .. '\n')
    if config.verbose == 2 then return end

    -------- Print Level 3
    print('Hypers:\n' .. utils.tnsr2str(x))
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
