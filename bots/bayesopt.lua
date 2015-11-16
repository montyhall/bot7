------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Bayesian Optimization bot class for bot7.

Authored: 2015-09-18 (jwilson)
Modified: 2015-11-16
--]]

---------------- External Dependencies
local utils  = require('bot7.utils')
local Models = require('bot7.models')
local Scores = require('bot7.scores')
local math   = require('math')

------------------------------------------------
--                                      bayesopt
------------------------------------------------
local title  = 'bot7.bots.bayesopt'
local parent = 'bot7.bots.abstract'
local bot, parent = torch.class(title, parent)

function bot:__init(objective, hypers, config, cache)
  local cache = cache or {}
  parent.__init(self, objective, hypers, config, cache)
  self.config  = self:configure(config)
  local config = self.config

  -------- Initialize model / acquisition function
  self.model = cache.model or Models[config.model.type](config.model)
  self.score = cache.score or Scores[config.score.type](config.score)
end

function bot:configure(config)
  local config = parent.configure(self, config)

  -------- Model
  local model      = config.model  or {}
  model['type']    = model.type    or 'gp_regressor'
  model['kernel']  = model.kernel  or 'ardse'
  model['nzModel'] = model.nzModel or 'GaussianNoise_iso'
  model['mean']    = model.mean    or 'constant'
  model['sampler'] = model.sampler or 'slice'
  config['model']  = model

  -------- Score
  local score      = config.score or {}
  score['type']    = score.type or 'expected_improvement'
  config['score']  = score

  return config
end

---------------- Evaluate candidates according to score
function bot:eval(candidates)
  -------- Local aliases
  local X_obs  = self.observed
  local Y_obs  = self.responses
  local X_hid  = self.candidates
  local X_pend = self.pending

  -------- Assess acquisition function while marginalizing
  -- over hyperparameters via MC integration
  if self.model:class() == 'bot7.models.dngo' then
    return self.score(self.model, nil, X_obs, Y_obs, X_hid)
  else
    local samples  = self.model:sample_hypers(X_obs, Y_obs)
    local score    = torch.zeros(X_hid:size(1))
    local nSamples = self.config.bot.nSamples
    local hyp      = torch.Tensor()

    for s = 1,nSamples do
      hyp = self.model:sample_hypers(X_obs, Y_obs, nil, nil, true)
      hyp = self.model:parse_hypers(hyp)
      score:add(self.score(self.model, hyp, X_obs, Y_obs, X_hid))
      collectgarbage()
    end
    score:div(nSamples)
    return score
  end
end

---------------- Nominate a candidate
function bot:nominate(candidates)
  local candidates = candidates or self.candidates
  local idx

  -------- Select initial points randomly
  if self.nTrials <= self.config.bot.nInitial then
    idx = torch.rand(1):mul(candidates:size(1)):long():add(1)

  -------- Nominate according to acquistion values
  else
    local score = self:eval(candidates)
    min, idx  = score:max(1)
  end
  return idx
end
