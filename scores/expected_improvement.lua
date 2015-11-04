------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Expected Improvement (EI) acquisition function:

  EI(fval; fmin) := max(0, E[fmin - fval])

The (optional) 'tradeoff' parameter helps govern
the balance between exploration and exploitation.

Authored: 2015-09-16 (jwilson)
Modified: 2015-11-04
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local math  = require('math')

------------------------------------------------
--                          expected_improvement
------------------------------------------------
local title  = 'bot7.scores.expected_improvement'
local parent = 'bot7.scores.abstract'
local EI, parent = torch.class(title, parent)

function EI:__init(config)
  parent.__init(self)
  local config = config or {}
  config['tradeoff']   = config.tradeoff or 0.0
  config['nFantasies'] = config.nFantasies or 100
  self.config          = config
end

function EI:__call__(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  local hyp    = hyp or model.hyp
  local config = config or self.config
  local ei = EI.eval(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  collectgarbage()
  return ei
end

function EI.eval(model, hyp, X_obs, Y_obs, X_hid, X_pend, config)
  local nObs, nHid = X_obs:size(1), X_hid:size(1)
  if (X_obs:dim() == 1) then X_obs:resize(1, nObs); nObs=1 end
  if (X_hid:dim() == 1) then X_hid:resize(1, nHid); nHid=1 end

  ---------------- Fantasize outcomes for pending jobs:
  if torch.isTensor(X_pend) and X_pend:size(1) > 0 then
    local nPend = X_pend:size(1)
    if (X_pend:dim() == 1) then X_pend:resize(1 ,nPend); nPend=1 end
    local nOP   = nObs + nPend

    -------- Generate fantasies and append to _obs tensors
    local X_obs, Y_obs, Y_pend = X_obs, Y_obs, nil
    Y_pend = model:fantasize(config.nFantasies, X_obs, Y_obs, X_pend, hyp)
    X_obs  = X_obs:cat(X_pend, 1)
    Y_obs  = utils.tensor.vect(Y_obs):repeatTensor(1, config.nFantasies):cat(Y_pend, 1)
  end

  -------- Compute predictive posterior at X_hid
  local pred  = model:predict(X_obs, Y_obs, X_hid, hyp, {mean=true, var=true})
  local fmins = Y_obs:min(1)

  return EI.compute(pred.mean, pred.var, fmins, config)
end

function EI.compute(fval, fvar, fmin, tradeoff)
  local tradeoff = config.tradeoff or 0.0

  ---- Compute required terms
  local sigma = torch.sqrt(fvar):squeeze()
  local imprv = torch.add(fmin:expandAs(fval), -fval):add(-tradeoff)
  local zvals = torch.cdiv(imprv, sigma)

  ---- Calculate expected improvement
  local ei = imprv:cmul(utils.math.norm_cdf(zvals))
                  :add(sigma:cmul(utils.math.norm_pdf(zvals)))
                  :clamp(0, math.huge)

  -------- Take sample average
  if ei:dim() > 1 and ei:size(2) > 1 then
    ei = ei:mean(2)
  end
  
  return ei
end
