------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Deep Network for Global Optimization (DNGO);
neural network with a Bayes Linear regressor
on top.

Target Article:
"Scalable Bayesian Optimization Using Deep 
Neural Networks" (Snoek et. al 2015)

Authored: 2015-09-30 (jwilson)
Modified: 2015-10-27
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local BLR   = require('gp.models').bayes_linear
local nnTools = require('bot7.nnTools')

------------------------------------------------
--                                          dngo
------------------------------------------------
local title  = 'bot7.models.dngo'
local parent = 'bot7.models.abstract'
local dngo, parent = torch.class(title, parent)

--------------------------------
--                Initialization
--------------------------------
function dngo:__init(config, cache, X, Y)
  parent.__init(self)
  local cache    = cache or {}
  self.config    = cache.config or config or {}
  self.network   = cache.network
  self.predictor = cache.predictor
  self.state     = cache.state or {}
  self.criterion = criterion or nn.MSECriterion()
  self.optimizer = cache.optimizer or optim.sgd
  self:init(X,Y)
end


function dngo:init(X, Y)
  -------- Short-circuit
  if not (X and Y) then return end

  -------- Local Variables
  local config = self.config
  local data   = {xr=X, yr=Y}

  -------- Construct model (if necessary)
  if not self.network then
    self.network = nnTools.builder(config.model, data)
  end
  local network = self.network

  -------- Update network; do first to ensure tensor shapes
  nnTools.trainer(network, data, config.init or config.update,
                  self.optimizer, self.criterion, self.state)

  -------- Initialize predictor
  if not self.predictor then 
    self.predictor = BLR(nil, nil, X, Y)
  else
    self.predictor:init(X, Y)
  end

  -------- Find basis layer
  local basis_layer = config.basis_layer 
  local idx, count  = network:size(), 0
  local position    = 1 -- measured from last
  if basis_layer and basis_layer < 0 then
    position    = -basis_layer
    basis_layer = nil
  end
  -- Starting from the top module, iterate backwards
  -- to find k-th from last layer's output module
  while not basis_layer do  
    local mod = network:get(idx)
    if mod.output and count == position then
      basis_layer = idx 
    end
    if mod.weight then  -- Assumes each layer to have
      count = count + 1 -- a single weigthed module
    end
    idx = idx - 1
  end

  -------- Establish basis function
  self.basis  = network:get(basis_layer)
  config.zDim = self.basis.output:size(2) -- Output dim of basis
end

function dngo:predict(X0, Y0, X1, hyp, req, skip)
  local hyp       = hyp or 'marginalize'
  local req       = req or {mean=true, var=true}
  local network   = self.network
  local predictor = self.predictor
  local config    = self.config
  local batchsize = config.update.schedule.batchsize -- hack

  -------- Size Information 
  local N0, N1 = X0:size(1), X1:size(1)
  local xDim   = X0:size(2)

  -------- Tensor Preallotation
  local Z0 = torch.DoubleTensor(N0, config.zDim)
  local Z1 = torch.DoubleTensor(N1, config.zDim)
  local x  = torch.Tensor(batchsize, xDim) 

  -------- Update network / basis function
  if not skip then
    local criterion = self.criterion
    local optimizer = self.optimizer
    local verbose   = config.verbose or 0

    ---- Network error prior to update
    local info, theta0, theta1 = {pre={}}
    if verbose > 2 then
      info.pre['loss'], info.pre['err'] = nnTools.evaluator(network, criterion, X0, Y0)
      if verbose > 3 then
        theta0, _ = self.network:getParameters(); theta0 = theta0:clone()
      end
    end

    ---- Network Update
    info.post = nnTools.trainer(network, {xr=X0, yr=Y0}, config.update, optimizer, criterion, self.state)

    if verbose > 3 then 
      theta1, _  = self.network:getParameters()
      info['dw'] = theta1:dist(theta0)
      theta0, theta1 = nil, nil
    end
    self:report(info)
  end
  network:evaluate()
  
  -------- Transform X0 -> Z0
  local head = 0
  for tail = 1, N0, batchsize do
    head = math.min(head + batchsize, N0)
    x:resize(head-tail+1, xDim):copy(X0:sub(tail, head))
    network:forward(x)
    Z0:sub(tail, head):copy(self.basis.output)
  end

  -------- Transform X1 -> Z1
  head = 0
  for tail = 1, N1, batchsize do
    head = math.min(head + batchsize, N1)
    x:resize(head-tail+1, xDim):copy(X1:sub(tail, head))
    network:forward(x)
    Z1:sub(tail, head):copy(self.basis.output)
  end

  collectgarbage()
  return predictor:predict(Z0, Y0, Z1, nil, hyp, req)
end

function dngo:report(info)
  local pre  = info.pre or {}
  local post = info.post or {}
  local dW   = info.dw
  local msg  = {}

  -------- Learning Rate
  if self.config.verbose and self.config.verbose > 2 then
    local lRate  = self.config.update.optimizer.learningRate/
        (1 + self.state.evalCounter*self.config.update.optimizer.learningRateDecay)
    msg[1] = string.format('> Step-size: %.2E',lRate)
  end

  -------- l2-Norm of gradient
  if dW then
    msg[2] = string.format('> Norm ||dW||: %.2e', dW)
  end

  -------- Classification Error
  if post.err then
    msg[3] = string.format(' %.2e', post.err[{1,1}])
  end

  if pre.err then
    if msg[3] then msg[1] = ' =>' .. msg[3] end
    msg[3] = string.format(' %.2e', pre.err) .. msg[3]
  end

  if msg[3] then
    msg[3] = '> Classif. Error:' .. msg[1]
  end

  -------- Loss on training set
  if post.loss then
    msg[4] = string.format(' %.2e', post.loss[{1,1}])
  end

  if pre.loss then
    if msg[4] then msg[4] =  ' =>' .. msg[4] end
    msg[4] = string.format(' %.2e', pre.loss) .. msg[4]
  end

  if msg[4] then
    msg[4] = '> Training Loss:' .. msg[4]
  end

  -------- Printing
  if utils.tbl_size(msg) > 0 then
    utils.printSection('DNGO Update Report', {width=32})
    for k = 1, 4 do if msg[k] then print(msg[k]) end end
  end
end
