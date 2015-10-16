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
Modified: 2015-10-25
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
    self.network = nnTools.builder(config.network, data)
    -- self.network = self:build(X, Y, config.network)
  end

  -------- Update network; do first to ensure tensor shapes
  nnTools.trainer(self.network, data, config.init or config.update,
                  self.optimizer, self.criterion, self.state)
  -- self:update(X, Y, config.init or config.update)

  -------- Initialize predictor
  if not self.predictor then 
    self.predictor = BLR(nil, nil, X, Y)
  else
    self.predictor:init(X, Y)
  end

  -------- Find basis layer
  local network     = self.network
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

  
  -- if not skip then self:update(X0, Y0) end
  -------- Update network / basis function
  if not skip then
    local config = config.update
    local res = nnTools.trainer(self.network, {xr=X0, yr=Y0}, config, 
                    self.optimizer, self.criterion, self.state)
    print(string.format('Network MSE: %.2e', res[{1,1}]))

    local nEvals = self.state.evalCounter
    local lRate  = config.optimizer.learningRate/
                    (1 + nEvals*config.optimizer.learningRateDecay)
    print(string.format('Learning Rate: %.2E, nEvals: %d',lRate, nEvals))
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

  return predictor:predict(Z0, Y0, Z1, nil, hyp, req)
end

-- function dngo:update(X, Y, config)
--   -------- Local aliases
--   local config    = config or self.config.update
--   local network   = self.network
--   local criterion = self.criterion
--   local optimizer = self.optimizer
--   local W, grad   = network:getParameters()
--   local batchsize = config.batchsize

--   -------- Shape information
--   local N    = X:size(1)
--   local xDim = X:size(2)
--   local yDim = Y:size(2)
  
--   -------- Tensor preallocation
--   local inputs  = torch.Tensor(batchsize, xDim)
--   local targets = torch.Tensor(batchsize, yDim)

--   local nEvals = config.evalCounter or 0
--   local lRate  = config.learningRate/(1 + nEvals*config.learningRateDecay)
--   print(string.format('Learning Rate: %.2E, nEvals: %d',lRate, nEvals))

--   -------- Functional closure 
--   local closure = function(x)
--     if x ~= W then W:copy(x) end
--     grad:zero() -- reset grads
--     local outputs = network:forward(inputs) -- fprop
--     local fval    = criterion:forward(outputs, targets)
--     local df_dw   = criterion:backward(outputs, targets)
--     network:backward(inputs, df_dw) -- bprop
--     fval = fval/batchsize
--     return fval, grad
--   end

--   -------- Main training loop
--   -- config.evalCounter = 0 -- reset evalCounter?
--   network:training()
--   local order, idx, head, size
--   for epoch = 1, config.nEpochs do
--     order = torch.randperm(N, 'torch.LongTensor')
--     head  = 0
--     for tail = 1, N, batchsize do
--       head = math.min(head + batchsize, N)
--       size = head-tail+1
--       idx  = order:sub(tail, head)
--       inputs:resize(size,xDim):copy(X:index(1, idx))
--       targets:resize(size,yDim):copy(Y:index(1, idx))
--       optimizer(closure, W, config)
--     end
--   end
-- end