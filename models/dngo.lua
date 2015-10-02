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
Modified: 2015-10-02
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local BLR   = require('gp.models').bayes_linear

------------------------------------------------
--                                          dngo
------------------------------------------------
local title  = 'bot7.models.dngo'
local parent = 'bot7.models.metamodel'
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
  self.criterion = criterion or nn.MSECriterion()
  self.optimizer = cache.optimizer or optim.sgd
  self:init(X,Y)
end


function dngo:init(X, Y)
  -------- Short-circuit
  if not (X and Y) then return end

  -------- Local Variables
  local config = self.config

  -------- Construct model (if necessary)
  if not self.network then 
    self.network = self:build(X, Y, config.network)
  end

  -------- Update network, do first to ensure tensor shapes
  self:update(X, Y, config.init or config.update)

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

function dngo:cov(hyp, X0, X1)
  local config    = self.config
  local network   = self.network
  local predictor = self.predictor

  local N0, N1  = X0:size(1), X1:size(1)
  local N, xDim = N0 + N1, X0:size(2)

  local outputs = torch.FloatTensor(N, config.zDim) -- cpu
  local inputs  = torch.Tensor(config.batchsize, xDim)
  local head    = config.batchsize+1

  for tail = 1, N0, config.batchsize do
    inputs:resize(head-tail):copy(X0:sub(tail, head))
    network:forward(inputs)
    outputs:sub(tail, head):copy(self.basis.output)
    head = head + tail
  end

  return predictor:cov_func(outputs)
end

function dngo:predict(X0, Y0, X1, hyp, req, skip)
  local hyp       = hyp or 'marginalize'
  local req       = req or {mean=true, var=true}
  local network   = self.network
  local predictor = self.predictor
  local config    = self.config
  local batchsize = config.batchsize or config.update.batchsize -- hack

  -------- Size Information 
  local N0, N1 = X0:size(1), X1:size(1)
  local xDim   = X0:size(2)

  -------- Tensor Preallotation
  local Z0 = torch.DoubleTensor(N0, config.zDim)
  local Z1 = torch.DoubleTensor(N1, config.zDim)
  local x  = torch.Tensor(batchsize, xDim) 

  -------- Update network / basis function
  if not skip then self:update(X0, Y0) end
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

function dngo:update(X, Y, config)
  -------- Local aliases
  local config    = config or self.config.update
  local network   = self.network
  local criterion = self.criterion
  local optimizer = self.optimizer
  local W, grad   = network:getParameters()
  local batchsize = config.batchsize

  -------- Shape information
  local N    = X:size(1)
  local xDim = X:size(2)
  local yDim = Y:size(2)
  
  -------- Tensor preallocation
  local inputs  = torch.Tensor(batchsize, xDim)
  local targets = torch.Tensor(batchsize, yDim)

  local nEvals = config.evalCounter or 0
  local lRate  = config.learningRate/(1 + nEvals*config.learningRateDecay)
  print(string.format('Learning Rate: %.2E, nEvals: %d',lRate, nEvals))

  -------- Functional closure 
  local closure = function(x)
    if x ~= W then W:copy(x) end
    grad:zero() -- reset grads
    local outputs = network:forward(inputs) -- fprop
    local fval    = criterion:forward(outputs, targets)
    local df_dw   = criterion:backward(outputs, targets)
    network:backward(inputs, df_dw) -- bprop
    fval = fval/batchsize
    return fval, grad
  end

  -------- Main training loop
  -- config.evalCounter = 0 -- reset evalCounter?
  network:training()
  local order, idx, head, size
  for epoch = 1, config.nEpochs do
    order = torch.randperm(N, 'torch.LongTensor')
    head  = 0
    for tail = 1, N, batchsize do
      head = math.min(head + batchsize, N)
      size = head-tail+1
      idx  = order:sub(tail, head)
      inputs:resize(size,xDim):copy(X:index(1, idx))
      targets:resize(size,yDim):copy(Y:index(1, idx))
      optimizer(closure, W, config)
    end
  end
end


function dngo:build(X, Y, config)
  local config = config or {}

  ---------------- Default settings
  config['xDim']    = X:size(2)
  config['yDim']    = Y:size(2)
  config['output']  = config.output or ''
  config['weights'] = config.weights or 'Linear'
  config['nLayers'] = config.nLayers or 3
  config['nHidden'] = config.nHidden or 50
  config['dropout'] = config.dropout or 0
  config['activation'] = config.activation or 'Tanh'

  -------- Transform listed terms to tables 
  local nLayers = config.nLayers
  local keys = {'nHidden', 'dropout', 'weights', 'activation'}
  for k = 1, utils.tbl_size(keys) do
    if type(config[keys[k]]) ~= 'table' then
      local tbl = {}
      for l = 1, nLayers + 1 do 
        tbl[l] = config[keys[k]] 
      end
      config[keys[k]] = tbl
    end
  end
  config.nHidden[nLayers+1] = nil -- remove possible extra

  -------- Input-layer settings
  config.dropout[1] = 0.0

  -------- Output-layer settings
  config.activation[nLayers+1] = config.output
  config.dropout[nLayers+1]    = 0.0
  
  local dims = torch.cat({torch.Tensor{config.xDim},
                          torch.Tensor(config.nHidden), 
                          torch.Tensor{config.yDim}})

  -------- Iteratively construct network
  local network = nn.Sequential()
  for l = 1, nLayers+1 do
    -------- Weights Module 
    if nn[config.weights[l]] then 
        network:add(nn[config.weights[l]](dims[l], dims[l+1]))
    end
    -------- Activation Function
    if nn[config.activation[l]] then
      network:add(nn[config.activation[l]]())
    end
    -------- Dropout Layer
    local rate = config.dropout[l]
    if rate > 0 and rate < 1 then
      network:add(nn.Dropout(rate))
    end
  end


  -- -------- Input Layer 
  -- local model = NN.Sequential()
  -- model:add(nn.Linear(config.xDim, dims[1]))

  -- -------- Hidden Layers
  -- for layer = 1, config.nLayers+1 do
  --   model:add(nn[config.activation[layer]]())
  --   dropout = config.dropout[layer]
  --   if dropout > 0 and dropout < 1 then
  --     model:add(nn.Dropout(dropout))
  --   end
  --   model:add(nn.Linear(dims[layer], dims[layer+1]))
  -- end

  -- -------- Output Layer
  -- if nn[config.output] then
  --   model:add(nn[config.output]())
  -- end
  return network
end