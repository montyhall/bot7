------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple class for training neural nets.

Expects data to be passed in as:

  ------------------------------
  | Field | Content            |
  ------------------------------
  |  xr   | Training Inputs    |
  |  yr   | Training Targets   |
  |  xe   | Test Inputs        |
  |  ye   | Test Targets       |
  |  xv   | Validation Inputs  |
  |  yv   | Validation Targets |
  ------------------------------

Authored: 2015-09-30 (jwilson)
Modified: 2015-10-13
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local NN    = require('nn')

------------------------------------------------
--                                       trainer
------------------------------------------------
local title  = 'bot7.bots.trainer'
local parent = 'bot7.bots.abstract'
local bot, parent = torch.class(title, parent)

function bot:__init(config, data)
  parent.__init(self, config)
  self.config = self:configure(config or {}, data)
end

function bot:__call__(data, model, optimizer, criterion, config)
  self.config     = self:configure(config or self.config or {}, data)
  local model     = model or self:build(self.config.model)
  local optimizer = optimizer or optim.sgd
  local criterion = criterion or nn[self.config.optim.criterion]()
  return self:train(data, model, optimizer, criterion)
end

function bot:configure(config, data)
  local config = config or {}

  ---------------- User Interfacing 
  local UI = config.UI or {}
  UI['verbose']  = UI.verbose or 1
  UI['save']     = UI.save or 0
  UI['msg_freq'] = UI.msg_freq or 10
  config['UI']   = UI

  ---------------- Training Schedule
  local sched = config.schedule or {}
  sched['nEpochs']   = sched.nEpochs or 100
  sched['batchsize'] = sched.batchsize or 32
  sched['eval_freq'] = sched.eval_freq or sched.nEpochs
  config['schedule'] = sched

  ---------------- Model settings
  local model      = config.model or {}
  model['problem'] = model.problem or 'classification'
  model['output']  = model.output or 'LogSoftMax'
  if data then
    model['xDim']  = data.xr:size(2)
    if data.yr:dim() > 1 then 
      model['yDim']  = data.yr:size(2)
      model.problem  = 'regression'
    elseif model.problem == 'classification' then
      model['yDim']  = data.yr:max()
    else
      model['yDim']  = 1
    end
  end
  model['nLayers'] = model.nLayers or 3
  model['nHidden'] = model.nHidden or {100, 100, 100, 100}
  model['dropout'] = model.dropout or {0, 0, 0, 0}
  config['model']  = model

  ---------------- Optimization Settings
  local optim                = config.optim or {}
  optim['learningRate']      = optim.learningRate or 1e-2
  optim['learningRateDecay'] = optim.learningRateDecay or 1e-4
  optim['weightDecay']       = optim.weightDecay  or 1e-5
  optim['momentum']          = optim.momentum     or 9e-1
  optim['criterion']         = optim.criterion    or 'ClassNLLCriterion'
  config['optim']            = optim
  return config
end

function bot:build(config)
  local config = config or self.config.model
  local dims   = torch.cat(torch.Tensor(config.nHidden), 
                           torch.Tensor{config.yDim})

  -------- Input Layer 
  local model = NN.Sequential()
  model:add(nn.Linear(config.xDim, dims[1]))

  -------- Hidden Layers
  for layer = 1, config.nLayers+1 do
    model:add(nn.Tanh())
    dropout = config.dropout[layer]
    if dropout > 0 and dropout < 1 then
      model:add(nn.Dropout(dropout))
    end
    model:add(nn.Linear(dims[layer], dims[layer+1]))
  end

  -------- Output Layer
  if nn[config.output] ~= nil then
    model:add(nn[config.output]())
  end

  return model
end

function bot:train(data, model, optimizer, criterion, state)
  local config  = self.config
  local sched   = config.schedule
  local UI      = config.UI
  local state   = state or {} -- state for optimizer

  -------- Shape Information
  local counts = {r=data.xr:size(1), e=0, v=0}
  if data.xe and data.ye then counts.e = data.xe:size(1) end
  if data.xv and data.yv then counts.v = data.xv:size(1) end

  local xDim, nCol = data.xr:size(2), 0

  local yDim, targets
  if data.yr:dim() > 1 then
    yDim    = data.yr:size(2) 
    targets = torch.Tensor(sched.batchsize, yDim)
  else
    yDim    = 1
    targets = torch.Tensor(sched.batchsize)
  end

  -------- Local Aliases / Tensor Preallocation
  local W, grad   = model:getParameters()
  local inputs    = torch.Tensor(sched.batchsize, xDim)
  
  local suffix    = {'e', 'r', 'v'}
  local nSets, nEvals, trace = 0, 0, {} 
  if sched.eval_freq > 0 then
    nEvals = math.floor(sched.nEpochs/sched.eval_freq)
    if counts.r > 0 then nSets = nSets + 1 end
    if counts.e > 0 then nSets = nSets + 1 end
    if counts.v > 0 then nSets = nSets + 1 end
    trace = torch.Tensor(nEvals, nSets+1) -- first position stores epoch
    trace:select(2,1):fill(-1); trace:narrow(2, 2, nSets):fill(0/0)
  end
  
  local shuffle = function(suffix)
    assert(counts[suffix] > 0)
    local perm = torch.randperm(counts[suffix],'torch.LongTensor')
    data['x'..suffix] = data['x'..suffix]:index(1,perm)
    data['y'..suffix] = data['y'..suffix]:index(1,perm)
  end

  local closure = function(x)
    if x ~= W then W:copy(x) end
    grad:zero() -- reset grads
    
    local outputs = model:forward(inputs)
    local fval  = criterion:forward(outputs, targets)
    local df_dw = criterion:backward(outputs, targets) -- ~= df/dW

    model:backward(inputs, df_dw)
    fval = fval/sched.batchsize
    return fval, grad
  end

  local eval = function(suffix)
    local suffix = suffix or 'r'
    model:evaluate()
    local head, size = 0, 0
    local batchsize  = sched.batchsize
    local outputs, loss = nil, 0
    for tail = 1, counts[suffix], batchsize do
      head = math.min(head + batchsize, counts[suffix])
      nCol = head-tail+1
      inputs:resize(nCol,xDim):copy(data['x'..suffix]:sub(tail, head))
      if yDim > 1 then
        targets:resize(nCol,yDim):copy(data['y'..suffix]:sub(tail, head))
      else
        targets:resize(nCol):copy(data['y'..suffix]:sub(tail, head))
      end

      outputs = model:forward(inputs)
      loss    = loss + criterion:forward(outputs, targets)
    end
    loss = loss/math.ceil(counts[suffix]/batchsize)
    return loss
  end

  local update = function(suffix)
    local suffix = suffix or 'r'
    model:training()
    shuffle(suffix)
    local head, size = 0, 0
    local batchsize  = sched.batchsize
    for tail = 1, counts[suffix], batchsize do
      head = math.min(head + batchsize, counts[suffix])
      nCol = head-tail+1
      inputs:resize(nCol,xDim):copy(data['x'..suffix]:sub(tail, head))
      if yDim > 1 then
        targets:resize(nCol,yDim):copy(data['y'..suffix]:sub(tail, head))
      else
        targets:resize(nCol):copy(data['y'..suffix]:sub(tail, head))
      end
      optimizer(closure, W, config.optim, state)
    end
  end

  local report = function(epoch)
    if epoch % config.UI.msg_freq == 0 then
      local t   = epoch / UI.msg_freq
      local msg = string.format('Epoch: %d of %d', epoch, sched.nEpochs)
      print('================================================')
      print(string.rep(' ', 48-msg:len()).. msg)
      print('================================================')
      local prev = utils.nanop(torch.max, trace:select(2,1))
      local idx  = trace:select(2,1):eq(prev):nonzero():storage()[1]
      local loss = trace:select(1, idx)
      idx, msg = 2, ''
      for k = 1, nSets do
        if counts[suffix[k]] > 0 then 
          if msg ~= '' then msg = msg .. ' | ' end
          msg = msg .. string.format('%.2e (%s)', loss[idx], suffix[k]:upper())
          idx = idx + 1
        end
      end
      print(string.format('> Loss at epoch %d: %s', prev, msg))
      local nEvals = config.optim.evalCounter or 0
      local lRate  = config.optim.learningRate/(1 + nEvals*config.optim.learningRateDecay)
      print(string.format('Learning Rate: %.2E',lRate))
      print('')
    end
  end

  -------- Main Training Loop
  for epoch = 1, sched.nEpochs do

    ---- Perform parameter update
    update('r')

    ---- Evaluate model on provided data
    if epoch % sched.eval_freq == 0 then
      local t, idx  = epoch / sched.eval_freq, 2
      trace[{t, 1}] = epoch
      for k = 1, nSets do
        if counts[suffix[k]] > 0 then 
          trace[{t, idx}] = eval(suffix[k])
          idx = idx + 1
        end
      end
    end

    ---- Report current status to user
    report(epoch)
  end
  return model, trace
end
