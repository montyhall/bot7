------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple class for training neural networks.

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
Modified: 2015-11-04
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local nn    = require('nn')
local evaluator = require('bot7.nnTools.evaluator')

---------------- Constants
local defaults = 
{
  -------- User Interface
  ui = {verbose=1, save=0, msg_freq=math.huge},

  -------- Training Schedule
  schedule = {gpu=false, nEpochs=100, batchsize=32},

  -------- Criterion
  criterion = {type = 'ClassNLLCriterion'},

  -------- Optimizer
  optimizer = 
  {
    type              = 'sgd',
    learningRate      = 1e-1,
    learningRateDecay = 1e-4,
    weightDecay       = 1e-5,
    momentum          = 9e-1,
  },
}

------------------------------------------------
--                                       trainer
------------------------------------------------
local trainer = function(network, data, config, optimizer, criterion, state)
  -------- Initialization
  local network, data = network, data
  local config = utils.table.update(config, defaults, true)
  if not config.schedule.eval_freq then
    config.schedule.eval_freq = config.schedule.nEpochs
  end
  config.ui.msg_freq = math.max(config.ui.msg_freq, config.schedule.eval_freq)

  local optimizer = optimizer or optim[config.optimizer.type]
  local criterion = criterion or nn[config.criterion.type]()
  local UI, sched = config.ui, config.schedule
  local state     = state or {} -- state for optimizer

  -------- Shape Information
  local counts = {r=data.xr:size(1), e=0, v=0}
  if data.xe and data.ye then counts.e = data.xe:size(1) end
  if data.xv and data.yv then counts.v = data.xv:size(1) end
  local xDim, yDim = data.xr:size(2), 1
  if data.yr:dim() > 1 then yDim = data.yr:size(2)  end

  -------- Detect Problem Type (Classif. / Regression)
  local ttype, classif = data.yr:type(), false
  if ttype == 'torch.LongTensor' 
  or ttype == 'torch.ByteTensor'
  or data.yr:eq(torch.floor(data.yr)):all() then
    classif = true
  end

  -------- Local Aliases / Tensor Preallocation
  local W, grad = network:getParameters()
  local suffix  = {'e', 'r', 'v'}
  local nSets, nEvals, trace = 0, 0, {} 
  if sched.eval_freq > 0 then
    nEvals = math.floor(sched.nEpochs/sched.eval_freq)
    if nEvals > 0 then 
      if counts.r > 0 then nSets = nSets + 1 end
      if counts.e > 0 then nSets = nSets + 1 end
      if counts.v > 0 then nSets = nSets + 1 end
      trace = {loss = torch.Tensor(nEvals, nSets+1):fill(0/0)} -- last position stores epoch
      if classif then
        trace.err = torch.Tensor(nEvals, nSets+1):fill(0/0)
      end
    end
  end

  -------- Allocate Buffers (GPU only)
  local inputs, targets
  if sched.gpu then 
    inputs = torch.CudaTensor(sched.batchsize, xDim)
    if yDim > 1 then
      targets = torch.CudaTensor(sched.batchsize, yDim)
    else
      targets = torch.CudaTensor(sched.batchsize)
    end
  end

  -------- Batch Generator
  local get_batch = function(idx, suffix)
    local suffix = suffix or 'r'
    local X, Y   = data['x'..suffix], data['y'..suffix]
    if config.gpu then
      local nRow = idx:nElement()
      inputs:resize(nRow, xDim):copy(X:index(1, idx))
      if yDim > 1 then
        targets:resize(nRow,yDim):copy(Y:index(1, idx))
      else
        targets:resize(nRow):copy(Y:index(1, idx))
      end
      return inputs, targets
    else
      return X:index(1, idx), Y:index(1, idx)
    end
  end

  -------- Explicit Feature Corruption (regularizer)
  local corrupt = function(x)
    local config, ttype = sched.corruption, x:type()
    local degree, std   = config.degree, config.std
    if (degree > 0  or std > 0) and (not config.type) then
      config.type = 'gaussian'
    end

    if config.type == 'blankout' then
      if config.gpu then 
        return x:cmul(torch.rand(x:size()):gt(degree):cuda())
      else
        return torch.cmul(x, torch.rand(x:size()):gt(degree):type(ttype))
      end

    elseif config.type == 'gaussian' then
      if not std and degree then
        if type(degree) == 'number' then
          std = data.xr:std(1):mul(degree):expandAs(x)
        else
          std = data.xr:std(1):cmul(degree):expandAs(x)
        end
      elseif type(std) == 'number' then 
        std = torch.Tensor{{std}}:type(ttype):expand(x:size(2), 1)
      end

      if config.gpu then 
        return x:add(torch.randn(x:size()):cuda():cmul(std))
      else
        return torch.add(x, torch.randn(x:size()):type(ttype):cmul(std))
      end
    else
      print('Error: Unrecognized noise distribution '..
            'specified; skipping feature corruption...')
      return
    end

  end

  -------- Functional Closure (for optimizer)
  local closure = function(x)
    if x ~= W then W:copy(x) end
    grad:zero() -- reset grads
    
    local outputs = network:forward(inputs)
    local fval  = criterion:forward(outputs, targets)
    local df_dw = criterion:backward(outputs, targets) -- ~= df/dW
    network:backward(inputs, df_dw)
    fval = fval/targets:size(1)
    return fval, grad
  end

  -------- Parameter updates (single epoch)
  local update = function(suffix)
    local suffix = suffix or 'r'
    local perm = torch.randperm(counts[suffix], 'torch.LongTensor')
    local head, size = 0, 0
    local batchsize  = sched.batchsize
    network:training()

    for tail = 1, counts[suffix], batchsize do
      head = math.min(head + batchsize, counts[suffix])
      idx  = perm:sub(tail, head)
      inputs, targets = get_batch(idx, suffix)

      ---- Feature Corruption (Blankout noise)
      if sched.corruption then
        inputs = corrupt(inputs)
      end

      optimizer(closure, W, config.optimizer, state)
    end
  end

  -------- Wrapper for evaluator
  local eval = function(suffix)
    local loss, err = evaluator(network, criterion, data['x'..suffix], data['y'..suffix])
    return loss, err
  end

  -------- Message Printing UI
  local report = function(epoch)
    if UI.msg_freq > 0 and  epoch % UI.msg_freq == 0 then
      local t   = epoch / UI.msg_freq
      local msg = string.format('Epoch: %d of %d', epoch, sched.nEpochs)
      utils.ui.printSection(msg)

      local loss = trace.loss
      local nCol = loss:size(2)
      local prev = utils.math.nanop(torch.max, loss[{{}, nCol}])
      local idx  = loss:select(2, nCol):eq(prev):repeatTensor(nCol, 1):t()
      local loss = loss:maskedSelect(idx)
      local err  = nil
      if trace.err then 
        err = trace.err:maskedSelect(idx)
      end
      idx, msg = 1, ''
      for k = 1, nSets do
        if counts[suffix[k]] > 0 then 
          if msg ~= '' then msg = msg .. ' | ' end
          msg = msg .. string.format('%.2e (%s)', loss[idx], suffix[k]:upper())
          idx = idx + 1
        end
      end

      if err then
        print(string.format('> Loss  at epoch %d: %s', prev, msg))
        idx, msg = 1, ''
        for k = 1, nSets do
          if counts[suffix[k]] > 0 then 
            if msg ~= '' then msg = msg .. ' | ' end
            msg = msg .. string.format('%.2e (%s)', err[idx], suffix[k]:upper())
            idx = idx + 1
          end
        end
        print(string.format('> Error at epoch %d: %s', prev, msg))
      else
        print(string.format('> Loss at epoch %d: %s', prev, msg))
      end

      local nEvals = state.evalCounter or 0
      local lRate  = config.optimizer.learningRate/(1 + nEvals*config.optimizer.learningRateDecay)
      print(string.format('Learning Rate: %.2E',lRate))
      print('')
    end
  end

  -------- Main Training Loop
  for epoch = 1, sched.nEpochs do

    ---- Perform parameter update
    update('r')

    ---- Evaluate network on provided data
    if epoch % sched.eval_freq == 0 then
      local t, idx = epoch / sched.eval_freq, 1
      trace.loss[{t, nSets+1}] = epoch -- record epoch
      if classif then
        trace.err[{t, nSets+1}] = epoch -- record epoch
      end
      for set = 1, 3 do
        if counts[suffix[set]] > 0 then
          local loss, err = eval(suffix[set])
          trace.loss[{t, idx}] = loss
          if classif and err then
            trace.err[{t, idx}] = err
          end
          idx = idx + 1
        end
      end
    end

    ---- Report current status to user
    report(epoch)
  end
  return trace
end

return trainer
