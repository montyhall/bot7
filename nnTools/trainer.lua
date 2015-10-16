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
Modified: 2015-10-26
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local nn    = require('nn')

---------------- Constants
local defaults = 
{ 
  -------- User Interface
  UI = {verbose=1, save=0, msg_freq=math.huge},

  -------- Training Schedule
  schedule = {nEpochs=100, batchsize=32},

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
local trainer = function(model, data, config, optimizer, criterion, state)
  -------- Initialization
  local config = utils.tbl_update(config, defaults, true)
  if not config.schedule.eval_freq then
    config.schedule.eval_freq = config.schedule.nEpochs
  end
  config.UI.msg_freq = math.max(config.UI.msg_freq, config.schedule.eval_freq)

  local optimizer = optimizer or optim[config.optimizer.type]
  local criterion = criterion or nn[config.criterion.type]()
  local UI, sched = config.UI, config.schedule
  local state     = state or {} -- state for optimizer

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
  local W, grad = model:getParameters()
  local inputs  = torch.Tensor(sched.batchsize, xDim)
  local suffix  = {'e', 'r', 'v'}
  local nSets, nEvals, trace = 0, 0, {} 
  if sched.eval_freq > 0 then
    nEvals = math.floor(sched.nEpochs/sched.eval_freq)
    if nEvals > 0 then 
      if counts.r > 0 then nSets = nSets + 1 end
      if counts.e > 0 then nSets = nSets + 1 end
      if counts.v > 0 then nSets = nSets + 1 end
      trace = torch.Tensor(nEvals, nSets+1):fill(0/0) -- last position stores epoch
    end
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

      ---- Feature Corruption (Blankout noise)
      if sched.corruption and sched.corruption > 0 then
        inputs:cmul(torch.rand(inputs:size()):gt(sched.corruption):double())
      end

      if yDim > 1 then
        targets:resize(nCol,yDim):copy(data['y'..suffix]:sub(tail, head))
      else
        targets:resize(nCol):copy(data['y'..suffix]:sub(tail, head))
      end
      optimizer(closure, W, config.optimizer, state)
    end
  end

  local report = function(epoch)
    if UI.msg_freq > 0 and  epoch % UI.msg_freq == 0 then
      local t   = epoch / UI.msg_freq
      local msg = string.format('Epoch: %d of %d', epoch, sched.nEpochs)
      print('================================================')
      print(string.rep(' ', 48-msg:len()).. msg)
      print('================================================')
      local nCol = trace:size(2)
      local prev = utils.nanop(torch.max, trace:select(2, nCol))
      local idx  = trace:select(2, nCol):eq(prev):nonzero():storage()[1]
      local loss = trace:select(1, idx)
      idx, msg = 1, ''
      for k = 1, nSets do
        if counts[suffix[k]] > 0 then 
          if msg ~= '' then msg = msg .. ' | ' end
          msg = msg .. string.format('%.2e (%s)', loss[idx], suffix[k]:upper())
          idx = idx + 1
        end
      end
      print(string.format('> Loss at epoch %d: %s', prev, msg))
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

    ---- Evaluate model on provided data
    if epoch % sched.eval_freq == 0 then
      local t, idx = epoch / sched.eval_freq, 1
      trace[{t, nSets+1}] = epoch -- record epoch
      for set = 1, 3 do
        if counts[suffix[set]] > 0 then 
          trace[{t, idx}] = eval(suffix[set])
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
