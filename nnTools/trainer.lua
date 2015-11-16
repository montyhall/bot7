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

Authored: 2015-11-04 (jwilson)
Modified: 2015-11-17
--]]

---------------- External Dependencies
local utils = require('bot7.utils')
local optim = require('optim')
local math  = require('math')
local nn    = require('nn')
local evaluator = require('bot7.nnTools.evaluator')
local Buffers = require('bot7.nnTools.buffers')
local timer = require('bot7.nnTools.timer')()

---------------- Constants
local defaults = 
{ 
  -------- Global terms
  batchsize = 32,

  -------- User Interface
  ui = {verbose=1, save=0, msg_freq=math.huge},

  -------- Training Schedule
  schedule = {nEpochs=100},

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

  buffers = {fullsize=5120}
}

-- Set inheritence for sub-tables
for key, field in pairs(defaults) do
  if type(field) == 'table' then
    setmetatable(field,{__index=defaults})
  end
end

------------------------------------------------
--                                       trainer
------------------------------------------------
local trainer = function(network, data, config, cache)
  -------- Initialization
  local network, data = network, data
  local config = utils.table.deepcopy(config or {})
  utils.table.update(config, defaults, true)

  local UI, sched = config.ui, config.schedule
  local cache     = cache or {}
  local optimizer = cache.optimizer or optim[config.optimizer.type]
  local criterion = cache.criterion or nn[config.criterion.type]()
  local state     = cache.state or {} -- state for optimizer
  local buffers   = cache.buffers or Buffers(config.buffers, data)
  local flags     = cache.flags or {classif=false}
  local surrogate = cache.surrogate

  -------- Enforce GPU settings
  if sched.gpu then
    require('cunn'); require('cutorch');
    network:cuda(); criterion:cuda(); buffers:cuda() -- Convert to GPU
    if surrogate then surrogate:cuda() end
    buffers.config.type = 'CudaTensor'
  end

  -------- Shape Information
  local suffix, counts = {'e', 'r', 'v'}, {}
  for idx, key in pairs(suffix) do
    if data['x'..key] and data['y'..key] then
      counts[key] = data['x'..key]:size(1)
    end
  end

  -------- Safeguard against bad settings
  if not sched.eval_freq then sched.eval_freq = sched.nEpochs end
  buffers.config.batchsize = math.min(sched.batchsize, buffers.config.fullsize)

  -------- Detect Problem Type (Classif. / Regression)
  local ttype = data.yr:type()
  if ttype == 'torch.LongTensor' 
  or ttype == 'torch.ByteTensor'
  or data.yr:eq(torch.floor(data.yr)):all() then
    flags.classif = true
  end

  -------- Local Aliases / Result tensor reallocation
  local W, grad = network:getParameters()
  local nSets, nEvals, trace = 0, 0, {} 
  if sched.eval_freq > 0 then
    nEvals = math.floor(sched.nEpochs/sched.eval_freq)
    if nEvals > 0 then 
      for key, count in pairs(counts) do
        if count > 0 then nSets = nSets + 1 end
      end
      ---- last column in trace stores epoch
      trace = {loss = torch.zeros(nEvals, nSets+1)}
      trace.loss:narrow(2, 1, nSets):fill(0/0)
      if flags.classif then
        trace.err = torch.zeros(nEvals, nSets+1)
        trace.err:narrow(2, 1, nSets):fill(0/0)
      end
    end
  end

  -------- Functional Closure (for optimizer)
  local closure = function(x)
    if x ~= W then W:copy(x) end
    grad:zero() -- reset grads

    local inputs  = buffers('x')
    local targets = buffers('y')
    if buffers:has_key('ys') then        -- 'ys' denotes a surrogate
      targets = {targets, buffers('ys')} -- model's predictions, e.g.,
    end                                  -- Dark Knowledge

    local outputs = network:forward(inputs)
    local fval    = criterion:forward(outputs, targets)
    local df_dw   = criterion:backward(outputs, targets)

    network:backward(inputs, df_dw)
    fval = fval/outputs:size(1)
    return fval, grad
  end

  -------- Parameter updates (single epoch)
  local update = function(suffix)
    timer:start('update')
    local suffix = suffix or 'r'
    local perm = torch.randperm(counts[suffix], 'torch.LongTensor')
    local tail, size  = 0, 0
    local buffer_size = buffers.config.fullsize
    local batchsize   = buffers.config.batchsize
    local nBatches    = 0

    ---- Establish which buffers need to be updated/incremented
    local keys = {'x'..suffix, 'y'..suffix}
    if surrogate and buffers:has_key('xs') then
      table.insert(keys, 'xs') -- input buffer for surrogate
    elseif buffers:has_key('ys') then
      table.insert(keys, 'ys') -- cached outputs from surrogate
    end

    ---- Set modules to 'training' mode
    if network.training   then network:training()   end
    if criterion.training then criterion:training() end

    for head = 1, counts[suffix], buffer_size do -- iterate over databatches
      tail     = math.min(head+buffer_size-1, counts[suffix])
      nBatches = math.ceil((tail-head+1)/batchsize)
      idx      = perm:sub(head, tail)
      buffers:update(data, idx, keys) -- load next databatch

      for batch = 1, nBatches do -- iterate over minibatches
        ---- Compute surrogate model's prediction
        if surrogate then
          if buffers:has_key('xs') then
            buffers:set('ys', surrogate:forward(buffers('xs')))
          else
            buffers:set('ys', surrogate:forward(buffers('x')))
          end
        end
        optimizer(closure, W, config.optimizer, state)
        buffers:increment(keys) -- prepare for next minibatch
      end
    end
    timer:stop('update')
  end

  -------- Wrapper for evaluator
  local eval = function(suffix)
    timer:start('eval')
    if criterion.evaluate then criterion:evaluate() end
    local temp = buffers.config.batchsize
    local loss, err = evaluator(network, criterion, data['x'..suffix],
                        data['y'..suffix], config.eval, {buffers=buffers})

    -- Reset batchsize (eval might use different batchsize)
    buffers.config.batchsize = temp
    buffers:index_reset() -- reset all indices to 1 (as a precaution)
    timer:stop('eval')
    return loss, err
  end

  -------- Message Printing UI
  local report = function(epoch)
    if UI.msg_freq > 0 and  epoch % UI.msg_freq == 0 then
      local t   = epoch / UI.msg_freq
      local msg = string.format('Epoch: %d of %d', epoch, sched.nEpochs)
      utils.ui.printSection(msg)

      ---- Runtime tracking 
      local eta = timer:predict(sched.nEpochs - epoch, 'epoch')
      msg = string.format('> Runtime: %.2es (ETA) | %.2es per epoch',
                          eta, timer.memory.epoch.cma)
      print(msg)

      ---- Learning rate
      local nEvals = state.evalCounter or 0
      local lRate  = config.optimizer.learningRate/(1 + nEvals*config.optimizer.learningRateDecay)
      print(string.format('> Learning Rate: %.2E (after %d updates)',lRate, nEvals))

      ---- Performance measures
      local env  = {Loss= trace.loss, Error=trace.err}
      local nCol = env.Loss:size(2)
      local prev, idx = env.Loss:select(2, nCol):max(1)
      prev, idx = prev[1], idx[1]
      if prev > 0 then
        env.Loss = env.Loss:select(1, idx)
        if env.Error then env.Error = env.Error:select(1, idx) end
      else
        env.Loss, env.Error = nil, nil
      end

      for key, field in pairs(env) do
        idx, msg = 1, ''
        for k = 1, nCol-1 do
          if counts[suffix[k]] then 
            if msg ~= '' then msg = msg .. ' | ' end
            msg = msg .. string.format('%.2e (%s)', env[key][idx], suffix[k]:upper())
            idx = idx + 1
          end
        end
        print(string.format('> %s at epoch %d: %s', key, prev, msg))
      end
    end
  end

  -------- Main Training Loop
  for epoch = 1, sched.nEpochs do
    timer:start('epoch')
    ---- Perform parameter update
    update('r')

    ---- Evaluate network on provided data
    if epoch % sched.eval_freq == 0 then
      local t, idx = epoch / sched.eval_freq, 1
      trace.loss[{t, nSets+1}] = epoch -- record epoch
      if flags.classif then
        trace.err[{t, nSets+1}] = epoch -- record epoch
      end
      for set = 1, 3 do
        if counts[suffix[set]] then
          local loss, err = eval(suffix[set])
          trace.loss[{t, idx}] = loss
          if flags.classif and err then
            trace.err[{t, idx}] = err
          end
          idx = idx + 1
        end
      end
    end

    ---- Report current status to user
    timer:stop('epoch')
    report(epoch)
  end
  return trace
end

return trainer
