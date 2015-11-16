------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Simple runtime tracker. Computes estimates time
remaining using cumulative moving averages (CMA).

Authored: 2015-11-04 (jwilson)
Modified: 2015-11-16
--]]

-------- Safeguard for isolated request
bot7 = bot7 or {}
bot7.nnTools = bot7.nnTools or {}

------------------------------------------------
--                                         timer
------------------------------------------------
local title   = 'bot7.nnTools.timer'
local timer = torch.class(title)

function timer:__init(typ, measure)
  self.type    = typ
  self.measure = measure or 'real' -- type of time to measure
  self.memory = {} -- memory for external values
  self.state  = {} -- internal memory
  self.timer  = torch.Timer()
end

function timer:__call__(times, t, req)
  
  if times then self:update(times) end
  if t then return self:predict(t, req) end
end

function timer:tic(req, measure)
  local measure = measure or self.measure
  local tic = self.timer:time()[measure]

  local mem = self.memory
  if type('req') == 'table' then
    for _, key in pairs(req) do
      mem[req] = mem[req] or {}
      mem[req].tic = tic
    end
  elseif req then
    mem[req] = mem[req] or {}
    mem[req].tic = tic
  else
    self.state.tic = tic
  end
  return tic
end

function timer:toc(req, measure)
  local measure = measure or self.measure
  local toc = self.timer:time()[measure]

  local mem = self.memory
  if type('req') == 'table' then
    for _, key in pairs(req) do
      mem[req] = mem[req] or {}
      self.memory[req].toc = toc
    end
  elseif req then
    mem[req] = mem[req] or {}
    mem[req].toc = toc
  else
    self.state.toc = toc
  end
  return toc
end

function timer:start(req, measure)
  return self:tic(req, measure)
end

function timer:stop(req, measure)
  self:toc(req, measure)

  local elapsed
  if type(req) == 'table' then
    local mem = self.memory
    self.type = self.type or 'table' 
    elapsed = {}
    for _, key in pairs(req) do
      elapsed[key] = mem[key].toc-mem[key].tic
    end
    self:update(elapsed)
  elseif req then
    self.type = self.type or 'table'
    elapsed = self.memory[req].toc - self.memory[req].tic
    self:update(elapsed, self.memory[req])
  else
    elapsed = self.state.toc - self.state.tic
  end

  return elapsed
end


function timer:update(times, memory)
  local times, typ = times, type(times)
  local memory = memory or self.memory
  self.type = self.type or typ

  -------- Branch according to input type
  if typ == 'number' then
    local count = memory.count or 0
    local cma   = memory.cma or 0
    memory.cma   = (times + count*cma)/(count+1)
    memory.count = count + 1

  elseif typ == 'table' then
    for key, field in pairs(times) do
      memory[key] = memory[key] or {}
      self:update(field, memory[key])
    end

  else
    print('Error: timer.update() encountered an '..
          'unrecognized input type; returning...')
  end
end

function timer:predict(t, req)
  local t, req = t, req
  local memory = self.memory
  local types  = {t=type(t), req=type(req)}

  if torch.isTensor(t) then
    types.t = 'tensor'
    if self.type ~= 'table' then
      t = torch.Tensor(t)
    end
  end

  if self.type == 'number' then
    if types.t == 'number' then
      return t*memory.cma
    elseif types.t == 'tensor' then
      return torch.mul(t, memory.cma)
    end

  elseif self.type == 'table' then
    local res, mem, len, keys = {}, {}, {}

    if type(req) == 'table' then
      keys = req
    elseif types.req == 'string' then
      keys = {req}
    else
      local idx = 1
      keys = {}

      for key, field in pairs(self.memory) do
        keys[idx] = key
        idx = idx + 1
      end
    end

    idx = 1
    for idx, key in pairs(keys) do
      if types.t == 'number' then
        len[key] = t
      elseif types.t == 'table' then
        len[key] = t[key]
      elseif types.t == 'tensor' then
        len[key] = tensor[idx]
        idx = idx + 1
      end
    end

    for idx, key in pairs(keys) do
      res[key] = memory[key].cma * len[key]
    end

    if types.req == 'string' and types.t == 'number' then
      return res[req]
    else
      return res
    end

  end
  print('Error: timer.predict() encountered an '..
        'unrecognized input type; returning...')
end

function timer:reset()
  self.memory = {}
  self.state  = {}
  self.timer:reset()
end


return bot7.nnTools.timer