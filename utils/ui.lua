------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
User Interface (UI) utility method for bot7.

Authored: 2015-10-30 (jwilson)
Modified: 2015-11-16
--]]

------------------------------------------------
--                                            ui
------------------------------------------------
local self = {}

--------------------------------
--          Print section header
--------------------------------
function self.printSection(str, config)
  if not os then require 'os' end
  local config  = config or {}
  local str     = str   or ''
  local width   = config.width or 48
  local tstamp  = config.timestamp or config.time
  local rdelim  = config.ldelim or config.delim or ''
  local ldelim  = config.rdelim or config.delim or ''
  local ljust   = config.ljust or false
  local newline = config.newline or config.newline == nil
  local bar     = config.bar or '='

  local msg  = ldelim
  local len  = ldelim:len() + rdelim:len() + str:len()
  local time = nil
  if tstamp == nil then tstamp = 'auto' end
  if tstamp then
    time = os.date("%d/%m - %H:%M%p ")
    local tlen = time:len()
    if tstamp == true or (tstamp == 'auto' and len+tlen < width) then
      len  = len + tlen
    else
      time = nil
    end
  end

  local reps = width/bar:len()
  bar = string.rep(bar, reps)

  if ljust then
    msg = '\n' .. msg .. str .. string.rep(' ', width - len)
    if time then msg = msg .. time end
    msg = msg .. rdelim
  else
    if time then msg = msg .. time end
    msg = '\n' .. msg .. string.rep(' ', width - len) .. str .. rdelim
  end
  
  if newline then
    msg = '\n' .. bar .. msg .. '\n' .. bar
  else
    msg = bar .. msg .. '\n' .. bar
  end

  print(msg)
end

--------------------------------
--   Print command line argument
--------------------------------
function self.printArgs()
  local nArgs = #arg
  if nArgs == 0 then return end
  self.printSection('Command Line Arguments')
  local count,k = 0, 0
  while k < nArgs do
    count, k = count+1, k+1
    if arg[k] ~= '-verbose' then
      if k < #arg and arg[k+1]:sub(1,1) ~= '-' then
        print(string.format('[%d]  %s \t %s',count, arg[k],arg[k+1]))
        k = k+1
      else
        print(string.format('[%d]  %s',count, arg[k]))
      end
    end
  end
  print(' ')
end

return self