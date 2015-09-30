------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for bot7 models.

Authored: 2015-09-15 (jwilson)
Modified: 2015-09-30
--]]

------------------------------------------------
--                                     metamodel
------------------------------------------------
local title = 'bot7.models.metamodel'
local model = torch.class(title)

function model:__init()
end

function model:save()
end

function model:load()
end

function model:update()
end

function model:cache()
  local cache = 
  {
    config  = self.config,
    kernel  = self.kernel,
    nzModel = self.nzModel,
    mean    = self.mean,
    hyp     = self.hyp
  }
  return cache
end

function model:class()
  return self:__tostring__()
end

function model:__tostring__()
  return torch.type(self)
end
