import json

class Arg:
  def __init__(self, **entries):
    self.__dict__.update(entries)

class ModelConfig:
  def __init__(self, config_path):
    self.config_path = config_path
    f = open(self.config_path, 'r')
    self.config_json = json.load(f)
    self.arg = Arg(**self.config_json)

  def get_config(self):
    return self.arg

if __name__=='__main__':
  configs = ModelConfig('/ds_config.json').get_config()
  print(configs)


