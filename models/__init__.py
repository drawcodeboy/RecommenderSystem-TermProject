from .MF.mf import MF, NeuralCF

def load_model(cfg):
    if cfg['name'] == 'MF':
        return MF.from_config(cfg)
    elif cfg['name'] == 'NeuralCF':
        return NeuralCF.from_config(cfg)