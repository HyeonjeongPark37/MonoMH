from lib.models.MonoMH import MonoMH

def build_model(cfg,mean_size):
    if cfg['type'] == 'MonoMH':
        return MonoMH(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
