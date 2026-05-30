import logging
logger = logging.getLogger('base')
 
def create_model(opt):
    """
    Factory function for creating a model.
    Initializes the DDPM model based on the provided options.
    """
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m 