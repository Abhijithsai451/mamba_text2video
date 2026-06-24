import logging
import os
import wandb

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
_project_name = "GAN_Scientific_Computing"
def setup_logger(config):
    """
    Sets up logger for console output and Training Output
    """
    global _project_name
    _project_name = config.project_name
    logger = logging.getLogger(_project_name)

    if logger.hasHandlers():
        return logger

    log_dir = config.logger.get("log_dir", "logs")
    log_file = config.logger.get("log_file", "training.log")
    log_path = os.path.join(log_dir, log_file)

    os.makedirs(log_dir, exist_ok=True)

    # Use the configuration you provided
    logging.basicConfig(
        level=getattr(logging, config.logger.get('level', 'INFO')),
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    if wandb.run is not None:
        logger.info("W&B Logic Integrated with system logger")
    return logger

def get_logger():
    """
    Returns the logger instance
    """
    return logging.getLogger(_project_name)