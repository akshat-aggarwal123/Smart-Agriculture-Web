import logging
import sys

def setup_logger():
    """Configure application-wide logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler and set level to info
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add formatter to ch
    ch.setFormatter(formatter)
    
    # Add ch to logger
    logger.addHandler(ch)
    
    return logger

# Initialize logger
logger = setup_logger()