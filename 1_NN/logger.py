import os
import logging
from logging import handlers
class Logger(object):
    
    def __init__(self):
        pass
    
    def set_logger(self, log_path):
        if os.path.exists(log_path) is True:
           os.remove(log_path)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
    
        if not self.logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)
    
            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)

   