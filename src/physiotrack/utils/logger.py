import logging
import logging.handlers


def get_debug_logger(name, log_file_path, level=logging.DEBUG):
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(funcName)s:%(lineno)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    return _logger
