import logging
import time
import re

import torch.distributed as dist

class TrainingFormatter(logging.Formatter):
    LEVEL_COLORS = {
        'F': '\033[35m',  # Magenta (FATAL)
        'E': '\033[31m',  # Red (ERROR)
        'W': '\033[33m',  # Yellow (WARNING)
        'I': '\033[32m',  # Green (INFO)
        'D': '\033[36m',  # Cyan (DEBUG)
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    LEVEL_MAP = {
        logging.FATAL: 'F',
        logging.ERROR: 'E',
        logging.WARNING: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def format(self, record):
        level = self.LEVEL_MAP.get(record.levelno, '?')
        color = self.LEVEL_COLORS.get(level, '')
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        msg = record.msg % record.args if record.args else record.msg
        msg = re.sub(
            r'(\d+\.?\d*\s*(?:%|GB|MB|ms|TFLOPS|tok/s|params))',
            lambda m: f'{self.BOLD}{m.group(1)}{self.RESET}',
            msg
        )
        msg = re.sub(
            r'(step \d+)',
            lambda m: f'{self.BOLD}{m.group(1)}{self.RESET}',
            msg
        )
        formatted = '%s%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d]%s %s' % (
            color,
            level,
            date.tm_mon, date.tm_mday,
            date.tm_hour, date.tm_min, date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.filename,
            record.lineno,
            self.RESET,
            msg
        )

        return formatted

def setup_logging(level=logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(TrainingFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    return logging.getLogger("nmoe")


# Module-level logger
logger = setup_logging()

# Convenience exports
info = logger.info
warning = logger.warning
error = logger.error
