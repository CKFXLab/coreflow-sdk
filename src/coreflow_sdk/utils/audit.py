import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

class AppLogger:
    """
    Single class logger with support for colored ASCII output and JSON output.
    """
    
    # ANSI color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset to default
    }
    
    def __init__(self, module_name: str, level: str = "INFO", output_format: str = "ascii"):
        """
        Initialize the AppLogger.
        
        Args:
            module_name: Name of the module (usually __name__)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            output_format: Either "ascii" for colored text or "json" for JSON output
        """
        self.module_name = module_name
        self.level = level.upper()
        self.output_format = output_format.lower()
        
        # Create the underlying logger
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(getattr(logging, self.level))
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Set up the appropriate formatter and handler
        self._setup_logging()
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _setup_logging(self):
        """Set up the logging handler and formatter based on output format."""
        handler = logging.StreamHandler(sys.stdout)
        
        if self.output_format == "json":
            handler.setFormatter(self._create_json_formatter())
        else:  # ascii (default)
            handler.setFormatter(self._create_ascii_formatter())
        
        self.logger.addHandler(handler)
    
    def _create_ascii_formatter(self):
        """Create a colored ASCII formatter."""
        class ColoredFormatter(logging.Formatter):
            def __init__(self, colors):
                super().__init__()
                self.colors = colors
            
            def format(self, record):
                # Get color for the log level
                color = self.colors.get(record.levelname, self.colors['RESET'])
                reset = self.colors['RESET']
                
                # Format the message with colors
                timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
                formatted_message = (
                    f"{color}[{timestamp}] {record.levelname:<8} "
                    f"{record.name} - {record.getMessage()}{reset}"
                )
                
                # Add exception info if present
                if record.exc_info:
                    formatted_message += f"\n{color}{self.formatException(record.exc_info)}{reset}"
                
                return formatted_message
        
        return ColoredFormatter(self.COLORS)
    
    def _create_json_formatter(self):
        """Create a JSON formatter."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "thread": record.thread,
                    "thread_name": record.threadName,
                    "process": record.process,
                }
                
                # Add exception info if present - check if it's a tuple, not just truthy
                if record.exc_info and isinstance(record.exc_info, tuple):
                    log_data["exception"] = self.formatException(record.exc_info)
                
                # Add stack info if present
                if record.stack_info:
                    log_data["stack_info"] = record.stack_info
                
                # Add any extra fields
                extra_data = {}
                standard_attrs = {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                    'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage'
                }
                
                for key, value in record.__dict__.items():
                    if key not in standard_attrs and not key.startswith('_'):
                        try:
                            json.dumps(value)  # Test if JSON serializable
                            extra_data[key] = value
                        except (TypeError, ValueError):
                            extra_data[key] = str(value)
                
                if extra_data:
                    log_data["extra"] = extra_data
                
                return json.dumps(log_data, ensure_ascii=False)
        
        return JSONFormatter()
    
    def debug(self, message: str, **extra):
        """Log a debug message."""
        self._log('debug', message, **extra)
    
    def info(self, message: str, **extra):
        """Log an info message."""
        self._log('info', message, **extra)
    
    def warning(self, message: str, **extra):
        """Log a warning message."""
        self._log('warning', message, **extra)
    
    def error(self, message: str, **extra):
        """Log an error message."""
        self._log('error', message, **extra)
    
    def critical(self, message: str, **extra):
        """Log a critical message."""
        self._log('critical', message, **extra)
    
    def exception(self, message: str, **extra):
        """Log an exception with traceback."""
        self._log('error', message, exc_info=True, **extra)
    
    def _log(self, level: str, message: str, exc_info: bool = False, **extra):
        """Internal logging method."""
        log_method = getattr(self.logger, level)
        
        if extra:
            # Get the actual exception info if exc_info is True
            actual_exc_info = None
            if exc_info:
                actual_exc_info = sys.exc_info()
                # Only use it if there's actually an exception
                if actual_exc_info[0] is None:
                    actual_exc_info = None
            
            # Create a custom LogRecord with extra fields
            record = self.logger.makeRecord(
                name=self.logger.name,
                level=getattr(logging, level.upper()),
                fn='',
                lno=0,
                msg=message,
                args=(),
                exc_info=actual_exc_info  # Pass the actual tuple, not boolean
            )
            
            # Add extra fields to the record
            for key, value in extra.items():
                setattr(record, key, value)
            
            self.logger.handle(record)
        else:
            if exc_info:
                log_method(message, exc_info=True)
            else:
                log_method(message)





# Example usage and testing
if __name__ == "__main__":
    print("=== Testing ASCII colored output ===")
    ascii_logger = AppLogger("test.module", level="DEBUG", output_format="ascii")
    
    ascii_logger.debug("This is a debug message")
    ascii_logger.info("This is an info message")
    ascii_logger.warning("This is a warning message")
    ascii_logger.error("This is an error message")
    ascii_logger.critical("This is a critical message")
    
    # Test with extra data
    ascii_logger.info("User logged in", user_id="12345", action="login")
    
    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except Exception:
        ascii_logger.exception("An exception occurred")
    
    print("\n=== Testing JSON output ===")
    json_logger = AppLogger("test.module", level="DEBUG", output_format="json")
    
    json_logger.debug("This is a debug message")
    json_logger.info("This is an info message")
    json_logger.warning("This is a warning message")
    json_logger.error("This is an error message")
    json_logger.critical("This is a critical message")
    
    # Test with extra structured data
    json_logger.info("User action", 
                    user_id="12345", 
                    action="purchase", 
                    amount=99.99,
                    metadata={"product": "widget", "category": "tools"})
    
    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except Exception:
        json_logger.exception("An exception occurred with context", 
                             error_code="E001", 
                             user_id="12345")