import logging

# Configure the logger
logging.basicConfig(
    filename='app.log',  # Log file
    filemode='a',        # Append mode, so new logs are appended to the file
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.DEBUG  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

# Creating a custom logger
logger = logging.getLogger(__name__)

def get_logger():
    return logger
