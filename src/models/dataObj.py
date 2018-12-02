import datetime
import time
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

logging.info(datetime.datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"))
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

test = "sometstring"

def my_log(msg):
    ''' Return a string representation of the time '''
    logging.info(datetime.datetime.now().strftime("%d-%m-%Y %I:%M:%S %p: "),
            msg)
while True:
    my_log(
    time.sleep(1)
