import logging

logging.basicConfig(
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
        level=logging.INFO)

logging.info("Train model")
