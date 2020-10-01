import logging
import sacred

logger = logging.getLogger("simulation")

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ex = sacred.Experiment()
ex.logger = logger
ex.observers.append(sacred.observers.MongoObserver.create(connect=False))
