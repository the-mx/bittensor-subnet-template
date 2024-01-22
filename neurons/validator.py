import time

import bittensor as bt

from neurons.base_validator import BaseValidatorNeuron
from template.validator import forward
from config import read_config


class Validator(BaseValidatorNeuron):
    def __init__(self, config: bt.config):
        super(Validator, self).__init__(config)

        # self.load_state()

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)


def main():
    config = read_config(Validator)
    bt.logging.info(f"Starting with config: {config}")

    with Validator(config):
        while True:
            bt.logging.debug("Validator running...", time.time())
            time.sleep(60)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    main()
