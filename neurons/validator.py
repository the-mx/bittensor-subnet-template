import time

import bittensor as bt

from neurons.base_validator import BaseValidatorNeuron
from config import read_config
from protocol import Dummy


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
        miners = self.get_miners()
        # miner_uids = get_random_uids(self, k=self.read_config.neuron.sample_size)

        responses = await self.dendrite.forward(
            axons=miners,
            synapse=Dummy(dummy_input=self.step),
            deserialize=True,
        )

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        # TODO(developer): Define how the validator scores responses.
        # Adjust the scores based on responses from miners.
        # rewards = get_rewards(self, query=self.step, responses=responses)

        # bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        # self.update_scores(rewards, miner_uids)


def main():
    config = read_config(Validator)
    bt.logging.info(f"Starting with config: {config}")

    with Validator(config) as validator:
        while True:
            bt.logging.debug("Validator running...", time.time())
            time.sleep(60)

            if validator.should_exit.is_set():
                bt.logging.debug("Stopping the validator")
                break


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    main()
