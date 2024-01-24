import time
import typing

import bittensor as bt

from neurons import protocol
from neurons.base_miner import BaseMinerNeuron
from neurons.config import read_config


class Miner(BaseMinerNeuron):
    def __init__(self, config: bt.config):
        super(Miner, self).__init__(config=config)

    async def forward(self, synapse: protocol.Dummy) -> protocol.Dummy:
        synapse.dummy_output = synapse.dummy_input * 2
        return synapse

    async def blacklist(self, synapse: protocol.Dummy) -> typing.Tuple[bool, str]:
        # TODO: check stake and other
        # if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
        #     # Ignore requests from unrecognized entities.
        #     bt.logging.trace(
        #         f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
        #     )
        #     return True, "Unrecognized hotkey"
        #
        # bt.logging.trace(
        #     f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        # )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: protocol.Dummy) -> float:
        return 1.0
        # # TODO(developer): Define how miners should prioritize requests.
        # caller_uid = self.metagraph.hotkeys.index(
        #     synapse.dendrite.hotkey
        # )  # Get the caller index.
        # prirority = float(
        #     self.metagraph.S[caller_uid]
        # )  # Return the stake as the priority.
        # bt.logging.trace(
        #     f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        # )
        # return prirority


def main():
    config = read_config(Miner)
    bt.logging.info(f"Starting with config: {config}")

    with Miner(config) as miner:
        pass
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(60)

            if miner.should_exit.is_set():
                bt.logging.debug("Stopping the validator")
                break


if __name__ == "__main__":
    main()
