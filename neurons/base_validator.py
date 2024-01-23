import abc
import asyncio
import argparse
import copy
import random
import threading
import time
from abc import ABC
from typing import List

import bittensor as bt
import torch

from config import check_config


class BaseValidatorNeuron(ABC):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    config: bt.config
    """Copy of the original config."""
    uid: int
    """Each miner gets a unique identity (UID) in the network for differentiation."""
    device: torch.device
    """Device to run computations on."""

    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the miner."""
    subtensor: bt.subtensor
    """The subtensor is our connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    axon: bt.axon | None = None
    """Axon for external connections."""
    dendrite: bt.dendrite | None = None

    scores: torch.Tensor
    """1D (vector) with scores for each neuron on the subnet."""
    hotkeys: List[str]
    """A copy of metagraph.hotkeys."""

    is_background_thread_running: bool = False
    """Indicates whether background thread is running."""
    should_stop_background_thread: bool = False
    """When set background thread is supposed to stop."""
    should_exit: threading.Event
    """Set in the background thread on errors. Should lead to the application stop."""
    thread: threading.Thread | None = None
    """Background thread."""

    step: int = 0
    """Increased each time the validator sends probes to the miners."""

    loop: asyncio.AbstractEventLoop
    """Async loop."""

    _cached_block: int = 0
    _cached_block_time: float = 0.0
    _cached_block_ttl: float = 12.0
    """Not to request the current block on each validator tick, it's value is cached for the next `ttl` seconds."""

    def __init__(self, config: bt.config):
        self.config: bt.config = copy.deepcopy(config)
        check_config(config)
        bt.logging(config=config, logging_dir=config.full_path)

        self.device = torch.device(self.config.neuron.device)
        if (
            self.device.type.lower().startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                f"{self.device.type} device is selected while CUDA is not available"
            )

        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._check_for_registration()

        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor)  # Sync metagraph with subtensor.
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        # # Set up initial scores.
        self.scores = torch.zeros(self.metagraph.n, device=self.device)

        # # Serve axon to enable external connections.
        # if not self.config.neuron.axon_off:
        #     self.serve_axon()
        # else:
        #     bt.logging.warning("axon off, not serving ip to chain.")

        self.should_exit = threading.Event()

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

    def __enter__(self):
        self._run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_background_thread()

    @abc.abstractmethod
    def forward(self):
        """Named after neural network forward propagation. This method sends a probe/test to the miners
        to score them and assign weights."""
        ...

    def block(self):
        cur_time = time.time()
        if cur_time > self._cached_block_time + self._cached_block_ttl:
            self._cached_block = self.subtensor.get_current_block()
            self._cached_block_time = cur_time

        return self._cached_block

    def get_random_miners_uids(self, sample_size=None) -> List[int]:
        miners = [uid for uid, stake in enumerate(self.metagraph.S) if stake < 1.0]
        random.shuffle(miners)
        return miners if sample_size is None else miners[:sample_size]

    def _check_for_registration(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    def _run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if self.is_background_thread_running:
            return
        bt.logging.debug("Starting validator in a background thread.")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.should_stop_background_thread = False
        self.is_background_thread_running = True
        bt.logging.debug("Background thread started.")

    def _stop_background_thread(self):
        """
        Stops the validator's background operations.
        """
        if not self.is_background_thread_running:
            return
        bt.logging.debug("Stopping the validator background thread.")
        self.should_stop_background_thread = True
        self.thread.join(30)
        self.is_background_thread_running = False
        if self.axon is not None:
            self.axon.stop()
        bt.logging.debug("Background thread stopped.")

    def _run(self):
        """
        Manages the Bittensor miner's loop, ensuring graceful shutdowns on interrupts and logging unexpected errors.
        Key tasks performed by this function:
        1. Checking the miner's registration status on the Bittensor network.
        2. Persistently forwarding queries to network miners, rewarding and scoring their responses.
        3. Regularly syncing with the chain to update the metagraph with the current network state and adjust weights.

        The 'forward' function, called at each loop iteration, is central to validation operations,
        as it queries the network and appraises the responses.

        Notes:
        * Uses global configurations defined during miner setup.
        * The miner's axon is the interface for Bittensor network communications.

        Raises:
            KeyboardInterrupt: Upon a manual interruption.
            Exception: For unexpected issues during operation, with logs for diagnostics.
        """
        bt.logging.info(f"Validator starting at block: {self.block()}")

        try:
            while True:
                bt.logging.debug(f"Step {self.step} at block {self.block()}")

                self._sync()  # Check that the validator is registered on the network.

                if self.should_stop_background_thread:
                    break

                time.sleep(10)

                self.loop.run_until_complete(self._concurrent_forward())

                self.step += 1
        except KeyboardInterrupt:
            self.should_exit.set()
            bt.logging.success("Validator killed by keyboard interrupt.")
        except Exception as e:
            self.should_exit.set()
            bt.logging.exception(f"Error during validation: {e}")

    async def _concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def _sync(self):
        # Ensure miner or validator hotkey is still registered on the network.
        self._check_for_registration()

        if self._should_sync_metagraph():
            self._resync_metagraph()

        if self._should_set_weights():
            self.set_weights()

        # Always save state.
        self.save_state()

    def _should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block() - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def _resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""

        bt.logging.info("Resyncing metagraph")

        previous_hotkeys = list(self.metagraph.hotkeys)
        self.metagraph.sync(subtensor=self.subtensor)
        if previous_hotkeys == self.metagraph.hotkeys:
            return

        bt.logging.info("Metagraph updated, re-syncing hotkeys and moving averages")

        prev_hotkey_to_score = {
            hotkey: self.scores[uid] for uid, hotkey in enumerate(previous_hotkeys)
        }
        bt.logging.debug(f"Prev scores {prev_hotkey_to_score}")

        new_scores = torch.zeros(self.metagraph.n, device=self.device)

        # Populate score values based on the previous hotkeys.
        for uid, axon in enumerate(self.metagraph.axons):
            new_scores[uid] = prev_hotkey_to_score.get(axon.hotkey, 0.0)

        # Replace old score tensor with the new tensor.
        self.scores = new_scores

        bt.logging.debug(f"New scores {self.scores}")

    def _should_set_weights(self) -> bool:
        return False

        # # Don't set weights on initialization.
        # if self.step == 0:
        #     return False
        #
        # # Check if enough epoch blocks have elapsed since the last epoch.
        # if self.config.neuron.disable_set_weights:
        #     return False
        #
        # # Define appropriate logic for when set weights.
        # return (
        #         self.block() - self.metagraph.last_update[self.uid]
        # ) > self.config.neuron.epoch_length

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(
            self.moving_averaged_scores, p=1, dim=0
        )

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", self.metagraph.uids.to("cpu"))

        raise RuntimeError("time to rest")

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=True,
            version_key=spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed")

    def update_scores(self, scores: torch.FloatTensor, miner_uids: List[int]):
        """Updates moving average scores based on the recent received scores"""

        # Check if rewards contains NaN values.
        if torch.isnan(scores).any():
            bt.logging.warning(f"NaN values detected in rewards: {scores}")
            # Replace any NaN values in rewards with 0.
            scores = torch.nan_to_num(scores, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_scores: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(miner_uids, device=self.device), scores
        ).to(self.device)
        bt.logging.debug(f"Scattered rewards: {scattered_scores}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_scores + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        # torch.save(
        #     {
        #         "step": self.step,
        #         "scores": self.scores,
        #         "hotkeys": self.hotkeys,
        #     },
        #     self.config.neuron.full_path + "/state.pt",
        # )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        # state = torch.load(self.config.neuron.full_path + "/state.pt")
        # self.step = state["step"]
        # self.scores = state["scores"]
        # self.hotkeys = state["hotkeys"]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """
        Add Validator specific arguments.
        """
        # Netuid Arg: The netuid of the subnet to connect to.
        parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

        parser.add_argument(
            "--neuron.device",
            type=str,
            help="Device to run on (cpu/cuda:%d).",
            default="cpu",
        )

        parser.add_argument(
            "--neuron.epoch_length",
            type=int,
            help="The default epoch length (how often we set weights, measured in 12 second blocks).",
            default=100,
        )

        parser.add_argument(
            "--neuron.num_concurrent_forwards",
            type=int,
            help="The number of concurrent forwards running at any time.",
            default=1,
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=10,
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            help="Moving average alpha parameter, how much to add of the new observation.",
            default=0.05,
        )

        parser.add_argument(
            "--neuron.vpermit_tao_limit",
            type=int,
            help="The maximum number of TAO allowed to query a validator with a vpermit.",
            default=4096,
        )
