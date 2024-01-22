import asyncio
import argparse
import copy
import threading
from abc import ABC
from traceback import print_exception
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
    moving_averaged_scores: torch.Tensor
    """1D (vector) with weights for each neuron on the subnet."""
    is_running: bool = False
    """Indicates whether background thread is running."""
    should_exit: bool = False
    """Indicates whether background thread should stop."""
    thread: threading.Thread | None = None

    def __init__(self, config: bt.config):
        self.config: bt.config = copy.deepcopy(config)
        check_config(config)
        bt.logging(config=config, logging_dir=config.full_path)

        self.device = torch.device(self.config.neuron.device)
        if (
            self.device.type.lower().startswith("cuda")
            and not torch.cuda.is_available()
        ):
            bt.logging.error(
                f"{self.device.type} device is selected while CUDA is not available"
            )
            exit()

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

        self.moving_averaged_scores = torch.zeros(self.metagraph.n).to(self.device)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        # # Save a copy of the hotkeys to local memory.
        # self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        #
        # # Dendrite lets us send messages to other nodes (axons) in the network.
        # self.dendrite = bt.dendrite(wallet=self.wallet)
        # bt.logging.info(f"Dendrite: {self.dendrite}")
        #
        # # Set up initial scoring weights for validation
        # bt.logging.info("Building validation weights.")
        # self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        #
        # # Init sync with the network. Updates the metagraph.
        # self.sync()
        #
        # # Serve axon to enable external connections.
        # if not self.config.neuron.axon_off:
        #     self.serve_axon()
        # else:
        #     bt.logging.warning("axon off, not serving ip to chain.")
        #
        # # Create asyncio event loop to manage async tasks.
        # self.loop = asyncio.get_event_loop()
        #
        # # Instantiate runners
        # self.thread: threading.Thread = None
        # self.lock = asyncio.Lock()

    def __enter__(self):
        self._run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_background_thread()

    def _check_for_registration(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )
            exit()

    def _run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if self.is_running:
            return
        bt.logging.debug("Starting validator in a background thread.")
        self.should_exit = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.is_running = True
        bt.logging.debug("Background thread started.")

    def _stop_background_thread(self):
        """
        Stops the validator's background operations.
        """
        if self.is_running:
            bt.logging.debug("Stopping the validator background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Background thread stopped.")

    def run(self):
        return  # mx
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

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

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

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

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(uids).to(self.device), rewards
        ).to(self.device)
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = torch.load(self.config.neuron.full_path + "/state.pt")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]

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
