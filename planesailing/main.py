"""Precursor module with aircraft processing logic, that will eventually be moved into a standalone package"""

from cachetools import TTLCache
from collections.abc import Iterable
from dataclasses import dataclass
import time
from datetime import datetime
from threading import Thread

import numpy as np
import pandas as pd
import structlog
import rs1090

import cosmosdr.signal_acquisition as s_acq

from planesailing.adsb_processing import (
    preprocess_iq,
    convert_to_binary,
    decode_to_adsb,
    convert_adsb_binary_to_hex,
)

logger = structlog.get_logger()

ADSB_FREQUENCY = 1090e6


@dataclass
class ADSBPreParsedHit:
    timestamp: float
    hex_string: str
    index: int


class ADSBRecentMessagesCache:
    def __init__(self, max_length=np.inf):
        self.max_length = max_length
        self.cache = TTLCache(maxsize=max_length, ttl=300)  # 5 minute TTL

    def add_message(self, message: ADSBPreParsedHit):
        self.cache[message.hex_string] = message

    def add_messages(self, messages: Iterable[ADSBPreParsedHit]):
        for message in messages:
            self.add_message(message)

    def get_messages(self):
        return self.cache

    def decode_current_cached_messages(self, verbose=False):
        """Decode all currently cached messages, returning a DataFrame of decoded messages."""
        cached_messages = self.get_messages()
        if verbose:
            logger.info("Decoding %s cached messages", len(cached_messages))

        timestamps = [msg.timestamp for msg in cached_messages.values()]
        hexes = [msg.hex_string for msg in cached_messages.values()]

        parsed_msgs = rs1090.decode(msg=hexes, timestamp=timestamps)

        parsed_msgs_df = pd.DataFrame(parsed_msgs)

        if verbose:
            logger.info("Parsed ADSB messages: %s", len(parsed_msgs_df))
            if "latitude" in parsed_msgs_df.columns:
                parsed_msgs_with_position = parsed_msgs_df[
                    parsed_msgs_df["latitude"].notnull()
                ]
                logger.info(
                    "Messages with position: %s", parsed_msgs_with_position.shape[0]
                )
        return parsed_msgs_df


class ADSBStreamer:
    """
    A Tool for continuously streaming ADSB signals from an SDR device.:
    """

    def can_start_acquisition(self):
        return self.thread is None

    def __init__(self):
        self.enabled: bool = False
        # If thread is none, then no acquisition loop is running, and we can start one
        self.thread: Thread | None = None

    def start_adsb_stream(
        self,
        sample_rate: float = 2.4e6,
        target_sr: float = 12e6,
        n_reads_per_acquisition: int = 64,
        n_reads_to_try_to_parse: int = 64,
        n_samples_per_read: int = 4096,
        sleep_length_s: float = 0.01,
        sdr_gain: str = "auto",
        n=1,  # retained for later experimentation, not useful atm
        verbose=False,
    ):
        """
        Tell the ADSBStreamer to begin streaming with a given set of parameters.

        Creates a thread to run the acquisition loop in the background.
        """
        if not self.can_start_acquisition():
            logger.warning("ADSBStreamer: Acquisition loop already running")

        self.enabled = True

        # Start acquisition loop in background thread
        thread = Thread(
            target=self.stream_and_decode_adsb,
            args=(
                sample_rate,
                target_sr,
                n_reads_per_acquisition,
                n_reads_to_try_to_parse,
                n_samples_per_read,
                sleep_length_s,
                sdr_gain,
                n,
                verbose,
            ),
            daemon=True,  # stops automatically if main thread exits
        )
        thread.start()
        logger.info(
            "ADSBStreamer: ADSB Streamer",
            sample_rate=sample_rate,
            target_sr=target_sr,
            n_reads_per_acquisition=n_reads_per_acquisition,
            n_samples_per_read=n_samples_per_read,
            sleep_length_s=sleep_length_s,
            sdr_gain=sdr_gain,
            n=n,
            verbose=verbose,
        )
        self.thread = thread

    def stop_stream(self):
        """Tell the SignalStreamer to stop streaming, if there is an acquisition loop running."""
        if self.thread is None:
            logger.warning("ADSBStreamer: Acquisition loop not running")
            return

        # Signal the acquisition loop to stop
        self.enabled = False

        # Wait for clean shutdown
        self.thread.join()
        del self.thread
        self.thread = None
        logger.info("ADSBStreamer: Acquisition loop stopped")

    def stream_and_decode_adsb(
        self,
        sample_rate: float = 2.4e6,
        target_sr: float = 12e6,
        n_reads_per_acquisition: int = 64,
        n_reads_to_try_to_parse: int = 64,
        n_samples_per_read: int = 4096,
        sleep_length_s: float = 0.01,
        sdr_gain: str = "auto",
        n=1,  # retained for later experimentation, not useful atm
        verbose=False,
    ):
        """Start the signal streamer, then repeatedly grab the current signal, process it, and try to decode any ADSB messages.

        Default parameters for n, reads per acquisition, samples per read, n_reads_to_try_to_parse and sleep time are chosen to maximise the number of
        parsed hexes retrieved in a given block of time

        - Set up the signal stream
        - While true, repeatedly run the inner loop: _batch_decode_adsb()
            - N times, grab a batch of reads
                - Select the read with the highest individual peak (likely to be an ADSB pulse)
                - Decode the ADSB pulse if possible
            - Now we have all the ADSB pulses, feed them into the decoder again to derive positional info
            - Write them to the RecentMessagesCache, where they can be later processed and written to the DB

        n: number of reads to attempt, no longer used, retained for later experimentation
        n_reads_per_acquisition: number of reads to acquire in each acquisition loop iteration
        n_samples_per_read: number of samples per read in each acquisition loop iteration
        sleep_length_s: time to wait between reads, purely for managing compute load
        sample_rate: sample rate to use for the SDR, needs to be adjusted in-line with target_sr, see process_iq()
        target_sr: target sample rate to resample to for processing, see process_iq()
        verbose: if True, log more information
        """

        # Ensure any existing stream is stopped
        s_acq.streamer.stop_stream()

        # Kick off a SignalStream - this runs in the background, continuously updating the current signal buffer
        s_acq.streamer.start_stream(
            center_freq=ADSB_FREQUENCY,
            sample_rate=sample_rate,
            n_reads_per_acquisition=n_reads_per_acquisition,
            n_samples_per_read=n_samples_per_read,
            sleep_length_s=sleep_length_s
            / 2,  # Ensure the signal stream is updated more frequently than the reads happen
            sdr_gain="auto",
        )
        try:
            # Give the streamer a moment to start up
            time.sleep(3)

            while self.enabled:  # TODO CWI finish plugging this in, figure out if we can avoid manually adding the messages to the cache
                self._batch_generate_hexes(
                    n=n,
                    sleep_length_s=sleep_length_s,
                    sample_rate=sample_rate,
                    target_sr=target_sr,
                    n_reads_to_try_to_parse=n_reads_to_try_to_parse,
                    verbose=verbose,
                )

        finally:
            # Always ensure the stream is stopped on exit
            s_acq.streamer.stop_stream()

    def _batch_generate_hexes(
        self,
        n: int,
        sleep_length_s: float,
        n_reads_to_try_to_parse: int,
        sample_rate: float = 2.4e6,
        target_sr: float = 12e6,
        verbose: bool = False,
    ):
        """Inner loop of the main process. Intended to run repeatedly, indefinitely.

        Args:
            n: The number of reads to process in this batch.
            verbose (bool, optional): _description_. Defaults to False.
            sleep_length_s: Time to wait between reads, purely for managing compute load.
            sample_rate: Sample rate to use for the SDR, needs to be adjusted in-line with target_sr, see process_iq().
            target_sr: Target sample rate to resample to for processing, see process_iq().
            n_reads_to_try_to_parse: Number of reads to try to parse in each batch.
            verbose: If True, log more information.

        Returns:
            None, the parsed messages are added to the RecentMessagesCache.
        """
        hits = 0

        t0 = datetime.now()
        for i in range(n):
            try:
                # Sleep a bit to allow the signal streamer to fill the buffer, and to not hammer CPU
                time.sleep(sleep_length_s)
                signal = s_acq.streamer.get_current_signal()
                # Pick the read with the highest peak, which is likely an ADSB pulse
                indices_of_strongest_signal = s_acq.get_indices_of_highest_peaks(
                    signal, verbose=verbose, n=n_reads_to_try_to_parse
                )
                # Loop over the top N strongest signals
                for i, index in enumerate(indices_of_strongest_signal):
                    iq = signal[index]

                    # Only do the expensive signal processing on the strongest read
                    iq_mag = preprocess_iq(
                        iq, orig_sr=sample_rate, target_sr=target_sr, verbose=verbose
                    )
                    iq_mag_bin = convert_to_binary(iq_mag)
                    adsb = decode_to_adsb(iq_mag_bin, verbose=verbose)

                    if adsb is not None:
                        # We have what looks like an ADSB message, try to decode it
                        if verbose:
                            logger.info("ADSB message identified, attempting to decode")
                        hex_string = convert_adsb_binary_to_hex(adsb)
                        hits += 1
                        # multiple timestamped messages are required to decode true position
                        hit = ADSBPreParsedHit(
                            timestamp=datetime.now().timestamp(),
                            hex_string=hex_string,
                            index=index,
                        )
                        RecentMessagesCache.add_message(hit)

            except Exception as e:
                if verbose:
                    logger.warning(
                        "Error processing read, moving on to next read", error=e
                    )
                # Don't worry if one read fails, just move on to the next read
                # We expect to receive chopped off signals and imperfect data

        if verbose:
            t1 = datetime.now()
            delta_t = (t1 - t0).total_seconds()

            logger.info("Hit rate: %s\t/\t%s", hits, n)
            logger.info("Total processing time for %s reads: %s seconds", n, delta_t)
            logger.info("Average time per read: %s seconds", delta_t / n)


ADSBStreamer = ADSBStreamer()
RecentMessagesCache = ADSBRecentMessagesCache()


def main_ADSBStreamer(n_seconds=10):
    """Example pipeline to acquire and process an ADSB signal from an SDR using the ADSBStreamer"""

    sample_rate: float = 2.4e6
    target_sr: float = 12e6
    n_reads_per_acquisition: int = 64
    n_reads_to_try_to_parse: int = 64
    n_samples_per_read: int = 4096
    sleep_length_s: float = 0.01
    sdr_gain: str = "auto"
    n = 1
    verbose = False

    # Start the ADSBStreamer
    ADSBStreamer.start_adsb_stream(
        sample_rate=sample_rate,
        target_sr=target_sr,
        n_reads_per_acquisition=n_reads_per_acquisition,
        n_reads_to_try_to_parse=n_reads_to_try_to_parse,
        n_samples_per_read=n_samples_per_read,
        sleep_length_s=sleep_length_s,
        sdr_gain=sdr_gain,
        n=n,
        verbose=verbose,
    )

    for i in range(n_seconds):
        temp_adsb_messages_df = RecentMessagesCache.decode_current_cached_messages(
            verbose=verbose
        )

        logger.info(
            "Main ADSBStreamer: Running... %s seconds elapsed\tprocessed %s messages",
            i,
            len(temp_adsb_messages_df),
        )
        time.sleep(1)

    # Stop the streamer
    ADSBStreamer.stop_stream()

    # Decode all cached messages
    adsb_messages_df = RecentMessagesCache.decode_current_cached_messages(verbose=True)

    return adsb_messages_df


def main():
    """Example pipeline to acquire and process an ADSB signal from an SDR"""

    # reccomended upper limit of sample rate. Fast enough to oversample
    sample_rate = 2.4e6
    n_reads = 64
    n_samples = 4096

    # choose integer samples per microsecond: 12 samples/us
    # This is a good choice, because 6x2 = 12, and 5*2.4=12, so we can sample to 12, then subsample back to 1 block per 0.5us
    target_sr = 12e6

    # samples_per_us_after_upsampling = int(round(target_sr / 1e6))  # 12

    iq = get_example_dataset(
        center_freq=ADSB_FREQUENCY,
        sample_rate=sample_rate,
        n_reads=n_reads,
        n_samples=n_samples,
    )

    iq_mag = preprocess_iq(iq, orig_sr=sample_rate, target_sr=target_sr)

    iq_mag_bin = convert_to_binary(iq_mag)

    adsb = decode_to_adsb(iq_mag_bin)

    return adsb


if __name__ == "__main__":
    main()


### DEPRECATED functions


def _batch_decode_adsb(
    n: int,
    sleep_length_s,
    sample_rate: float = 2.4e6,
    target_sr: float = 12e6,
    verbose: bool = False,
):
    """
    NOTE: retained for experimentation in notebooks, but has been now replaced by the batch_generate_hexes method of the ADSBStreamer class
    Inner loop of the main process. Intended to run repeatedly, indefinitely.

    Args:
        n: The number of reads to process in this batch.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    hits = 0

    pre_parsed_msgs = []

    t0 = datetime.now()
    for i in range(n):
        try:
            # Sleep a bit to allow the signal streamer to fill the buffer, and to not hammer CPU
            time.sleep(sleep_length_s)
            signal = s_acq.streamer.get_current_signal()
            # Pick the read with the highest peak, which is likely an ADSB pulse
            indices_of_strongest_signal = s_acq.get_indices_of_highest_peaks(
                signal, verbose=verbose, n=20
            )
            # Loop over the top N strongest signals
            for i, index in enumerate(indices_of_strongest_signal):
                iq = signal[index]

                # Only do the expensive signal processing on the strongest read
                iq_mag = preprocess_iq(
                    iq, orig_sr=sample_rate, target_sr=target_sr, verbose=verbose
                )
                iq_mag_bin = convert_to_binary(iq_mag)
                adsb = decode_to_adsb(iq_mag_bin, verbose=verbose)

                if adsb is not None:
                    # We have what looks like an ADSB message, try to decode it
                    if verbose:
                        logger.info("ADSB message identified, attempting to decode")
                    hex_string = convert_adsb_binary_to_hex(adsb)
                    decoded = rs1090.decode(hex_string)
                    if decoded:
                        if verbose:
                            logger.warning(
                                "\t ! (i=%d) ADSB message decoded: %s", i, decoded
                            )
                        hits += 1
                        # (ts, {message}), multiple timestamped messages are required to decode true position
                        hit = (datetime.now().timestamp(), hex_string, decoded, index)
                        pre_parsed_msgs.append(hit)
                    else:
                        decoded = None

        except Exception as e:
            if verbose:
                logger.warning("Error processing read, moving on to next read", error=e)
            # Don't worry if one read fails, just move on to the next read
            # We expect to receive chopped off signals and imperfect data

    # The rs1090 decoder can infer latlons when multiple messages are provided with timestamps, so we decode them all
    # together here
    timestamps = [x[0] for x in pre_parsed_msgs]
    hexes = [x[1] for x in pre_parsed_msgs]
    parsed_msgs = rs1090.decode(msg=hexes, timestamp=timestamps)

    if verbose:
        t1 = datetime.now()
        delta_t = (t1 - t0).total_seconds()

        logger.info("Hit rate: %s\t/\t%s", hits, n)
        logger.info("Total processing time for %s reads: %s seconds", n, delta_t)
        logger.info("Average time per read: %s seconds", delta_t / n)
    return parsed_msgs


def _batch_generate_hexes(
    n: int,
    sleep_length_s,
    sample_rate: float = 2.4e6,
    target_sr: float = 12e6,
    n_reads_to_try_to_parse=20,
    verbose: bool = False,
):
    """
    NOTE, replaced by ADSBStreamer class method

    ALTERNATIVE TO _batch_decode_adsb Inner loop of the main process. Intended to run repeatedly, indefinitely.

    Args:
        n: The number of reads to process in this batch.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    hits = 0

    pre_parsed_msgs = []

    t0 = datetime.now()
    for i in range(n):
        try:
            # Sleep a bit to allow the signal streamer to fill the buffer, and to not hammer CPU
            time.sleep(sleep_length_s)
            signal = s_acq.streamer.get_current_signal()
            # Pick the read with the highest peak, which is likely an ADSB pulse
            indices_of_strongest_signal = s_acq.get_indices_of_highest_peaks(
                signal, verbose=verbose, n=n_reads_to_try_to_parse
            )
            # Loop over the top N strongest signals
            for i, index in enumerate(indices_of_strongest_signal):
                iq = signal[index]

                # Only do the expensive signal processing on the strongest read
                iq_mag = preprocess_iq(
                    iq, orig_sr=sample_rate, target_sr=target_sr, verbose=verbose
                )
                iq_mag_bin = convert_to_binary(iq_mag)
                adsb = decode_to_adsb(iq_mag_bin, verbose=verbose)

                if adsb is not None:
                    # We have what looks like an ADSB message, try to decode it
                    if verbose:
                        logger.info("ADSB message identified, attempting to decode")
                    hex_string = convert_adsb_binary_to_hex(adsb)
                    hits += 1
                    # multiple timestamped messages are required to decode true position
                    hit = ADSBPreParsedHit(
                        timestamp=datetime.now().timestamp(),
                        hex_string=hex_string,
                        index=index,
                    )
                    pre_parsed_msgs.append(
                        hit
                    )  # TODO don't return anything, just add to cache
                    RecentMessagesCache.add_message(hit)

        except Exception as e:
            if verbose:
                logger.warning("Error processing read, moving on to next read", error=e)
            # Don't worry if one read fails, just move on to the next read
            # We expect to receive chopped off signals and imperfect data

    if verbose:
        t1 = datetime.now()
        delta_t = (t1 - t0).total_seconds()

        logger.info("Hit rate: %s\t/\t%s", hits, n)
        logger.info("Total processing time for %s reads: %s seconds", n, delta_t)
        logger.info("Average time per read: %s seconds", delta_t / n)
    return pre_parsed_msgs


def get_example_dataset(
    center_freq=1090.0, sample_rate=2.4e6, n_reads=32, n_samples=4096
):
    try:
        sdr = s_acq.get_sdr(center_freq=center_freq, sample_rate=sample_rate)

        s = s_acq.acquire_signal(sdr, n_reads=n_reads, n_samples=n_samples)

        # Grab the read with the highest peak, which is likely an ADSB pulse
        highest_peak_read = s_acq.get_index_of_highest_peak(s)
        iq = s[highest_peak_read]

        return iq
    finally:
        sdr.close()
