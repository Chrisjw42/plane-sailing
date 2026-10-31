"""Signal processing functions that are specific to ADSB processing requirements.

Typically these function are used to operate on pre existing data, e.g. performing transformations on arrays. This
is as opposed to managing signal pipelines, caching, etc.

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""

from copy import copy

import numpy as np
import pandas as pd
import structlog

import cosmosdr.signal_processing as s_proc


logger = structlog.get_logger()

# Total number of bits in a single ADSB message
ADSB_BITS = 112


# Half-width of the window kept around the strongest pulse, in samples at the 12 MS/s target rate. A reply is
# 120 us long (8 us preamble + 112 us of bits) and its strongest pulse can be anywhere in it, so each half must hold
# a whole frame: 1600 samples = 133 us. (It was 1000 = 83 us: a frame then fitted only if its strongest pulse was
# 37-83 us in. Otherwise the preamble fell outside the window, or the tail did and np.packbits zero-padded it: in
# October ~35% of stored DF20 frames ended in padded zeros, and nearly all of those were filed under a ghost address.)
TRIM_HALF_WIDTH = 1600


def trim_iq_around_peak(iq_mag, n_either_side=TRIM_HALF_WIDTH):
    """Keep a window of 2 * n_either_side samples around the strongest pulse, so the whole frame is in it.

    The window is clamped to start inside the read: a peak near the start used to give a negative slice start,
    an empty window, and a ValueError from threshold_elbow that abandoned the rest of the grab.
    """
    idx = int(iq_mag.argmax())
    start = max(idx - n_either_side, 0)
    return iq_mag[start : start + 2 * n_either_side]


def shift_to_optimal_phase(iq_mag, samples_per_us, verbose=False):
    """Given that the signal is almost certainly not aligned perfectly with the initial sampling start time, and our
    eventual aim of bucketing these samples into 0.5us buckets, we want to try and center the aircraft pulses well
    onto the buckets.

    e.g. If we have oversampled and have 12x the samples, there are 12 possible 'start points' or phase positions.

    We assess each phase position, scoring them based on how much difference there is between neighboring 6-sample
    blocks (i.e. 0.5us blocks). The phase position with the highest average difference between neighboring blocks
    is likely to be the best position.

    # TODO parametrise this to different sample rates and whatnot
    """
    iq_mag = copy(iq_mag)

    max_score = -1
    max_score_phase = None

    # We check each of the first n starting points, after a while we just get back to the start of the cycle again so we can stop checking
    steps_to_check = int(samples_per_us)
    scores = {}

    # Step through the possible starting points
    for phase in range(0, steps_to_check - 1):
        data_phase = iq_mag[phase:]

        # 0,0,0,0,0,0,1,1,1....
        indices = np.arange(len(data_phase)) // steps_to_check

        data_phase = pd.Series(data_phase, index=indices)

        block_averages = data_phase.groupby(data_phase.index).mean()

        # calculate the differences between the blocks
        deltas = (block_averages - block_averages.shift(1)).abs()
        score = deltas.mean()
        scores[phase] = score

        # The score will oscillate naturally, so we only update if the score is significantly higher than the previous max
        if score > max_score * 1.01:
            max_score = score
            max_score_phase = phase

    # TODO do this in numpy, dict is lame
    if verbose:
        logger.info("----------")
        logger.info("max_score: %s", max_score)
        logger.info("max_score_phase: %s", max_score_phase)

    iq_mag = iq_mag[max_score_phase:]
    return iq_mag


def downsample_to_buckets(iq_mag, samples_per_us):
    """
    Downsample the data back to the convenient bucketing (1 bucket per 0.5us)

    Uses mean to downsample, but we could also take the max from the bucket.
    """
    # 0,0,0,0,0,0,1,1,1,1,1...
    # We want 0.5us buckets, so we need to half the samples_per_us
    indices = np.arange(len(iq_mag)) // (samples_per_us / 2)

    data_downsampled = pd.DataFrame(iq_mag).groupby(indices).mean()
    return data_downsampled[0].to_numpy()


def threshold_elbow(s, upper_pulses_to_ignore=20):
    """
    Identify the largest jump in values, after they have been sorted. Works well in cases where the
    pulses are well above the background noise.

    If we received an ADSB from an aircraft, there wil be dozens of high strength pulses at roughly the same strength,
    so we ignore the top N pulses.
    """
    x = np.sort(s)[:-upper_pulses_to_ignore]
    diffs = np.diff(x)

    if len(diffs) == 0:
        raise ValueError("Not enough unique values to determine a threshold")

    # grab the location of largest jump
    gap_idx = np.argmax(diffs)
    # Return the value half-way between the largest jump
    return np.mean([x[gap_idx], x[gap_idx + 1]])  # threshold is value at gap


def convert_to_binary(iq_mag):
    """
    Use an elbow analysis to identify a threshold, then convert the signal to binary based on this threshold.

    The returned signal is a binary array indicating the presence of pulses, it is NOT yet decoded, or represented as binary
    """

    threshold = threshold_elbow(iq_mag)
    iq_mag_bin = (iq_mag > threshold).astype(int)
    return iq_mag_bin


def preprocess_iq(iq, orig_sr, target_sr, verbose=False):
    """Resample the IQ data to a target sample rate, and optimise to highlight the presence of pulses

    The returned data is a 1D array of magnitudes, with one value per 0.5us bucket

    Then, Find the optimal phase starting position
    - We don't know the exact timing of the pulses
    - It could be anywhere from n, ..., n+11
    - We can define the most optimal position as that which has the highest difference between neighboring 6-width blocks
    """
    iq_resampled, sr = s_proc.resample_to_target(iq, orig_sr, target_sr)
    iq_mag = s_proc.iq_to_envelope(iq_resampled)

    iq_mag = trim_iq_around_peak(iq_mag)

    samples_per_us = int(round(target_sr / 1e6))  # 12

    iq_mag = shift_to_optimal_phase(
        iq_mag, samples_per_us=samples_per_us, verbose=verbose
    )

    iq_mag = downsample_to_buckets(iq_mag, samples_per_us=samples_per_us)

    return iq_mag


def decode_to_adsb(s_bin, verbose=False):
    """
    Take a binary pulse signal, and convert it to ADSB 1s and 0s.
    """
    # mode S preamble, 8us
    preamble = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    len_preamble = len(preamble)
    assert len_preamble == 16

    # Find where the preamble starts, if at all
    for i in range(len(s_bin)):
        candidate = s_bin[i : i + len_preamble]
        if np.array_equal(candidate, preamble):
            if verbose:
                logger.info("ADSB preamble identified @ idx: %s, damn!", i)
            break
        if i >= (len(s_bin) - len_preamble):
            if verbose:
                logger.info("No ADSB preamble found, exiting")
            return None

    # the index of the last piece of the preamble
    final_preamble_idx = i + len_preamble

    # Drop the preamble off the front of the burst
    adsb = s_bin[final_preamble_idx:]

    # Note, the last part of the signal may indeed be one or many zeros, if this doens't fit into a standard msg length, that's likely why
    adsb = np.trim_zeros(adsb)

    # If we end up with exactly one less bit than was expected, then assume we just chopped off a bit
    if (len(adsb) + 1) == ADSB_BITS * 2:
        adsb = np.append(adsb, 0)

    # Take every other value (a.k.a. take the value in the first half of each bucket
    # This is the same as 'if there is a pule in the first half, then 1.'
    adsb = adsb[::2]
    return adsb


def convert_adsb_binary_to_hex(adsb):
    """Convert a binary array of 1s and 0s to a hex string, as is required by rs1090.decode()"""
    # Step 1: pack bits into bytes
    packed = np.packbits(adsb)

    # Step 2: convert to hex string
    hex_string = packed.tobytes().hex()

    return hex_string


# ---- Address confirmation (ghost suppression) ----------------------------------------------------------------------
#
# DF4/5/20/21 replies carry no checkable parity: the transponder XORs its address into the CRC, and the receiver
# recovers the address as CRC(data) XOR AP. One corrupted bit therefore yields a different, plausible-looking
# address, with the real aircraft's altitude. Only frames whose parity can be checked prove an address exists.

MODE_S_GENERATOR = 0x1FFF409  # CRC-24 generator polynomial (25 bits)


def mode_s_crc_remainder(hex_frame: str) -> int:
    """Remainder of the whole frame (data + parity bits) divided by the Mode S generator."""
    bits = len(hex_frame) * 4
    v = int(hex_frame, 16)
    for i in range(bits - 1, 23, -1):
        if v >> i & 1:
            v ^= MODE_S_GENERATOR << (i - 24)
    return v & 0xFFFFFF


def df11_parity_ok(hex_frame: str) -> bool:
    """A 56-bit DF11 all-call reply is intact if its remainder is an interrogator code (II or SI: below 0x80).

    rs1090 decodes DF11 without checking this, so a corrupted DF11 would vouch for a ghost."""
    if len(hex_frame) != 14 or int(hex_frame[:2], 16) >> 3 != 11:
        return False
    return mode_s_crc_remainder(hex_frame) < 0x80


def confirmed_addresses(decoded) -> set:
    """icao24s vouched for by a frame whose parity can be checked: DF17/18 (rs1090 drops those on a bad CRC) or an
    intact DF11. `decoded` is the DataFrame of rs1090.decode output for the message cache."""
    if (
        decoded is None
        or decoded.empty
        or "icao24" not in decoded.columns
        or "df" not in decoded.columns
    ):
        return set()
    df_codes = decoded["df"].astype(str)
    confirmed = set(decoded.loc[df_codes.isin(["17", "18"]), "icao24"].dropna())
    if "frame" in decoded.columns:
        df11 = decoded[df_codes == "11"]
        confirmed |= {
            a
            for a, f in zip(df11["icao24"], df11["frame"])
            if isinstance(f, str) and df11_parity_ok(f)
        }
    return confirmed
