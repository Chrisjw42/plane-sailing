"""Soft-decision Mode S demodulation: one read of IQ in, every reply in it that can be trusted out.

The previous front end (adsb_processing: preprocess_iq -> convert_to_binary -> decode_to_adsb) makes one hard
decision per read: a single threshold (the elbow of the sorted 0.5 us bucket means) turns the whole window into 0/1
pulses, the preamble must match exactly, and each data bit is "was the first half-bucket above the threshold". On
real air (a 3.7 s `auto` recording at 1090 MHz, 2026-10-25) that gives 0 CRC-valid DF17s where a matched decode of
the same air finds 26: every DF17-shaped output had bit errors.

This keeps the per-read interface and changes the decisions:

1. Preamble by correlation, not equality: every sample offset at 12 S/us is scored by the mean power in the four
   preamble pulse slots against the twelve quiet half-slots and against the read's noise floor.
2. Each bit by comparing its own two half-bit energies (PPM: a 1 puts the pulse in the first half). No global
   threshold, so a reply 3 dB or 30 dB above the noise decodes the same way, and a loud reply doesn't hide a weak one.
3. Parity as a check and, for DF17/18, as a corrector: a remainder equal to one bit's syndrome (or to two of the
   least confident bits' syndromes) is flipped back, as dump1090 does, and (also as dump1090 does) only for an
   address already heard with clean parity. Output only DF17/18 with CRC 0 after at
   most that repair, DF11 with an interrogator remainder (< 0x80), and the address/parity replies (DF0/4/5/20/21)
   whose bits were decided cleanly; those carry no checkable parity, so confirmed_addresses still decides them
   before anything is written.
4. Every reply in the read: a read is 1.7 ms and a reply 64-120 us.

On the same 3.7 s, read by read: 23 of the 26 DF17s (the other 3 straddle two reads, which no per-read decoder can
see), 6 more recovered by parity repair, and 5.4 ms per read instead of 30. On synthetic replies it recovers half
of all DF17s at ~8.6 dB SNR, where the old front end needed ~13.5 dB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import cosmosdr.signal_processing as s_proc

SPU = 12  # samples per microsecond at plane-sailing's 12 MS/s target rate
HALF = SPU // 2  # one half-bit slot
DATA0 = 8 * SPU  # the data starts 8 us after the first preamble pulse
PULSES = (0, 12, 42, 54)  # preamble pulses at 0, 1.0, 3.5 and 4.5 us
GAPS = (6, 18, 24, 30, 36, 48, 60, 66, 72, 78, 84, 90)  # the quiet half-slots of the 8 us preamble
PULSE_OVER_GAP = 2.5
PULSE_OVER_NOISE = 4.0
WEAK_BITS = 8  # two-bit repair only among this many least-confident bits
GENERATOR = 0x1FFF409

LONG_DF = (16, 17, 18, 19, 20, 21, 24)
SHORT_DF = (0, 4, 5, 11)
UNCHECKED_DF = (0, 4, 5, 20, 21)  # address/parity: nothing to check here; the vouch filter decides
# Unchecked replies must be decided cleanly: median bit margin >= 0.5 (|e1 - e2| / (e1 + e2)). On the 10-25
# recording's two captures with aircraft this kept every reply whose address a checked frame vouched for (89 of 89,
# 27 of 27) and dropped 62-66% of the rest. DF16 (air-air) is left out: 0 of 57 named a vouched address.
UNCHECKED_MIN_MARGIN = 0.5
KNOWN_TTL_S = 300.0  # as long as plane-sailing's message cache


class KnownAddresses:
    """Addresses heard in a frame whose parity passed with no repair (DF17/18 with CRC 0, DF11 with an interrogator
    remainder), remembered for `ttl_s`. A parity-REPAIRED DF17/18 is passed on only for one of these, as dump1090
    does: repair turns ~8e-6 of corrupt DF17/18-headed frames into CRC-valid ones (1 + 107 + 28 syndromes in 2^24),
    and on its own such a frame would vouch for an aircraft nobody heard (review, 2026-11-03)."""

    def __init__(self, ttl_s: float = KNOWN_TTL_S, clock=time.monotonic):
        self.ttl_s, self.clock, self._seen = ttl_s, clock, {}

    def add(self, address: int) -> None:
        now = self.clock()
        self._seen[address] = now
        if len(self._seen) > 4096:  # forget the expired now and then
            self._seen = {a: t for a, t in self._seen.items() if now - t <= self.ttl_s}

    def __contains__(self, address: int) -> bool:
        t = self._seen.get(address)
        return t is not None and self.clock() - t <= self.ttl_s


KNOWN = KnownAddresses()  # the collector's: one process, one receiver


def crc_remainder(value: int, nbits: int) -> int:
    """Remainder of an nbits-long frame (data + 24 parity bits) divided by the Mode S generator."""
    for i in range(nbits - 1, 23, -1):
        if value >> i & 1:
            value ^= GENERATOR << (i - 24)
    return value & 0xFFFFFF


# syndrome of a single flipped bit at position k (0 = first bit sent) in a 112-bit frame
_SYNDROME_112 = [crc_remainder(1 << (111 - k), 112) for k in range(112)]
_BIT_OF_SYNDROME = {s: k for k, s in enumerate(_SYNDROME_112)}


@dataclass
class Reply:
    at: int  # sample index of the first preamble pulse, at 12 S/us within the slice
    hex: str
    df: int
    repaired: int  # bits flipped back by the parity (DF17/18 only)
    margin: float  # median |first half - second half| / (first + second) over the bits: 0 = coin toss, 1 = clean


def _slot_sum(c: np.ndarray, starts, n: int) -> np.ndarray:
    acc = np.zeros(n)
    for o in starts:
        acc += c[o + HALF : o + HALF + n] - c[o : o + n]
    return acc


def _decide(p: np.ndarray, at: int, nbits: int):
    first = at + DATA0 + SPU * np.arange(nbits)
    idx = first[:, None] + np.arange(HALF)[None, :]
    e1 = p[idx].sum(axis=1)
    e2 = p[idx + HALF].sum(axis=1)
    bits = (e1 > e2).astype(np.uint8)
    conf = np.abs(e1 - e2) / np.maximum(e1 + e2, 1e-30)
    return bits, conf


def _to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _repair_long(value: int, conf: np.ndarray) -> tuple[int, int] | None:
    """A DF17/18 frame with CRC 0, or one or two bits flipped back to get there; None if parity can't vouch."""
    syn = crc_remainder(value, 112)
    if syn == 0:
        return value, 0
    k = _BIT_OF_SYNDROME.get(syn)
    if k is not None and k >= 5:  # never "repair" the DF field itself into a DF17
        return value ^ (1 << (111 - k)), 1
    weak = [int(k) for k in np.argsort(conf)[:WEAK_BITS] if k >= 5]
    for a in range(len(weak)):
        for b in range(a + 1, len(weak)):
            if _SYNDROME_112[weak[a]] ^ _SYNDROME_112[weak[b]] == syn:
                return value ^ (1 << (111 - weak[a])) ^ (1 << (111 - weak[b])), 2
    return None


def frames_in_power(p: np.ndarray, noise: float | None = None, known: KnownAddresses | None = None) -> list[Reply]:
    """Every Mode S reply this demodulator vouches for in a power series at 12 samples per microsecond.

    `known`: addresses for which a parity-repaired DF17/18 may be passed on (default: the module's KNOWN); every
    DF17/18 with CRC 0 and every parity-clean DF11 adds its address to it."""
    known = KNOWN if known is None else known
    short_span = DATA0 + 56 * SPU
    n = len(p) - short_span  # a short reply may start this late; a long one is decided only if it fits
    if n <= 0:
        return []
    if noise is None:
        noise = float(np.median(p)) / np.log(2)  # median of exponential power = mean * ln 2
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    pulse = _slot_sum(c, PULSES, n) / (HALF * len(PULSES))
    gap = _slot_sum(c, GAPS, n) / (HALF * len(GAPS))
    cand = np.flatnonzero((pulse > PULSE_OVER_GAP * gap) & (pulse > PULSE_OVER_NOISE * noise))
    out: list[Reply] = []
    last_end, i = -1, 0
    while i < len(cand):
        j = i
        while j + 1 < len(cand) and cand[j + 1] - cand[i] < DATA0:
            j += 1
        grp = cand[i : j + 1]
        i = j + 1
        at = int(grp[np.argmax(pulse[grp])])
        if at < last_end:
            continue
        fits_long = at + DATA0 + 112 * SPU <= len(p)
        bits, conf = _decide(p, at, 112 if fits_long else 56)
        df = _to_int(bits[:5])
        nb = 112 if df in LONG_DF else 56
        if nb > len(bits):
            continue  # a long reply cut off by the end of the read
        margin = float(np.median(conf[:nb]))
        reply = None
        if df in (17, 18):
            fixed = _repair_long(_to_int(bits), conf)
            if fixed is not None:
                value, nrep = fixed
                address = value >> 80 & 0xFFFFFF  # bits 8-31
                if nrep == 0:
                    known.add(address)
                if nrep == 0 or address in known:
                    reply = Reply(at, f"{value:028x}", df, nrep, margin)
        elif df == 11:
            v = _to_int(bits[:56])
            if crc_remainder(v, 56) < 0x80:
                known.add(v >> 24 & 0xFFFFFF)
                reply = Reply(at, f"{v:014x}", df, 0, margin)
        elif df in UNCHECKED_DF and margin >= UNCHECKED_MIN_MARGIN:
            reply = Reply(at, f"{_to_int(bits[:nb]):0{nb // 4}x}", df, 0, margin)
        if reply is not None:
            out.append(reply)
        if reply is not None or margin >= UNCHECKED_MIN_MARGIN:
            # a reply we used, or one decided cleanly that we didn't (DF16/19/24, an unrepairable DF17): either way
            # its data isn't a place to look for another preamble (review, 2026-11-03: a strong rejected DF16 left
            # ~5 false preambles in its own bits, and ~30% of those came out as clean-looking DF0/4/5/20/21)
            last_end = at + DATA0 + nb * SPU
    return out


def demodulate(iq: np.ndarray, orig_sr: float = 2.4e6, target_sr: float = 12e6,
               known: KnownAddresses | None = None) -> list[str]:
    """One read of complex IQ in, the hex strings of every reply in it out (rs1090.decode takes them as they are)."""
    if int(round(target_sr / 1e6)) != SPU:
        raise ValueError(f"soft_demod's slot geometry assumes a {SPU} MS/s target rate, not {target_sr:g}")
    y, _ = s_proc.resample_to_target(iq, orig_sr, target_sr)
    p = (y.real ** 2 + y.imag ** 2).astype(np.float64)
    return [r.hex for r in frames_in_power(p, known=known)]
