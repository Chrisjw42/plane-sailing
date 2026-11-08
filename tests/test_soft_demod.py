"""Soft-decision demodulation (planesailing/soft_demod.py) on synthetic, band-limited Mode S replies.

The replies are built as the air delivers them: a pulse-position envelope at 24 MS/s, low-passed to the 2.4 MS/s
band, decimated, then complex noise of unit power added, so each read looks like one of the collector's 4096-sample
reads. The frames are made up (valid parity, invented addresses).
"""

import numpy as np
import pytest

import cosmosdr.signal_acquisition as s_acq
import cosmosdr.signal_processing as s_proc
from planesailing import main as ps_main
from planesailing import soft_demod as sd

READ = 4096
RATE = 2.4e6


def make_frame(df: int, payload: int, nbits: int = 112, ap: int = 0) -> str:
    data = (df << (nbits - 29)) | (payload & ((1 << (nbits - 29)) - 1))
    value = data << 24
    value |= sd.crc_remainder(value, nbits) ^ ap
    return f"{value:0{nbits // 4}x}"


def flip(hexs: str, *bits: int) -> str:
    n = len(hexs) * 4
    v = int(hexs, 16)
    for k in bits:
        v ^= 1 << (n - 1 - k)
    return f"{v:0{len(hexs)}x}"


def burst(hexs: str, over: int = 10) -> np.ndarray:
    per_us = RATE * over / 1e6
    bits = bin(int(hexs, 16))[2:].zfill(len(hexs) * 4)
    env = np.zeros(int(round((12 + len(bits)) * per_us)))
    for t in [0.0, 1.0, 3.5, 4.5] + [8 + k + (0.0 if b == "1" else 0.5) for k, b in enumerate(bits)]:
        env[int(round(t * per_us)) : int(round((t + 0.5) * per_us))] = 1.0
    spec = np.fft.fft(env)
    spec[np.abs(np.fft.fftfreq(len(env), d=1 / (RATE * over))) > RATE / 2] = 0
    return np.fft.ifft(spec).real[::over]


def read_with(frames, seed=1):
    """One read: unit-power complex noise plus each (hex, start sample, snr_db) reply."""
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=READ) + 1j * rng.normal(size=READ)) / np.sqrt(2)
    for hexs, at, snr_db in frames:
        b = burst(hexs)
        x[at : at + len(b)] += 10 ** (snr_db / 20) * b
    return x.astype(np.complex64)


DF17 = make_frame(17, 0x5A5A5A_123456789ABCDE)
DF17_ADDRESS = int(DF17[2:8], 16)
DF11 = make_frame(11, 0x0F0F0F, nbits=56, ap=0x1F)  # parity carries an interrogator code, as real DF11s do


def knowing(*addresses):
    k = sd.KnownAddresses()
    for a in addresses:
        k.add(a)
    return k


def power(x):
    y, _ = s_proc.resample_to_target(x, RATE, 12e6)
    return (y.real ** 2 + y.imag ** 2).astype(np.float64)


def test_clean_df17_makes_its_address_known():
    known = sd.KnownAddresses()
    assert sd.demodulate(read_with([(DF17, 1000, 20.0)]), known=known) == [DF17]
    assert DF17_ADDRESS in known


def test_one_bit_error_is_repaired_for_a_known_address():
    assert sd.demodulate(read_with([(flip(DF17, 40), 1000, 20.0)]), known=knowing(DF17_ADDRESS)) == [DF17]


def test_a_repaired_frame_needs_an_address_already_heard_cleanly():
    assert sd.demodulate(read_with([(flip(DF17, 40), 1000, 20.0)]), known=sd.KnownAddresses()) == []
    clock = [0.0]
    known = sd.KnownAddresses(ttl_s=300, clock=lambda: clock[0])
    known.add(DF17_ADDRESS)
    clock[0] = 301.0
    assert sd.demodulate(read_with([(flip(DF17, 40), 1000, 20.0)]), known=known) == []


def test_two_weak_bits_are_repaired():
    p = power(read_with([(DF17, 1000, 25.0)]))
    start = sd.frames_in_power(p, known=sd.KnownAddresses())[0].at
    med = float(np.median(p))
    sent = bin(int(DF17, 16))[2:].zfill(112)
    for k in (40, 77):  # each made wrong, barely: the wrong half-slot 5% stronger than the right one
        a = start + sd.DATA0 + sd.SPU * k
        right, wrong = (a, a + sd.HALF) if sent[k] == "1" else (a + sd.HALF, a)
        p[right : right + sd.HALF] = 2 * med
        p[wrong : wrong + sd.HALF] = 2.1 * med
    out = sd.frames_in_power(p, known=knowing(DF17_ADDRESS))
    assert [(r.hex, r.repaired) for r in out] == [(DF17, 2)]


def test_a_cleanly_sent_two_bit_error_is_dropped_not_invented():
    assert sd.demodulate(read_with([(flip(DF17, 40, 77), 1000, 20.0)]), known=knowing(DF17_ADDRESS)) == []


def test_every_reply_in_a_read():
    out = sd.demodulate(read_with([(DF17, 300, 20.0), (DF11, 2500, 20.0)]), known=sd.KnownAddresses())
    assert out == [DF17, DF11]


@pytest.mark.parametrize("strong_db", [12.0, 35.0])
def test_a_loud_reply_does_not_hide_a_weak_one(strong_db):
    out = sd.demodulate(read_with([(DF17, 300, strong_db), (DF11, 2500, 12.0)], seed=3), known=sd.KnownAddresses())
    assert out == [DF17, DF11]


def test_a_short_reply_in_the_last_120_us_of_a_read():
    at = READ - 64 * 24 // 10 - 20  # starts ~72 us before the end: room for 56 bits, not 112
    assert sd.demodulate(read_with([(DF11, at, 20.0)]), known=sd.KnownAddresses()) == [DF11]


def test_a_strong_reply_we_reject_is_not_mined_for_false_preambles():
    # clean DF16s (air-air, not passed on) at 12-30 dB: their own bits must not come out as DF0/4/5/20/21
    rng = np.random.default_rng(16)
    out = []
    for k in range(40):
        df16 = make_frame(16, int(rng.integers(0, 1 << 62)), ap=int(rng.integers(0, 1 << 24)))
        out += sd.demodulate(read_with([(df16, 700, 12.0 + 18.0 * k / 39)], seed=k), known=sd.KnownAddresses())
    assert out == []


def test_noise_alone_gives_no_checked_reply():
    for seed in range(200):
        out = sd.demodulate(read_with([], seed=seed), known=sd.KnownAddresses())
        assert [h for h in out if int(h[:2], 16) >> 3 in (11, 17, 18)] == []


def test_target_rate_must_be_12_ms_s():
    with pytest.raises(ValueError):
        sd.demodulate(read_with([]), target_sr=24e6)


def test_collector_loop_keeps_every_reply_in_a_read(monkeypatch):
    """_batch_generate_hexes on a fake 64-read grab: both replies in the loudest read reach the message cache."""
    grab = np.stack([read_with([], seed=100 + k) for k in range(64)])
    grab[5] = read_with([(DF17, 300, 20.0), (DF11, 2500, 20.0)])

    class FakeStreamer:
        def get_current_signal(self):
            return grab

    monkeypatch.setattr(s_acq, "streamer", FakeStreamer(), raising=False)
    cache = ps_main.ADSBRecentMessagesCache()
    monkeypatch.setattr(ps_main, "RecentMessagesCache", cache)
    monkeypatch.setattr(sd, "KNOWN", sd.KnownAddresses())
    streamer = type(ps_main.ADSBStreamer)()  # the module rebinds the name to its singleton
    streamer._batch_generate_hexes(n=1, sleep_length_s=0.0, n_reads_to_try_to_parse=4)
    assert {DF17, DF11} <= set(cache.get_messages())
