# plane-sailing
A tool for processing aircraft signals, storing them, and plotting them

# Installation

Assumes `cosmosdr` is checked out as a sibling of this repo (i.e. `../cosmosdr`), since it's pulled in as a local editable install via `[tool.uv.sources]`.

- install `librtlsdr` (native dep for `pyrtlsdr`)
  - mac: `brew install librtlsdr`
  - debian/ubuntu: `sudo apt install librtlsdr-dev`
- `uv sync`
- `uv run python -c "import planesailing; import cosmosdr; print(planesailing, cosmosdr)"`

### macOS: librtlsdr lookup

On macOS, `pyrtlsdr` won't find Homebrew's `librtlsdr` at runtime unless its lib dir is on `DYLD_LIBRARY_PATH`. Don't put this in `.zshrc` — `DYLD_*` vars affect every process launched from the shell. Scope it to the project instead.

Recommended: [direnv](https://direnv.net/) with an `.envrc` in the repo root:

```
export DYLD_LIBRARY_PATH="$(brew --prefix librtlsdr)/lib:$DYLD_LIBRARY_PATH"
```

Or prepend it inline:

```
DYLD_LIBRARY_PATH="$(brew --prefix librtlsdr)/lib" uv run python -m planesailing
```

Linux doesn't need this — `apt`'s `librtlsdr` lands in a path `ld.so` already searches.

## Running

### Stream ADS-B from the SDR

Plug in an RTL-SDR dongle, then:

```
uv run python -m planesailing.main
```

This calls `main_ADSBStreamer()`, which:
1. Starts `ADSBStreamer` in a background thread — it tunes to 1090 MHz, reads IQ samples, and decodes ADS-B frames into the in-memory `RecentMessagesCache`
2. Polls the cache every second for 10 seconds, logging the decoded message count
3. Stops the SDR stream and returns a DataFrame of all decoded messages

To adjust the run duration, call it directly:

```python
from planesailing.main import main_ADSBStreamer
df = main_ADSBStreamer(n_seconds=60)
print(df)
```

The cache has a 5-minute TTL, so messages accumulate across repeated calls within the same process.

### COSMOS_N sweep experiment

`infra/experiment/cosmos_n_cycle.sh` runs the collector for 24h per value of
`COSMOS_N` through the sequence `1,2,3,4,6,8,...,30`. A user crontab entry fires
it daily at 00:00 local; each firing bumps `COSMOS_N` in `infra/.env` and
rebuilds+recreates only the collector container (Postgres stays up). After
n=30's day it stops the collector and writes `done` to
`infra/experiment/state`.

- Current position: `infra/experiment/state` (`start` = armed, first firing
  applies n=1; a number = the n now running; `done` = finished)
- Flip history: `infra/experiment/cycle.log`; stray cron/docker output: `cron.log`
- Each day's rows are linked to their config via the `PipelineConfig`
  fingerprint in `cosmos.pipeline_configs`, so analysis can group by n directly
- Recovery: `cosmos_n_cycle.sh --apply-current` re-applies the state file's
  value without advancing; `--dry-run` previews the next flip
- To end the experiment early: `crontab -e` and remove the line

### Run the database
NOTE: the main pipeline is not hooked up to the DB yet
```
export PLANE_SAILING_DB_USER=<whatever you want>
export PLANE_SAILING_DB_PASSWORD=<whatever you want>
export PLANE_SAILING_DB_NAME=<whatever you want>
cd plane-sailing/infra && docker compose up
```
