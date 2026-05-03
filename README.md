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

### run the database
```
export PLANE_SAILING_DB_USER=<whatever you want>
export PLANE_SAILING_DB_PASSWORD=<whatever you want>
export PLANE_SAILING_DB_NAME=<whatever you want>
cd plane-sailing/infra && docker compose up
```
