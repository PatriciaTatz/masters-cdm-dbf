# CDM-DBF Simulator

A Python-based simulation framework for evaluating Code Division Multiplexing (CDM) for satellite communications. The project investigates how CDM can be used to reduce digital beamforming complexity while maintaining signal recovery performance. The simulator supports modulation, spreading/despreading, beamforming, interference analysis, Doppler estimation, and BER performance evaluation.

## Features
- Code Division Multiplexing (CDM) using orthogonal spreading codes
- Walsh-Hadamard (WH) and Zadoff-Chu (ZC) code support
- Baseband (BB) and Intermediate Frequency (IF) spreading architectures
- Multi-element phased-array receiver simulation
- Digital despreading and signal recovery
- Angle-of-arrival estimation
- Digital beamforming
- Doppler shift estimation
- Power Spectral Density (PSD) analysis
- BER performance evaluation
- BPSK and QPSK modulation support

## Repository Structure

```text
.
├── analysis/              # BER, Doppler, and performance simulations
├── components/            # Core processing blocks
│   ├── CDMx.py
│   ├── Beamformer.py
│   ├── Filter.py
│   ├── Mixer.py
│   ├── SpecAn.py
│   └── ...
├── models/                # Shared data structures
├── signals/               # Signal generation
├── results/               # Simulation outputs
├── config.py              # Simulation configuration
├── utils.py
└── receiver.py            # Receiver processing chain
```

---

## Receiver Overview

The main receiver implementation is contained in `receiver.py`. It imports processing blocks from the `components` package and performs the complete receive chain from signal reception to CDM recovery and beamforming.

### Receiver Processing Chain

```text
Received Signal
       │
       ▼
Downconversion
       │
       ▼
Image Rejection Filtering
       │
       ▼
CDM Spreading
       │
       ▼
Signal Combination
       │
       ▼
CDM Recovery
       │
       ▼
Optional Beamforming
       │
       ▼
Doppler / Performance Analysis
---

## Running the Receiver

To execute the receiver processing chain:

```bash
python receiver.py
```

The script creates a simulation based on `config.py`, generates a signal, assembles the receiver chain, performs signal recovery, and optionally estimates Doppler frequency.

---

## Running BER Simulations

BER simulations are executed from the `analysis/` directory rather than from `receiver.py`.

Example:

```bash
python analysis/<script_name>.py
```

The analysis scripts perform Monte Carlo simulations and parameter sweeps to evaluate:

- BER versus Eb/N0
- BER versus spreading factor (Lc)
- BER before and after CDM recovery
- BER under interference conditions
- Doppler performance
- 
## Configuration

Simulation parameters are defined using the `CDMcfg` dataclass in `config.py`.
This one file contains all signal, receiver, and feature configuration options used throughout the simulator. 

### Example Configuration

```python
config = CDMcfg(
    simtime=1e-3,
    fb=10e6,
    fc=1e9,
    alpha=0.25,
    lc=32,
    sps_chip=2,
    no_elem=8,
    codefam="WH",
    modType="BPSK",
    SPREADING_STAGE="BB",
    BEAMFORM=False,
    INTERFERENCE=False,
    DOPPLER=False
)
```

### Feature Parameters

| Parameter | Description |
|-----------|-------------|
| `BEAMFORM` | Enable digital beamforming after CDM recovery |
| `CALIBRATE` | Model an ideal matched-filter receiver |
| `INTERFERENCE` | Inject interference during signal generation |
| `CARRIER` | Enable RF carrier modulation |
| `SPREADING_STAGE` | Perform spreading at Baseband (`BB`) or Intermediate Frequency (`IF`) |
| `DOPPLER` | Enable Doppler modelling |

### Signal Parameters

| Parameter | Description |
|-----------|-------------|
| `fc` | Carrier frequency (Hz) |
| `fb` | Symbol rate (Hz) |
| `elevation` | Satellite elevation angle (degrees) |
| `alpha` | Root Raised Cosine (RRC) roll-off factor |
| `beta` | Doppler ratio |
| `span` | RRC filter span (symbols) |
| `modType` | Modulation scheme (`BPSK` or `QPSK`) |

### Receiver Parameters

| Parameter | Description |
|-----------|-------------|
| `lc` | Spreading code length |
| `sps_chip` | Samples per chip |
| `codefam` | Spreading code family (`WH`, `Gold`, `ZC`, etc.) |
| `no_elem` | Number of multiplexed array elements |
| `degree` | Gold code degree |
| `codeids` | Code indices assigned to antenna elements |

### Automatically Calculated Parameters

The following parameters are derived automatically when the configuration object is created:

| Parameter | Description |
|-----------|-------------|
| `sps_symbol` | Samples per symbol |
| `ts` | Symbol period |
| `fs` | Sampling frequency |
| `b` | Occupied RRC bandwidth |
| `fIF` | Intermediate frequency |
| `fd` | Doppler frequency shift |

---

## Visualisation

The simulator contains built-in plotting utilities for visualising:

- Received waveforms
- Downconverted signals
- Spread signals
- Combined CDM signals
- Recovered signals
- Power Spectral Density (PSD)
- Beam patterns
- Array responses


---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Research Context

This repository was developed as part of a Master's research project investigating Code Division Multiplexing (CDM) for reducing digital beamforming complexity in phased-array satellite communication systems.
