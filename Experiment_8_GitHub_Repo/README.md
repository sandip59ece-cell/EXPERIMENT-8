# Experiment 8 — Matched Filtering, ISI and Eye Diagrams

No communications toolbox is required.

## Contents
- BPSK mapping
- RRC pulse shaping
- AWGN channel
- RRC matched filtering
- Total cascade-delay validation
- Eye diagram
- BER vs SNR
- Eye height vs SNR
- BER vs sampling offset
- Roll-off comparison

## Run
```bash
pip install -r requirements.txt
python experiment8_bpsk_simulation.py
```

For `span=8` and `sps=8`, each RRC filter has 32 samples of group delay and the
two-filter cascade has **64 samples** total delay. Symbol decisions are made
only after this delay is compensated.
