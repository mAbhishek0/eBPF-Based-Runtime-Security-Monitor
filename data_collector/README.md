# eBPF Data Collector for Privilege Escalation Detection

Based on: *"Privilege Escalation Detection and Prediction Method Based on eBPF and Machine Learning"* (IEEE 2024)

This tool uses eBPF (via BCC/Python) to monitor privilege-escalation-relevant system calls in real time and produce labeled CSV datasets (plus a small per-run JSON summary) for the ML detection pipeline.

---

## Prerequisites

```bash
# These should already be installed on your system:
sudo apt install bpfcc-tools python3-bpfcc linux-headers-$(uname -r) gcc
```

Verify:
```bash
dpkg -l | grep bpfcc          # Should show bpfcc-tools, python3-bpfcc
python3 -c "from bcc import BPF; print('BCC OK')"
```

---

## Files

| File | Description |
|------|-------------|
| `collector.py` | Main eBPF data collector (BCC/Python) |
| `simulate_attacks.sh` | Generates attack syscall patterns (label=1) |
| `simulate_normal.sh` | Generates normal syscall patterns (label=0) |

---

## How to Run

### Step 1: Collect Normal Data

Open **Terminal 1** — start the collector:
```bash
cd data_collector
sudo python3 collector.py --label normal
```

Open **Terminal 2** — run normal workload:
```bash
cd data_collector
bash simulate_normal.sh
```

After the simulation finishes, go back to Terminal 1 and press **Ctrl+C**.

### Step 2: Collect Attack Data

Open **Terminal 1** — start the collector again:
```bash
sudo python3 collector.py --label attack
```

Open **Terminal 2** — run attack simulation:
```bash
cd data_collector
sudo bash simulate_attacks.sh
```

After the simulation finishes, press **Ctrl+C** in Terminal 1.

Each collection run writes one `syscalls_<label>_<timestamp>.csv` (plus a small
`summary_<label>_<timestamp>.json`) into `collected_data/`. Repeat Step 1 and
Step 2 as many times as you like — more sessions give a more diverse dataset.

### Step 3: Train

No merge step is needed. The training script reads every
`collected_data/syscalls_*.csv` directly, applies the labels, filters
collection noise, and builds the windowed dataset itself:

```bash
# from the project root
python3 anomaly_detector/bosc_train.py \
    --global-window --window 200 --stride 50 --ngram 2 --tfidf --balance
```

---

## Output Format

### CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ns` | int | Kernel timestamp (nanoseconds) |
| `timestamp_human` | string | Human-readable timestamp |
| `pid` | int | Process ID |
| `ppid` | int | Parent Process ID |
| `tid` | int | Thread ID |
| `uid` | int | User ID |
| `gid` | int | Group ID |
| `comm` | string | Command name (COMM) |
| `syscall_nr` | int | Syscall number (x86_64) |
| `syscall_name` | string | Syscall name (e.g., `execve`) |
| `arg1`–`arg3` | int | First three syscall arguments |
| `return_value` | int | Syscall return value |
| `filename` | string | File path (for file-related syscalls) |
| `is_root` | int | 1 if UID=0, else 0 |
| `label` | int | **0 = normal, 1 = attack** |

### Summary JSON

```json
{
  "label": "normal",
  "duration_s": 30.5,
  "total_events": 1234,
  "events_per_sec": 40.46,
  "syscall_distribution": {
    "openat": 800,
    "execve": 50,
    "clone": 200,
    "setuid": 5,
    ...
  },
  "unique_pids": 45,
  "unique_uids": 3
}
```

---

## Monitored System Calls

| Syscall | Nr | Why It Matters |
|---------|----|----------------|
| `execve` / `execveat` | 59/322 | Process execution — key indicator of shell spawning |
| `setuid` / `setgid` | 105/106 | Direct privilege changes |
| `setreuid` / `setregid` | 113/114 | Real/effective UID/GID changes |
| `setresuid` / `setresgid` | 117/119 | All three UID/GID values |
| `openat` / `open` | 257/2 | File access — sensitive file reads |
| `clone` | 56 | Process/thread creation |
| `ptrace` | 101 | Process injection / debugging |
| `chmod` / `fchmod` | 90/91 | Permission modification |
| `chown` / `fchown` | 92/93 | Ownership changes |
| `mount` | 165 | Filesystem mounting |
| `mmap` | 9 | Memory mapping (exploit payloads) |
| `prctl` | 157 | Process control |
| `capset` | 125 | Capability changes |
| `unshare` | 272 | Namespace manipulation |

---

## How to Interpret the Output

### Normal Data Characteristics
- Dominated by `openat` (file reads) and `clone` (process creation)
- Few `setuid`/`setgid` calls
- File paths are common locations (`/usr/`, `/lib/`, `/tmp/`)
- Return values mostly 0 (success)

### Attack Data Characteristics
- **Elevated `setuid`/`setgid` frequency** — repeated rapid calls
- **`execve` of shells** (`/bin/sh`, `/bin/bash`) after `setuid`
- **Sensitive file access** — `/etc/shadow`, `/etc/sudoers`, `/proc/kallsyms`
- **SUID bit operations** — `chmod` with mode `4755`, `6755`
- **`ptrace` calls** — process attachment/injection
- **Return value `-1`** (EPERM) on privilege calls — failed escalation attempts

### Key Patterns to Look For
1. **setuid → execve chain**: Classic privilege escalation sequence
2. **Rapid setuid/setgid bursts**: >10 calls in short succession = anomalous
3. **Root UID (0) in non-root processes**: `is_root=1` unexpectedly
4. **Sensitive file opens**: `/etc/shadow`, `/etc/sudoers` from unusual processes

---

## Feeding into ML Pipeline (Part 2)

The collected `syscalls_*.csv` files feed directly into the anomaly detector
(`anomaly_detector/`), which builds Bag-of-System-Calls (BoSC) frequency
vectors over sliding windows and trains:
1. **Isolation Forest** — unsupervised novelty scoring (normal-only baseline)
2. **Random Forest** — supervised normal-vs-attack classification

See `anomaly_detector/bosc_train.py` (training), `bosc_evaluate.py` (metrics
and plots), and `bosc_realtime.py` (live scoring, also used by
`collector.py --score`).
