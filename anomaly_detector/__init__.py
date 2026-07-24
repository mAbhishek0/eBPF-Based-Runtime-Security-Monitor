"""

Implementing the Bag-of-System-Calls (BoSC) method for detecting privilege escalation and other anomalous behaviours from eBPF-collected
system call traces.

Modules

bosc_realtime  : RealtimeScorer, global or per-PID sliding window scorer
bosc_train     : Feature engineering, model training, persistence
bosc_evaluate  : Offline evaluation, metrics, and plots
bosc_report    : HTML report generator
"""
