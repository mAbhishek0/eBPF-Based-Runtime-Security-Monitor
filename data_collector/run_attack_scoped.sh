#!/bin/bash
# run_attack_scoped.sh — launch the attack simulator inside a dedicated
# cgroup so the interleaved-collection labeler can cleanly separate attack
# events from your normal desktop activity.
#
# Why: the collector traces the whole machine. During interleaved
# collection you use the desktop normally WHILE this runs the 14 attack
# techniques. To label afterward, we put every attack process in its own
# cgroup (a transient systemd scope); the collector already stamps each
# event with its cgroup id, so an event is "attack" iff its cgroup id
# matches this scope's. Normal desktop activity lives in other cgroups.
#
set -u

DURATION="${1:-300}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SCOPE="pe-attack-$(date +%s)"

export PE_TIMELINE="${PE_TIMELINE:-/tmp/pe_technique_timeline.tsv}"
export PE_CGROUP_MARKER="${PE_CGROUP_MARKER:-/tmp/pe_attack_cgroup.txt}"

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] must run as root (attacks need root, and a system scope needs root):"
    echo "        sudo bash run_attack_scoped.sh $DURATION"
    exit 1
fi

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "[ERROR] systemd-run not found. Either install systemd, or run the"
    echo "        simulator directly and pass its cgroup id to the labeler:"
    echo "        sudo bash simulate_attacks.sh --duration $DURATION"
    exit 1
fi


echo "  Scoped attack run"
echo "  Scope unit : ${SCOPE}.scope"
echo "  Duration   : ${DURATION}s"
echo "  Timeline   : $PE_TIMELINE"
echo "  Cgroup mark: $PE_CGROUP_MARKER"


# --scope runs the command as a direct child in a new transient cgroup and
# inherits our environment (so PE_TIMELINE / PE_CGROUP_MARKER pass through).
# --collect garbage-collects the unit once it exits.
exec systemd-run --scope --unit="$SCOPE" --collect \
    --setenv=PE_TIMELINE="$PE_TIMELINE" \
    --setenv=PE_CGROUP_MARKER="$PE_CGROUP_MARKER" \
    bash "$DIR/simulate_attacks.sh" --duration "$DURATION"
