#!/bin/bash
# Normal Workload Simulation Script for eBPF Data Collection
#
# Loops normal workload continuously for a target duration so the
# collector captures enough data for training.
#
# Run in a SEPARATE terminal while collector.py --label normal runs.
#
#   bash simulate_normal.sh --duration 600  # runs for 10 minutes

DURATION=300
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

TMPDIR_SIM=$(mktemp -d /tmp/pe_normal_sim.XXXXXX)
trap "rm -rf $TMPDIR_SIM" EXIT


echo "  Normal Workload Simulator"
echo "  Target duration : ${DURATION}s"


echo

sleep 2  # giving collector time to attach

START_TIME=$(date +%s)
CYCLE=0

run_one_cycle() {
    CYCLE=$((CYCLE + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "[cycle $CYCLE | ${ELAPSED}s elapsed]"

    # 1. Regular File Operations 
    ls -la /tmp/       > /dev/null 2>&1
    ls -la /home/      > /dev/null 2>&1
    ls -la /var/log/   > /dev/null 2>&1
    cat /etc/hostname  2>/dev/null > /dev/null
    cat /etc/os-release 2>/dev/null > /dev/null
    cat /etc/timezone  2>/dev/null > /dev/null || true
    wc -l /etc/passwd  2>/dev/null > /dev/null

    # 2. Standard Process Creation
    echo "Hello World" | grep "Hello" > /dev/null
    date    > /dev/null
    whoami  > /dev/null
    uname -a > /dev/null
    id      > /dev/null
    ps aux  > /dev/null 2>&1
    uptime  > /dev/null

    # 3. File Read/Write 
    for i in $(seq 1 10); do
        echo "This is test line $i" > "$TMPDIR_SIM/testfile_$i.txt"
    done
    cat "$TMPDIR_SIM"/testfile_*.txt > "$TMPDIR_SIM/combined.txt"
    wc -l "$TMPDIR_SIM/combined.txt"  > /dev/null
    grep "test line 5" "$TMPDIR_SIM/combined.txt" > /dev/null
    sort "$TMPDIR_SIM/combined.txt"   > "$TMPDIR_SIM/sorted.txt"
    rm  "$TMPDIR_SIM"/testfile_*.txt

    # 4. Directory Traversal
    find /usr/bin -maxdepth 1 -name "*.sh" 2>/dev/null | head -5 > /dev/null
    find /etc     -maxdepth 1 -type f      2>/dev/null | head -10 > /dev/null
    du -sh /tmp   2>/dev/null > /dev/null

    # 5. Normal Process Fork/Wait 
    for i in $(seq 1 5); do
        (sleep 0.05 && echo "subprocess $i" > /dev/null) &
    done
    wait

    # Pipe chain
    cat /etc/passwd 2>/dev/null | grep "root" | wc -l > /dev/null

    # 6. Standard System Queries 
    df -h   > /dev/null 2>&1
    free -m > /dev/null 2>&1
    mount   2>/dev/null | head -3 > /dev/null
    env     > /dev/null 2>&1
    stat /etc/passwd > /dev/null 2>&1

    # 7. Extra variety: network + archive operations 
    cat /etc/resolv.conf 2>/dev/null > /dev/null || true
    cat /proc/net/dev    2>/dev/null | head -5 > /dev/null || true
    tar -czf "$TMPDIR_SIM/archive.tar.gz" /etc/hosts /etc/hostname 2>/dev/null || true
    rm -f "$TMPDIR_SIM/archive.tar.gz"

    # 8. Python / scripting workload 
    python3 -c "import math, os; [math.sqrt(i) for i in range(1000)]" 2>/dev/null || true
}

# main loop 
while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -ge "$DURATION" ]; then
        break
    fi
    run_one_cycle
done

echo

echo "  Normal simulation complete."
echo "  Ran $CYCLE cycle(s) over ${DURATION}s."

