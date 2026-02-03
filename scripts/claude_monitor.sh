#!/bin/bash
# Claude CLI Monitor Script
# Ensures claude --dangerously-skip-permissions -c is always running
# If it stops, restarts with the specified prompt

PROMPT="make sure you are done and everything is well tested and you are adhering to the tasks listed out and it's ready to be used in production"
LOG_FILE="/lambda/nfs/arizona/pi-openpi/logs/claude_monitor.log"
PID_FILE="/lambda/nfs/arizona/pi-openpi/logs/claude_monitor.pid"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cleanup() {
    log "Monitor script stopping..."
    rm -f "$PID_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Store our PID
echo $$ > "$PID_FILE"

log "Claude monitor started (PID: $$)"

while true; do
    # Check if claude with dangerously-skip-permissions is running
    if ! pgrep -f "claude.*--dangerously-skip-permissions" > /dev/null 2>&1; then
        log "Claude CLI not running. Starting..."

        cd /lambda/nfs/arizona/pi-openpi

        # Start claude in background with the prompt
        echo "$PROMPT" | claude --dangerously-skip-permissions -c &
        CLAUDE_PID=$!

        log "Started Claude CLI (PID: $CLAUDE_PID)"

        # Wait for it to finish
        wait $CLAUDE_PID
        EXIT_CODE=$?

        log "Claude CLI exited with code $EXIT_CODE"
    else
        log "Claude CLI is running"
    fi

    # Check every 30 seconds
    sleep 30
done
