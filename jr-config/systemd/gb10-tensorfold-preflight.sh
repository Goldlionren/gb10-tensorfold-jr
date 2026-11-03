#!/usr/bin/env bash
set -euo pipefail

echo "[TensorFold] Waiting for Hindsight models..."

healthy=0

for ((i=1; i<=120; i++)); do
    healthy=1

    for c in gb10-embedding-production gb10-reranker-production; do
        status=$(docker inspect -f \
          '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' \
          "$c" 2>/dev/null || true)

        if [[ "$status" != "healthy" ]]; then
            healthy=0
        fi
    done

    [[ "$healthy" == 1 ]] && break
    sleep 5
done

if [[ "$healthy" != 1 ]]; then
    echo "ERROR: Hindsight models not healthy. Refusing to start."
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' \
    gb10-flashnext-production 2>/dev/null || true)" == "true" ]]; then
    echo "ERROR: SGLang is running. Refusing simultaneous model load."
    exit 1
fi

echo "[TensorFold] Hindsight healthy; SGLang inactive."
free -h
