#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'Usage: entrypoint.sh <command>

Start the retrieval server.

Commands:
    server      Start a server to perform similarity search on the CLIP model.

Options:
    --host <host>               Host to bind to (default: localhost)
    --port <port>               Port to bind to (default: 1234)
    --clip_model <clip_model>   CLIP model to use (default: ViT-L/14)
'
    exit
fi

cd "$(dirname "$0")"

# set default values
HOST='0.0.0.0'
PORT='1234'
CLIP_MODEL='ViT-L/14'

# parse command
if [[ $# -eq 0 ]]; then
    echo "Error: no command specified" >&2
    exit 1
fi
COMMAND="$1"
shift

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2" ;;
        --port) PORT="$2" ;;
        --clip_model) CLIP_MODEL="$2" ;;
        *) echo "Error: unknown option: $1" >&2; exit 1 ;;
    esac
    shift 2
done

main() {
    if [[ "${COMMAND}" == "server" ]]; then
        python3 src/download.py
        python3 src/server.py \
            --indices_paths /app/artifacts/models/retrieval/indices.json \
            --host "$HOST" \
            --port "$PORT" \
            --clip_model "$CLIP_MODEL" \
            --enable_faiss_memory_mapping \
            --use_arrow
    fi
}

# activate conda environment
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate env

main "$@"
