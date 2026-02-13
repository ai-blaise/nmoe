#!/bin/bash
# Wrap every torchrun rank with compute-sanitizer memcheck
# torchrun sets LOCAL_RANK env var
RANK_ID="${LOCAL_RANK:-0}"
exec compute-sanitizer --tool memcheck --log-file "/tmp/sanitizer_rank${RANK_ID}.log" \
    /home/nourdine/nmoe/.venv/bin/python "$@"
