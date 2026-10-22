# Build a job-local environment from uv.lock and delete it when this shell exits.
# Source this file, then call create_job_venv with the repository root.
# The EXIT/INT/TERM trap removes the environment after success and after failure.

create_job_venv() {
    local repo_root="$1"
    local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local-$$}}"
    local task_id="${SLURM_ARRAY_TASK_ID:-0}"

    # Global so the trap can see it. One directory per Slurm task, so concurrent
    # array tasks do not share or delete each other's environment.
    JOB_VENV="${repo_root}/.venv-${job_id}_${task_id}"

    cleanup_job_venv() {
        local status=$?
        trap - EXIT INT TERM
        if [[ -n "${JOB_VENV:-}" && -d "$JOB_VENV" ]]; then
            echo "Removing job environment ${JOB_VENV}"
            rm -rf "$JOB_VENV" || true
        fi
        exit "$status"
    }
    trap cleanup_job_venv EXIT INT TERM

    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not on PATH. Install it on the cluster before submitting." >&2
        exit 1
    fi

    echo "Creating job environment ${JOB_VENV}"
    # UV_PROJECT_ENVIRONMENT keeps this off the shared .venv and off the default cluster env.
    # --frozen installs uv.lock as-is and does not rewrite it.
    UV_PROJECT_ENVIRONMENT="$JOB_VENV" uv sync --frozen --directory "$repo_root"
    # shellcheck disable=SC1091
    source "${JOB_VENV}/bin/activate"
}
