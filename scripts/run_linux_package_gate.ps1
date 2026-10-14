param(
    [string]$Image = "python:3.12-slim-bookworm"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

try {
    docker version --format '{{.Server.Version}}' | Out-Null
} catch {
    throw "docker_daemon_unavailable: start Docker Desktop and retry"
}

docker pull $Image
if ($LASTEXITCODE -ne 0) {
    throw "docker_image_unavailable: $Image"
}

docker run --rm `
    --network none `
    --read-only `
    --user 65532:65532 `
    --env HOME=/home/qlh-clean `
    --env XDG_CONFIG_HOME=/home/qlh-clean/.config `
    --env XDG_DATA_HOME=/home/qlh-clean/.local/share `
    --env XDG_STATE_HOME=/home/qlh-clean/.local/state `
    --env PYTHONDONTWRITEBYTECODE=1 `
    --tmpfs /tmp:rw,exec,nosuid,size=256m,uid=65532,gid=65532 `
    --tmpfs /home/qlh-clean:rw,nosuid,size=256m,uid=65532,gid=65532 `
    --mount "type=bind,source=$repositoryRoot,target=/workspace,readonly" `
    --workdir /workspace `
    $Image `
    python scripts/package_contract/linux_clean_gate.py

if ($LASTEXITCODE -ne 0) {
    throw "linux_clean_gate_failed: exit=$LASTEXITCODE"
}
