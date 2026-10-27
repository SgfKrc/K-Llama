<#
.SYNOPSIS
  Start a PC node as a QLH Route-A layer-stage worker (product path).

.DESCRIPTION
  Supersedes the older `build/perf/rog-start-worker.ps1`, which had two defects
  measured on 2026-10-11 (three-machine acceptance):

    1) it hard-coded an OBSOLETE cluster secret. Since ticket 9 the long-lived
       credential is owned by `LocalSecretStore` (node_config) and overrides
       `QLH_CLUSTER_SECRET` on import, so that script's HMAC registration always
       failed with "signature mismatch".
    2) it launched `src/inference_svc_main.py` -- a PyTorch *peer*. Peers are not
       v3 task workers, so the node never entered Route A's `eligible_node_ids`
       and the capacity solver reported "no usable PC nodes".

  This script instead starts `src/api_server.py` in client mode together with a
  cut-layer GGUF artifact and the keep-head shim -- the configuration that
  actually advertises `layer_ranges` and gets HELLO accepted.

  Get the current cluster secret on the MASTER with:

      python -c "import sys; sys.path.insert(0,'src'); import node_config as n; \
                 print(n.get_local_cluster_secret())"

  ASCII only: Windows PowerShell 5.1 mis-parses non-ASCII scripts without a BOM.

.EXAMPLE
  .\scripts\start_pc_layer_worker.ps1 -MasterHost 100.90.76.108 `
      -RepoRoot 'L:\repo' -LayerGguf 'L:\repo\artifacts\q35-2b-cut-16-20.gguf' `
      -KeepHeadShim 'L:\repo\build\keephead\build-cpu\bin\qlh_keep_head.dll' `
      -ClusterSecret '<from master>' -PreflightOnly
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$MasterHost,
  [string]$MasterTcpPort = '8888',
  [string]$NodeId        = $env:COMPUTERNAME,
  [string]$NodeType      = 'pc',
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [Parameter(Mandatory = $true)][string]$LayerGguf,
  [Parameter(Mandatory = $true)][string]$KeepHeadShim,
  [string]$ClusterSecret = $env:QLH_CLUSTER_SECRET,
  [string]$ApiPort       = '8001',
  [string]$ServerPort    = '8889',
  [string]$Python        = 'python',
  [switch]$PreflightOnly,
  # Added 2026-10-11 (measured). Starting a long-lived worker over ssh requires
  # detaching from the sshd job object, otherwise the whole process tree is killed
  # when ssh disconnects (symptom: ports never listen, log stops at "starting",
  # no error anywhere). -Detach re-launches this same script (without -Detach)
  # through Register-ScheduledTask, which is hosted by the Task Scheduler service.
  # Second trap: -Python must point at an interpreter that HAS llama_cpp
  # (e.g. <RepoRoot>\.venv-test\Scripts\python.exe); the host python lacks it and
  # the worker silently exits right after starting.
  [switch]$Detach
)

$ErrorActionPreference = 'Continue'

function Test-Preflight {
  $problems = New-Object System.Collections.Generic.List[string]
  $notes    = New-Object System.Collections.Generic.List[string]

  if (-not (Test-Path -LiteralPath $RepoRoot)) {
    $problems.Add("RepoRoot not found: $RepoRoot")
  } elseif (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'src\api_server.py'))) {
    $problems.Add("src\api_server.py not found under: $RepoRoot")
  }

  if (-not (Test-Path -LiteralPath $LayerGguf)) {
    $problems.Add("LayerGguf not found: $LayerGguf")
  } else {
    # The worker derives its advertised range from the sibling manifest; without
    # it the artifact is ignored (see scheduler_task_worker._configured_layer_artifact).
    $manifest = [System.IO.Path]::ChangeExtension($LayerGguf, '.manifest.json')
    if (-not (Test-Path -LiteralPath $manifest)) {
      $problems.Add("sibling manifest not found: $manifest")
    } else {
      try {
        $info = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
        $notes.Add("manifest: mode=$($info.mode) range=[$($info.source_layer_range -join ',')] id=$($info.source_model_id)")
        foreach ($field in @('artifact_sha256', 'source_model_sha256', 'hidden_size', 'tokenizer_sha256', 'source_model_id')) {
          if (-not $info.PSObject.Properties[$field]) {
            $notes.Add("WARN manifest lacks '$field' -> model_preflight may reject (descriptor_incomplete)")
          }
        }
      } catch {
        $problems.Add("manifest is not valid JSON: $manifest")
      }
    }
  }

  if (-not (Test-Path -LiteralPath $KeepHeadShim)) {
    $problems.Add("KeepHeadShim not found: $KeepHeadShim")
  }

  if ([string]::IsNullOrWhiteSpace($ClusterSecret)) {
    $problems.Add("ClusterSecret missing: pass -ClusterSecret or set QLH_CLUSTER_SECRET")
  }

  # llama.cpp stage execution needs EITHER the pip package OR the shim path.
  $hasPip = $false
  try {
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('llama_cpp') else 1)" 2>$null
    $hasPip = ($LASTEXITCODE -eq 0)
  } catch { $hasPip = $false }
  $notes.Add("llama_cpp pip package: $hasPip (shim path is also accepted by the worker)")

  foreach ($n in $notes)    { Write-Output "  note : $n" }
  foreach ($p in $problems) { Write-Error "  ERROR: $p" }
  return $problems.Count -eq 0
}

Write-Output "=== QLH PC layer worker preflight ==="
Write-Output "  repo   : $RepoRoot"
Write-Output "  master : ${MasterHost}:${MasterTcpPort} (http $ApiPort / tcp $ServerPort)"
Write-Output "  node   : $NodeId"
if (-not (Test-Preflight)) {
  Write-Error "preflight failed; not starting the worker"
  exit 2
}
Write-Output "preflight OK"

if ($PreflightOnly) { exit 0 }

$env:QLH_NODE_ROLE = 'client'
$env:QLH_NODE_ID = $NodeId
$env:QLH_NODE_TYPE = $NodeType
$env:QLH_CLIENT_MASTER_HOST = $MasterHost
$env:QLH_CLIENT_MASTER_PORT = $MasterTcpPort
$env:QLH_CLUSTER_SECRET = $ClusterSecret
$env:QLH_API_PORT = $ApiPort
$env:QLH_SERVER_PORT = $ServerPort
$env:QLH_TASK_WORKER_EXPERIMENTAL_ENABLED = '1'
$env:QLH_TASK_GRAPH_ENABLED = '1'
$env:QLH_LAYER_GGUF = $LayerGguf
$env:QLH_KEEP_HEAD_SHIM = $KeepHeadShim
# Optional, default OFF (PERF-SKIP-LOGITS-01): a middle layer segment only forwards hidden
# states, so it does not need logits at all. Exporting QLH_LAYER_SKIP_LOGITS=1 makes the
# keep-head shim leave batch.logits at 0 for the hidden-only entry points, so llama.cpp
# skips the output.weight matmul (measured 122.6 -> 102.6 ms/step on the 9B [4,24) segment;
# hidden output bit-identical, tail argmax unchanged). It is inherited by this process as-is.
# NOTE: -Detach starts a Scheduled Task, which does NOT inherit this shell's environment;
# set the variable inside that task if the detached worker should skip logits too.

Set-Location -LiteralPath $RepoRoot

if ($Detach) {
  $inner = "-ExecutionPolicy Bypass -NoProfile -File `"$PSCommandPath`" " +
           "-MasterHost $MasterHost -MasterTcpPort $MasterTcpPort -NodeId `"$NodeId`" -NodeType $NodeType " +
           "-RepoRoot `"$RepoRoot`" -LayerGguf `"$LayerGguf`" -KeepHeadShim `"$KeepHeadShim`" " +
           "-ClusterSecret `"$ClusterSecret`" -ApiPort $ApiPort -ServerPort $ServerPort -Python `"$Python`""
  $taskName = "qlh_layer_worker_$NodeId"
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $inner -WorkingDirectory $RepoRoot
  Register-ScheduledTask -TaskName $taskName -Action $action -Force | Out-Null
  Start-ScheduledTask -TaskName $taskName
  Write-Output "=== detached: scheduled task '$taskName' started ==="
  Write-Output "  watch: Get-ScheduledTaskInfo -TaskName $taskName ; Get-NetTCPConnection -State Listen"
  exit 0
}

Write-Output "=== starting api_server.py (client mode) ==="
& $Python 'src\api_server.py'
