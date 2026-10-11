<#
.SYNOPSIS
  Start (or dry-run) a journal replica receiver on this node.

.DESCRIPTION
  #93's quorum barrier only counts replicas that are actually running and
  durable, so the receiver must be startable from an operator script -- not only
  from an ad-hoc Python snippet.

  The receiver itself is scripts/start_journal_replica_receiver.py; this wrapper
  exists for Windows deployments (find python, check the secret is present in the
  environment, start detached, write a log and a pid file).

  By default this does NOT touch Task Scheduler: making the receiver survive a
  reboot is an explicit operator decision (see docs/journal复制部署检查单-*.md).
  Use -Foreground to run attached, or -PrintPlan for a secret-safe dry-run.

  ASCII only: Windows PowerShell 5.1 mis-parses non-ASCII scripts without a BOM.

.EXAMPLE
  $env:QLH_JOURNAL_REPLICATION_SECRET = '<40+ byte secret>'
  powershell -File scripts\start_journal_replica_receiver.ps1 `
      -Replica D:\qlh\jr\node.sqlite3 -Port 9200 -SourceNodeId node-leader -PrintPlan

.EXAMPLE
  powershell -File scripts\start_journal_replica_receiver.ps1 `
      -Replica D:\qlh\jr\node.sqlite3 -Port 9200 -SourceNodeId node-leader
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Replica,
  [Parameter(Mandatory = $true)][int]$Port,
  [Parameter(Mandatory = $true)][string]$SourceNodeId,
  [string]$BindHost = '0.0.0.0',
  [string]$StreamId = 'task-graph-journal',
  [string]$SecretEnv = 'QLH_JOURNAL_REPLICATION_SECRET',
  [string]$Python = '',
  [string]$LogPath = '',
  [string]$PidFile = '',
  [switch]$PrintPlan,
  [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$runner = Join-Path $here 'start_journal_replica_receiver.py'
if (-not (Test-Path $runner)) { throw "missing runner: $runner" }

if (-not $Python) {
  $candidates = @(
    (Join-Path $repo '.venv\Scripts\python.exe'),
    (Join-Path $repo '.venv-test\Scripts\python.exe')
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) { $Python = $candidate; break }
  }
  if (-not $Python) { $Python = 'python' }
}

# The secret must come from the environment: command lines leak into process
# listings, and the receiver refuses short secrets anyway.
$secretValue = [System.Environment]::GetEnvironmentVariable($SecretEnv)
if ([string]::IsNullOrWhiteSpace($secretValue)) {
  throw "$SecretEnv is not set; refusing to start a receiver without a shared secret"
}

$pyArgs = @(
  $runner,
  '--replica', $Replica,
  '--port', $Port,
  '--host', $BindHost,
  '--source-node-id', $SourceNodeId,
  '--stream-id', $StreamId,
  '--secret-env', $SecretEnv
)
if ($PrintPlan) { $pyArgs += '--print-plan' }
if ($PidFile) { $pyArgs += @('--pid-file', $PidFile) }

if ($PrintPlan -or $Foreground) {
  & $Python @pyArgs
  exit $LASTEXITCODE
}

if (-not $LogPath) {
  $LogPath = Join-Path $repo ('build\journal-receiver-{0}.log' -f $SourceNodeId)
}
$logDir = Split-Path -Parent $LogPath
if ($logDir -and -not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }

$process = Start-Process -FilePath $Python -ArgumentList $pyArgs -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $LogPath -RedirectStandardError ($LogPath + '.err')
Write-Output ('started journal replica receiver pid={0} log={1}' -f $process.Id, $LogPath)
if ($PidFile) { Set-Content -Path $PidFile -Value $process.Id -Encoding ascii }
