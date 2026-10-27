<#
.SYNOPSIS
  Verify explicit journal recovery files are in sync per node.

.DESCRIPTION
  This script prints the local SHA256 of the journal recovery dependencies and,
  when -Remote is given (an ssh alias), compares it against that host's copy.
  The probe script is fed to the remote interpreter over stdin, so nothing is
  written on the remote side.

.EXAMPLE
  powershell -File scripts\verify_journal_replication_sync.ps1

.EXAMPLE
  powershell -File scripts\verify_journal_replication_sync.ps1 -Remote y700-ip `
      -RemoteRoot '~/qlh-jr/src' -RemotePython python3

.EXAMPLE
  powershell -File scripts\verify_journal_replication_sync.ps1 -Remote 'ROG@100.116.17.12' `
      -RemoteRoot 'L:\repo\src' -RemotePython python -RemoteShell windows

  ASCII only: Windows PowerShell 5.1 mis-parses non-ASCII scripts without a BOM.
#>
[CmdletBinding()]
param(
  [string]$Remote = '',
  [string]$RemoteRoot = '',
  [string]$RemotePython = 'python3',
  [ValidateSet('posix', 'windows')][string]$RemoteShell = 'posix',
  [string]$LocalRoot = ''
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
if (-not $LocalRoot) { $LocalRoot = Join-Path $repo 'src' }
if (-not (Test-Path $LocalRoot)) { throw "local src directory not found: $LocalRoot" }

$groups = [ordered]@{
  'journal-recovery' = @(
    'journal_replication.py',
    'journal_replication_transport.py',
    'journal_replication_service.py',
    'recovery_journal.py'
  )
}

function Get-LocalHash([string]$path) {
  if (-not (Test-Path $path)) { return 'MISSING' }
  return (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLower()
}

function Get-RemoteHashes([string]$host_, [string]$root, [string]$python, [string]$shell, [string[]]$names) {
  $probe = @'
import hashlib, sys
for name in sys.argv[1:]:
    try:
        with open(name, "rb") as handle:
            print(name, hashlib.sha256(handle.read()).hexdigest())
    except OSError:
        print(name, "MISSING")
'@
  $quoted = ($names | ForEach-Object { '"' + $_ + '"' }) -join ' '
  # posix: a quoted '~/x' does NOT expand, so route it through $HOME.
  if ($shell -eq 'windows') {
    $cd = 'cd /d "{0}"' -f $root
  } elseif ($root.StartsWith('~')) {
    $cd = 'cd "$HOME{0}"' -f $root.Substring(1)
  } else {
    $cd = 'cd "{0}"' -f $root
  }
  $command = '{0} && {1} - {2}' -f $cd, $python, $quoted
  $output = ($probe -split "`n" | Out-String) | & ssh -o BatchMode=yes -o ConnectTimeout=20 $host_ $command 2>&1
  $hashes = @{}
  foreach ($line in $output) {
    $parts = ($line -as [string]).Trim() -split '\s+'
    if ($parts.Count -ge 2 -and $parts[1] -match '^[0-9a-fA-F]{64}$|^MISSING$') {
      $hashes[$parts[0]] = $parts[1].ToLower()
    }
  }
  return $hashes
}

$remoteHashes = @{}
if ($Remote) {
  if (-not $RemoteRoot) { throw '-RemoteRoot is required when -Remote is given' }
  $names = @()
  foreach ($group in $groups.Values) { $names += $group }
  $remoteHashes = Get-RemoteHashes -host_ $Remote -root $RemoteRoot -python $RemotePython -shell $RemoteShell -names $names
  if ($remoteHashes.Count -eq 0) {
    Write-Output "warn: no hashes parsed from $Remote (check -RemoteShell/-RemotePython)"
  }
}

$mismatches = 0
foreach ($groupName in $groups.Keys) {
  Write-Output ("== {0} ==" -f $groupName)
  foreach ($name in $groups[$groupName]) {
    $local = Get-LocalHash (Join-Path $LocalRoot $name)
    $localShort = if ($local -eq 'MISSING') { 'MISSING' } else { $local.Substring(0, 16) }
    if (-not $Remote) {
      Write-Output ("  local  {0}  {1}" -f $localShort, $name)
      continue
    }
    # NOTE: do not name this $remote -- PowerShell variable names are
    # case-insensitive, so it would clobber the -Remote parameter.
    $remoteHash = $remoteHashes[$name]
    if (-not $remoteHash) { $remoteHash = 'UNKNOWN' }
    $status = 'OK'
    if ($local -ne $remoteHash) { $status = 'SKEW'; $mismatches++ }
    $remoteShort = if ($remoteHash -eq 'MISSING' -or $remoteHash -eq 'UNKNOWN') { $remoteHash } else { $remoteHash.Substring(0, 16) }
    Write-Output ("  {0,-5} {1,-16} {2,-16} {3}" -f $status, $localShort, $remoteShort, $name)
  }
}

Write-Output ""
if ($Remote -and $mismatches -gt 0) {
  Write-Output ("FAIL: {0} file(s) differ from {1} -- sync ALL of them before restarting anything" -f $mismatches, $Remote)
  exit 1
}
Write-Output "done"
