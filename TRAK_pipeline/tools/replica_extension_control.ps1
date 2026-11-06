param(
 [Parameter(Mandatory=$true)][string]$Manifest,
 [ValidateSet('start','resume','stop','plan','summarize','refresh')][string]$Action='start'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Property($Object,[string]$Name) {
 if ($null -eq $Object) { return $null }
 $p=$Object.PSObject.Properties[$Name]
 if ($null -eq $p) { return $null }
 return $p.Value
}
function ResolvePath([string]$Value,[string]$Base) {
 if ([string]::IsNullOrWhiteSpace($Value)) { throw 'A configured path is empty.' }
 if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
 return [IO.Path]::GetFullPath((Join-Path $Base $Value))
}
$transcriptStarted=$false
try {
 $manifestFile=[IO.Path]::GetFullPath($Manifest)
 $batchRoot=Split-Path -Parent $manifestFile
 $manifestData=Get-Content -LiteralPath $manifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
 if (-not $manifestData.ready_for_execution -or -not $manifestData.release_relative) { throw 'This batch has not been frozen and marked ready. No cell started.' }
 $commonRoot=ResolvePath $manifestData.common_root_relative $batchRoot
 $configFile=ResolvePath $manifestData.datasets[0].config_relative $commonRoot
 $configBase=Split-Path -Parent $configFile
 $config=Get-Content -LiteralPath $configFile -Raw -Encoding UTF8 | ConvertFrom-Json
 $profile=Property (Property $config 'runtime_profiles') 'windows'
 if ($null -eq $profile) { throw 'The Windows runtime profile is missing.' }
 $direct=Property (Property $profile 'paths') 'science_python'
 if ($direct) { $sciencePython=ResolvePath $direct $configBase }
 else {
  $runtime=$null
  $override=Property $profile 'active_override'
  if ($override) {
   $overrideFile=ResolvePath $override $configBase
   if (Test-Path -LiteralPath $overrideFile -PathType Leaf) { $runtime=Get-Content -LiteralPath $overrideFile -Raw -Encoding UTF8 | ConvertFrom-Json }
  }
  $reference=ResolvePath (Property $profile 'reference_file') $configBase
  if ($null -eq $runtime) { $runtime=Get-Content -LiteralPath $reference -Raw -Encoding UTF8 | ConvertFrom-Json }
  $referenceBase=Property $profile 'reference_base'
  if ($referenceBase) { $runtimeBase=ResolvePath $referenceBase $configBase } else { $runtimeBase=Split-Path -Parent $reference }
  $sciencePython=ResolvePath (Property $runtime 'science_python') $runtimeBase
 }
 if (-not (Test-Path -LiteralPath $sciencePython -PathType Leaf)) { throw ('Existing Windows science Python is unavailable: '+$sciencePython+'. No package was installed.') }
 $releaseRoot=ResolvePath $manifestData.release_relative $commonRoot
 $runner=Join-Path $releaseRoot 'tools\run_replica_extension_batch.py'
 if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw ('Pinned batch controller is missing: '+$runner) }
 $cacheRoot=Join-Path $batchRoot 'runtime'
 $cacheMap=@{TEMP='tmp';TMP='tmp';TMPDIR='tmp';MPLCONFIGDIR='matplotlib';NUMBA_CACHE_DIR='numba';XDG_CACHE_HOME='xdg'}
 foreach ($entry in $cacheMap.GetEnumerator()) {
  $folder=Join-Path $cacheRoot $entry.Value
  [void](New-Item -ItemType Directory -Force -Path $folder)
  [Environment]::SetEnvironmentVariable($entry.Key,$folder,'Process')
 }
 $env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONNOUSERSITE='1';$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8';$env:MPLBACKEND='Agg'
 $logs=Join-Path $batchRoot 'logs';[void](New-Item -ItemType Directory -Force -Path $logs)
 $log=Join-Path $logs ('combined_'+$Action+'_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'_'+$PID+'.log')
 Start-Transcript -LiteralPath $log -Append | Out-Null;$transcriptStarted=$true
 Write-Host ('TRAK1 + TRAK2 | BR4/BR5 extension | '+$Action)
 Write-Host 'Only 156 new cells are queued. All 367 cells enter the five-replica summaries.'
 Write-Host 'Existing installed environments are reused; no cross-system replay, pilot or package installation.'
 Write-Host 'Control-C preserves committed checkpoints. STOP_AFTER_CURRENT_CELL stops the entire combined batch.'
 $pythonArguments=@('-B','-s','-u',$runner,'--manifest',$manifestFile,'--action',$Action)
 & $sciencePython @pythonArguments
 $batchExit=$LASTEXITCODE
 if ($null -eq $batchExit) { $batchExit=1 }
 exit $batchExit
} catch {
 Write-Host ('Replica extension batch stopped: '+$_.Exception.Message) -ForegroundColor Red
 exit 2
} finally {
 if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
