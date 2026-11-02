param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z][A-Za-z0-9_-]*$')][string]$Molecule,
    [Parameter(Mandatory=$true)][ValidateSet('start','resume','plan','refresh','summarize','stop')][string]$Action,
    [string]$Cell = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-Property($Object,[string]$Name) {
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}
function Resolve-ConfiguredPath([string]$Value,[string]$Base) {
    if ([string]::IsNullOrWhiteSpace($Value)) { throw 'A configured runtime path is empty.' }
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Base $Value))
}

try {
    $commonRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $configFile = Join-Path $commonRoot ('configs\' + $Molecule + '.json')
    if (-not (Test-Path -LiteralPath $configFile -PathType Leaf)) { throw ('Dataset configuration is missing: '+$configFile) }
    $config = Get-Content -LiteralPath $configFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $configBase = Split-Path -Parent $configFile
    $dataRoot = Resolve-ConfiguredPath $config.data_root $configBase
    if (-not (Test-Path -LiteralPath $dataRoot -PathType Container)) { throw ('Data drive/root is unavailable: '+$dataRoot) }
    $controlRelative = Get-Property $config 'control_relative'
    if ([string]::IsNullOrEmpty($controlRelative)) { $controlRelative = '_batch_pipeline' }
    $controlRoot = Join-Path $dataRoot $controlRelative
    $profile = Get-Property (Get-Property $config 'runtime_profiles') 'windows'
    if ($null -eq $profile) { throw 'The dataset has no Windows runtime profile.' }
    $directScience = Get-Property (Get-Property $profile 'paths') 'science_python'
    if ($directScience) {
        $sciencePython = Resolve-ConfiguredPath $directScience $configBase
    } else {
        $runtime = $null
        $activeOverride = Get-Property $profile 'active_override'
        if ($activeOverride) {
            $overrideFile = Resolve-ConfiguredPath $activeOverride $configBase
            if (Test-Path -LiteralPath $overrideFile -PathType Leaf) {
                $runtime = Get-Content -LiteralPath $overrideFile -Raw -Encoding UTF8 | ConvertFrom-Json
            }
        }
        if ($null -eq $runtime) {
            $referenceFile = Resolve-ConfiguredPath (Get-Property $profile 'reference_file') $configBase
            $runtime = Get-Content -LiteralPath $referenceFile -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        $baseValue = Get-Property $profile 'reference_base'
        if ($baseValue) { $runtimeBase = Resolve-ConfiguredPath $baseValue $configBase }
        else { $runtimeBase = $controlRoot }
        $sciencePython = Resolve-ConfiguredPath (Get-Property $runtime 'science_python') $runtimeBase
    }
    if (-not (Test-Path -LiteralPath $sciencePython -PathType Leaf)) {
        throw ('Existing science Python is unavailable. Use the preserved SETUP_WINDOWS.cmd to repair the environment; no packages were installed: '+$sciencePython)
    }
    # Configure controlled early import caches on the data drive. Full Java and
    # SciJava configuration remains in the common runtime module.
    $cacheRoot = Join-Path $controlRoot 'modular\runtime'
    $cacheMap = @{TEMP='tmp';TMP='tmp';TMPDIR='tmp';MPLCONFIGDIR='matplotlib';NUMBA_CACHE_DIR='numba'}
    foreach ($entry in $cacheMap.GetEnumerator()) {
        $folder = Join-Path $cacheRoot $entry.Value
        [void](New-Item -ItemType Directory -Force -Path $folder)
        [Environment]::SetEnvironmentVariable($entry.Key,$folder,'Process')
    }
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:MPLBACKEND = 'Agg'
    $command = switch ($Action) {
        'start' { 'run' }; 'resume' { 'run' }; 'plan' { 'plan' }
        'refresh' { 'refresh' }; 'summarize' { 'summarize' }; 'stop' { 'stop' }
    }
    $arguments = @('-B','-s','-u',(Join-Path $commonRoot 'tools\launch.py'),$command,'--config',$configFile)
    if ($Action -eq 'resume') { $arguments += '--resume' }
    if ($Action -in @('refresh','summarize')) { $arguments += '--completed-only' }
    if ($Cell) { $arguments += @('--cell',$Cell) }
    Write-Host ('Common protein pipeline | '+$Molecule+' | '+$Action)
    Write-Host 'Existing software environments are reused. No cross-platform numerical/pixel replay or pilot.'
    & $sciencePython @arguments
    $pipelineExit = $LASTEXITCODE
    if ($null -eq $pipelineExit) { $pipelineExit = 1 }
    exit $pipelineExit
} catch {
    Write-Host ('Pipeline launcher stopped: '+$_.Exception.Message) -ForegroundColor Red
    exit 2
}
