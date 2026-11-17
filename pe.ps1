<#
.SYNOPSIS
    Part Estimator developer entry point: setup, run, test.

.DESCRIPTION
    One script for everything. Run `.\pe.ps1 help` for the command list.

    Environment: a .venv INSIDE this folder, holding the exact pins from
    requirements.txt (wheels only, so no package runs its own setup code), plus
    an editable install of this package so code changes take effect at once and
    `part-estimator` works with the venv active. After setup nothing makes a
    network connection.

    Nothing is written outside this folder: pip's cache lives in .\scratch and
    is deleted after each install; user site-packages are ignored.

    Supported: CPython 3.11 - 3.13, 64-bit. 3.12 is preferred when several are
    installed. Override with $env:PE_PYTHON or a python-path.txt beside this
    script holding the full path to python.exe.

    Works in Windows PowerShell 5.1 and PowerShell 7+.
#>

# No param() block on purpose: every argument arrives in $args untouched, so
# Python options such as -o, -q or -k pass straight through to Python instead
# of being prefix-matched against PowerShell parameters.

$ErrorActionPreference = 'Continue'   # native-command stderr must not throw in 5.1
Set-StrictMode -Version 2.0

$ToolDir   = $PSScriptRoot
$Venv      = Join-Path $ToolDir '.venv'
$VPy       = Join-Path $Venv 'Scripts\python.exe'
$VPyW      = Join-Path $Venv 'Scripts\pythonw.exe'
$EntryExe  = Join-Path $Venv 'Scripts\part-estimator.exe'
$PytestExe = Join-Path $Venv 'Scripts\pytest.exe'
$Req       = Join-Path $ToolDir 'requirements.txt'
$ReqDev    = Join-Path $ToolDir 'requirements-dev.txt'
$Lock      = Join-Path $ToolDir 'installed-lock.txt'
$Stamp     = Join-Path $Venv 'requirements.sha256'
$Scratch   = Join-Path $ToolDir 'scratch'

# Probe order: the primary target first.
$SupportedVersions = @('3.12', '3.13', '3.11')
$VersionCheck = 'import sys; ok = (3,11) <= sys.version_info < (3,14) and sys.maxsize > 2**32; print(sys.executable if ok else ""); raise SystemExit(0 if ok else 1)'

# Applied for the duration of this script only, then restored, so the caller's
# session is left exactly as it was.
$ToolEnv = [ordered]@{
    PIP_CACHE_DIR                 = (Join-Path $Scratch 'pip-cache')
    PIP_DISABLE_PIP_VERSION_CHECK = '1'
    PIP_NO_INPUT                  = '1'
    PYTHONNOUSERSITE              = '1'
    QT_ENABLE_HIGHDPI_SCALING     = '1'
}

$Usage = @'
Part Estimator - developer entry point

USAGE
  .\pe.ps1 <command> [arguments]

APP
  gui [file.step] [-Wait]     Open the window. Detached by default; -Wait keeps
                              it in this console and streams the log.
  classify <step...>          Sheet metal or machined? Options: --json, --cutoff IN
  sheet <step> [options]      Sheet metal flat pattern, DXF, STEP and report.
                              `.\pe.ps1 sheet --help` lists every option.
  calibrate <csv>             Fit machining calibration factors to measured times.

CHECKS
  test [pytest args]          Fast tests: timing model and CLI (pytest, seconds).
  test -Full                  Everything: machining self-test plus the full suite
                              with the CAD kernel and Qt (about 300 checks).

ENVIRONMENT
  setup [-Dev] [-Force]       Create .venv here and install the pinned packages.
                              -Dev adds pytest. -Force rebuilds from scratch.
  reset                       Delete .venv.
  py [args]                   Run the venv's Python, e.g. `.\pe.ps1 py -m pip list`.
  help                        This text.

Every app and check command sets up .venv first if it is missing or
requirements.txt has changed. Exit codes come from the underlying command.

To use the tools directly instead:  .\.venv\Scripts\Activate.ps1
                                    part-estimator --help
'@

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$Message) {
    Write-Host "  $Message" -ForegroundColor Cyan
}

function Write-Fail([string]$Message) {
    [Console]::Error.WriteLine("pe: $Message")
}

# Pull PowerShell-side switches (e.g. -Dev) out of the argument list; return
# the found switch names and everything else, untouched and in order.
function Split-Switches([object[]]$ArgList, [string[]]$Names) {
    $found = @{}
    $rest = New-Object System.Collections.Generic.List[string]
    foreach ($a in $ArgList) {
        $s = [string]$a
        $hit = $null
        foreach ($n in $Names) {
            if ($s -ieq "-$n") { $hit = $n }
        }
        if ($hit) { $found[$hit] = $true } else { $rest.Add($s) }
    }
    return @{ Switches = $found; Rest = [string[]]$rest.ToArray() }
}

function Get-RequirementsHash {
    return (Get-FileHash -LiteralPath $Req -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-InstalledHash {
    if (-not (Test-Path -LiteralPath $Stamp)) { return '' }
    $line = Get-Content -LiteralPath $Stamp -TotalCount 1
    if ($null -eq $line) { return '' }
    return ([string]$line).Trim().ToLowerInvariant()
}

# Returns the path to a usable python.exe, or $null.
function Find-Python {
    if ($env:PE_PYTHON -and (Test-Path -LiteralPath $env:PE_PYTHON)) {
        return $env:PE_PYTHON
    }
    $hint = Join-Path $ToolDir 'python-path.txt'
    if (Test-Path -LiteralPath $hint) {
        foreach ($line in Get-Content -LiteralPath $hint) {
            $p = $line.Trim().Trim('"')
            if ($p -and (Test-Path -LiteralPath $p)) { return $p }
        }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in $SupportedVersions) {
            $exe = & py "-$v" -c $VersionCheck 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return ([string]$exe).Trim() }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $exe = & python -c $VersionCheck 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return ([string]$exe).Trim() }
    }
    return $null
}

function Test-EnvReady {
    if (-not (Test-Path -LiteralPath $VPy)) { return $false }
    if ((Get-InstalledHash) -ne (Get-RequirementsHash)) { return $false }
    return (Test-Path -LiteralPath $EntryExe)
}

function Invoke-Pip([string[]]$PipArgs) {
    & $VPy -m pip @PipArgs | Out-Host
    return ($LASTEXITCODE -eq 0)
}

# Create or update .venv. Returns $true on success.
function Install-Environment([bool]$Dev = $false, [bool]$Force = $false) {
    if (-not (Test-Path -LiteralPath $Req)) {
        Write-Fail 'requirements.txt is missing. pe.ps1 must stay in the project folder.'
        return $false
    }
    if ($Force -and (Test-Path -LiteralPath $Venv)) {
        if (-not (Remove-Environment)) { return $false }
    }

    $fresh = $false
    if (-not (Test-Path -LiteralPath $VPy)) {
        Write-Host ''
        Write-Step 'Creating .venv in this folder. The first install needs internet once'
        Write-Step 'and takes several minutes (the CAD kernel and 3D viewer are large).'
        $base = Find-Python
        if (-not $base) {
            Write-Fail 'No suitable Python found. Install CPython 3.12 64-bit (3.11 and 3.13 also work).'
            Write-Fail 'If it is installed but not found, set $env:PE_PYTHON or create python-path.txt'
            Write-Fail 'here containing the full path to python.exe.'
            return $false
        }
        Write-Step "Using $base"
        & $base -m venv $Venv | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VPy)) {
            Write-Fail 'Could not create .venv. The Python install may be missing the venv module.'
            return $false
        }
        $fresh = $true
    }

    & $VPy -c 'import sys' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Fail '.venv exists but does not run; the Python it was built from has probably moved.'
        Write-Fail 'Run `.\pe.ps1 setup -Force` to rebuild it.'
        return $false
    }

    $want = Get-RequirementsHash
    $installedRuntime = $false
    if ($fresh -or (Get-InstalledHash) -ne $want) {
        if (-not $fresh) {
            Write-Step 'requirements.txt has changed since the last install. Updating...'
            # The old lock describes the old requirements; it must not win.
            if (Test-Path -LiteralPath $Lock) { Remove-Item -LiteralPath $Lock -Force }
        }
        $fromLock = Test-Path -LiteralPath $Lock
        $reqFile = if ($fromLock) { $Lock } else { $Req }
        if ($fromLock) {
            Write-Step 'Installing from installed-lock.txt (exact, every package)...'
        } else {
            Write-Step 'Installing pinned dependencies (wheels only)...'
        }
        if (-not (Invoke-Pip @('install', '--only-binary=:all:', '--require-virtualenv', '-r', $reqFile))) {
            Write-Fail 'Installation failed. Nothing was installed outside this folder.'
            Write-Fail '"no matching distribution" means that Python has no wheel for a pin: use 3.12.'
            Write-Fail 'A "path too long" error means this folder needs a shorter path, e.g. C:\Tools\PartEstimator.'
            return $false
        }
        if (-not $fromLock) {
            # Record exactly what resolved, minus this package and test tools.
            $frozen = & $VPy -m pip freeze --exclude-editable --exclude pytest --exclude pluggy --exclude iniconfig --exclude pygments
            if ($LASTEXITCODE -eq 0) {
                Set-Content -LiteralPath $Lock -Value $frozen -Encoding ascii
                Write-Step 'Wrote installed-lock.txt (the exact set that was installed).'
            }
        }
        Set-Content -LiteralPath $Stamp -Value $want -Encoding ascii
        $installedRuntime = $true
    }

    if ($installedRuntime -or -not (Test-Path -LiteralPath $EntryExe)) {
        Write-Step 'Installing partestimator in editable mode...'
        if (-not (Invoke-Pip @('install', '--no-deps', '--require-virtualenv', '--quiet', '-e', $ToolDir))) {
            Write-Fail 'Editable install of partestimator failed.'
            return $false
        }
    }

    if ($Dev -and -not (Test-Path -LiteralPath $PytestExe)) {
        Write-Step 'Installing developer tools (requirements-dev.txt)...'
        if (-not (Invoke-Pip @('install', '--only-binary=:all:', '--require-virtualenv', '-r', $ReqDev))) {
            Write-Fail 'Installing requirements-dev.txt failed.'
            return $false
        }
    }

    $cache = Join-Path $Scratch 'pip-cache'
    if (Test-Path -LiteralPath $cache) {
        Remove-Item -LiteralPath $cache -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $true
}

function Remove-Environment {
    if (-not (Test-Path -LiteralPath $Venv)) {
        Write-Step 'Nothing to reset: there is no .venv.'
        return $true
    }
    Write-Step "Deleting $Venv ..."
    try {
        Remove-Item -LiteralPath $Venv -Recurse -Force -ErrorAction Stop
    } catch {
        Write-Fail 'Could not delete .venv completely. Close Part Estimator and try again.'
        return $false
    }
    return $true
}

function Use-Environment {
    if (Test-EnvReady) { return $true }
    return (Install-Environment)
}

# Start-Process joins arguments with spaces and does no quoting of its own.
function ConvertTo-ArgString([string[]]$ArgList) {
    $parts = foreach ($a in $ArgList) {
        if ($a -match '[\s"]') { '"' + ($a -replace '"', '\"') + '"' } else { $a }
    }
    return ($parts -join ' ')
}

# ---------------------------------------------------------------------------
# commands
#
# Commands let Python's output flow straight to this script's output, so it
# can be piped (`.\pe.ps1 classify *.step --json | ConvertFrom-Json`), and
# record the exit code in $script:ExitCode rather than returning it.
# ---------------------------------------------------------------------------

$script:ExitCode = 0

function Invoke-Gui([string[]]$ArgList) {
    $split = Split-Switches $ArgList @('Wait')
    if (-not (Use-Environment)) { $script:ExitCode = 1; return }
    $pyArgs = @('-m', 'partestimator', 'gui') + $split.Rest
    if ($split.Switches['Wait']) {
        & $VPy @pyArgs
        $script:ExitCode = $LASTEXITCODE
        return
    }
    # pythonw: no console window behind the app. Startup errors go to
    # last_error.log and a message box.
    $exe = if (Test-Path -LiteralPath $VPyW) { $VPyW } else { $VPy }
    Start-Process -FilePath $exe -ArgumentList (ConvertTo-ArgString $pyArgs) -WorkingDirectory (Get-Location).ProviderPath
}

function Invoke-Cli([string]$Command, [string[]]$ArgList) {
    if (-not (Use-Environment)) { $script:ExitCode = 1; return }
    & $VPy -m partestimator $Command @ArgList
    $script:ExitCode = $LASTEXITCODE
}

function Invoke-Test([string[]]$ArgList) {
    $split = Split-Switches $ArgList @('Full')
    if ($split.Switches['Full']) {
        Invoke-Cli 'selftest' @()
        return
    }
    if (-not (Test-EnvReady) -or -not (Test-Path -LiteralPath $PytestExe)) {
        if (-not (Install-Environment -Dev $true)) { $script:ExitCode = 1; return }
    }
    # pytest reads its config from pyproject.toml in this folder.
    Push-Location -LiteralPath $ToolDir
    try {
        & $VPy -m pytest @($split.Rest)
        $script:ExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Invoke-Main([string[]]$ArgList) {
    if ($ArgList.Count -eq 0) {
        Write-Host $Usage
        return
    }
    $command = $ArgList[0].ToLowerInvariant()
    $rest = [string[]]@($ArgList | Select-Object -Skip 1)

    switch ($command) {
        { $_ -in @('help', '-h', '--help', '-?', '/?') } {
            Write-Host $Usage
            return
        }
        'setup' {
            $split = Split-Switches $rest @('Dev', 'Force')
            if ($split.Rest.Count -gt 0) {
                Write-Fail "setup: unexpected argument '$($split.Rest[0])'. Options: -Dev, -Force"
                $script:ExitCode = 2
                return
            }
            if (Install-Environment -Dev ([bool]$split.Switches['Dev']) -Force ([bool]$split.Switches['Force'])) {
                Write-Host ''
                Write-Step 'Ready. Try: .\pe.ps1 gui    or    .\pe.ps1 test'
            } else {
                $script:ExitCode = 1
            }
            return
        }
        'reset' {
            if (-not (Remove-Environment)) { $script:ExitCode = 1 }
            return
        }
        'py' {
            if (-not (Use-Environment)) { $script:ExitCode = 1; return }
            & $VPy @rest
            $script:ExitCode = $LASTEXITCODE
            return
        }
        'gui'  { Invoke-Gui $rest; return }
        'test' { Invoke-Test $rest; return }
        { $_ -in @('classify', 'sheet', 'calibrate', 'selftest') } {
            Invoke-Cli $command $rest
            return
        }
        default {
            # `.\pe.ps1 part.step`, or a STEP file dropped on a shortcut.
            if ($ArgList[0] -match '\.(step|stp)$' -and (Test-Path -LiteralPath $ArgList[0])) {
                Invoke-Gui $ArgList
                return
            }
            Write-Fail "unknown command '$($ArgList[0])'. Run .\pe.ps1 help"
            $script:ExitCode = 2
            return
        }
    }
}

# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

$saved = @{}
foreach ($k in $ToolEnv.Keys) {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
    [Environment]::SetEnvironmentVariable($k, $ToolEnv[$k], 'Process')
}
try {
    Invoke-Main ([string[]]@($args | ForEach-Object { [string]$_ }))
} finally {
    foreach ($k in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process')
    }
}
exit ([int]$script:ExitCode)
