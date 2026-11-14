# Builds Phyllo-ViT-Final-Ready.zip (one folder level up from the project) with everything needed to run,
# excluding venvs, node_modules, build output, caches, local env files and credentials.
param([string]$Out = "")
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
if (-not $Out) { $Out = Join-Path (Split-Path $root -Parent) "Phyllo-ViT-Final-Ready.zip" }
if (Test-Path $Out) { Remove-Item $Out -Force }

$excludeDirs = @(".venv", "venv", "node_modules", ".next", "__pycache__", ".pytest_cache", ".git", ".vscode", ".idea", "out", ".vercel", "build_hf_space")
$excludeFiles = @(".env", ".env.local", "*.log", "*.pyc", "Thumbs.db", "desktop.ini")

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($Out, "Create")
$count = 0
try {
    Get-ChildItem $root -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length + 1)
        $parts = $rel.Split("\")
        if ($parts[0] -eq "frontend" -and $parts[1] -eq "out") { return }
        foreach ($p in $parts[0..($parts.Length - 2)]) { if ($excludeDirs -contains $p) { return } }
        foreach ($pat in $excludeFiles) { if ($_.Name -like $pat) { return } }
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, "Phyllo-ViT-Final/" + $rel.Replace("\", "/"), "Optimal")
        $count++
    }
} finally { $zip.Dispose() }
"{0} files -> {1} ({2:N0} MB)" -f $count, $Out, ((Get-Item $Out).Length / 1MB)
