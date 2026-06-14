$cache  = "agentic-rag/results/sweep_fb_tau060_cache.csv"
$scores = "agentic-rag/results/sweep_fb_tau060_scores.csv"
$maxIter = 90   # 90 * 80s = ~120 min safety cap
for ($i = 0; $i -lt $maxIter; $i++) {
    $hdr = Get-Content $cache -TotalCount 1
    if ($hdr -match "faithfulness") { "WATCHER_RESULT: MERGE_DONE"; break }
    $proc = Get-Process python -ErrorAction SilentlyContinue
    $crows = (Get-Content $cache | Measure-Object -Line).Lines - 1
    $srows = 0
    if (Test-Path $scores) { $srows = (Get-Content $scores | Measure-Object -Line).Lines - 1 }
    if (-not $proc) {
        "WATCHER_RESULT: PYTHON_DEAD  gen=$crows/150 scored=$srows/150 $(Get-Date -Format HH:mm:ss)"
        break
    }
    "WATCHER: gen=$crows/150 scored=$srows/150 py=$($proc.Id -join ',') $(Get-Date -Format HH:mm:ss)"
    Start-Sleep -Seconds 80
}
if ($i -ge $maxIter) { "WATCHER_RESULT: TIMEOUT_120MIN" }
