param([ValidateSet("serve", "test", "preprocess", "benchmark", "tune", "routes", "evaluator")][string]$Task = "serve")
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
$oldPath = $env:PYTHONPATH
try {
    if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") { $python = "$PSScriptRoot\.venv\Scripts\python.exe" }
    else { $python = "python"; $env:PYTHONPATH = "$PSScriptRoot\.runtime;$oldPath" }
    $env:OMP_NUM_THREADS = "2"
    $env:OPENBLAS_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
    switch ($Task) {
        "serve" { & $python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 }
        "test" { & $python -m pytest -q tests/test_review_fixes.py; if ($LASTEXITCODE -eq 0) { & $python -u tests/test_end_to_end.py } }
        "preprocess" { & $python -m src.dataset_builder }
        "benchmark" { & $python -u -m src.experimentation }
        "tune" { & $python -u -m src.bayesian_optimizer }
        "routes" { & $python -u -m src.pipeline }
        "evaluator" { & $python -u -m src.evaluate_evaluator }
    }
    if ($LASTEXITCODE -ne 0) { throw "Task $Task failed (exit $LASTEXITCODE)" }
} finally { $env:PYTHONPATH = $oldPath; Pop-Location }
