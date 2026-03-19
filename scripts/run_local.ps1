param(
  [string]$Config = ".\config\config.example.yaml"
)

$env:PYTHONPATH = "."
.\.venv\Scripts\python -m core.node $Config
