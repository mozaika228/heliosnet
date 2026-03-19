param(
  [string]$Config = ".\configs\dev.yaml"
)

$env:PYTHONPATH = ".\src"
python -c "from heliosnet.app import run; run(r'$Config')"
