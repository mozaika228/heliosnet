param(
  [string]$Config = ".\configs\dev.yaml",
  [int]$FramesPerSource = 5
)

$env:PYTHONPATH = ".\src"
python -c "from heliosnet.config import load_config; from heliosnet.pipeline.sources import build_sources; import time; cfg=load_config(r'$Config'); srcs=build_sources(cfg.ingest.get('sources',[]), cfg.ingest.get('max_fps',0), cfg.ingest.get('loop_files', True));
print(f'sources: {len(srcs)}');
for s in srcs:
  got=0
  t0=time.time()
  while got < $FramesPerSource and time.time()-t0<10:
    item=s.read()
    if item is None:
      time.sleep(0.01); continue
    got+=1
  print(f'{s.cfg.source_id} got {got} frames from {s.cfg.uri}')"
