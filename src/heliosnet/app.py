from heliosnet.config import load_config
from heliosnet.pipeline.ingest import IngestService
from heliosnet.pipeline.detect import DetectService
from heliosnet.pipeline.track import TrackService
from heliosnet.pipeline.events import EventsService
from heliosnet.pipeline.edge_bus import EdgeBusService
from heliosnet.runtime.scheduler import Scheduler
from heliosnet.observability.metrics import Metrics


def run(config_path: str) -> None:
    config = load_config(config_path)
    metrics = Metrics()

    ingest = IngestService(config, metrics)
    detect = DetectService(config, metrics)
    track = TrackService(config, metrics)
    events = EventsService(config, metrics)
    edge_bus = EdgeBusService(config, metrics)

    scheduler = Scheduler(config, metrics)

    # Wire pipeline (placeholder)
    ingest.set_next(detect)
    detect.set_next(track)
    track.set_next(events)
    events.set_next(edge_bus)

    scheduler.register(ingest, detect, track, events, edge_bus)
    scheduler.run()
