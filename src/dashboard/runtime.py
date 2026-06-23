# A_EYE 웹 대시보드용 오케스트레이터 스트림 런타임입니다.
import math
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.orchestrator.orchestrator import AnomalyInspectionOrchestrator


@dataclass(frozen=True)
class SensorSpec:
    sensor_id: int
    raw_rate: int
    sensor_type: str
    phase: float
    frequency: float


class DashboardRuntime:
    """Runs the existing inspection orchestrator and exposes UI-ready state."""

    def __init__(self, sensor_count: int = 200, interval_sec: float = 0.5):
        self.sensor_count = sensor_count
        self.interval_sec = interval_sec
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng(20260623)
        self._history: List[Dict[str, Any]] = []
        self._sensor_specs = self._build_sensor_specs(sensor_count)
        self._last_buffers: Dict[int, np.ndarray] = {}
        self._last_feedback: Dict[str, Any] = {"status": "none"}
        self._sequence = 0
        self._revision = 0
        self._running = False
        self._scenario: Dict[str, Any] = {"type": "normal", "sensors": []}
        self._orchestrator = self._create_orchestrator()
        self._state = self._empty_state()

    def start(self) -> None:
        with self._lock:
            self._running = True
            self._commit_state(self._compose_state(self._state.get("event", {})))
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="a-eye-dashboard-runtime", daemon=True)
            self._thread.start()
    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._commit_state(self._compose_state(self._state.get("event", {})))
    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self._history.clear()
            self._last_buffers.clear()
            self._last_feedback = {"status": "none"}
            self._sequence = 0
            self._scenario = {"type": "normal", "sensors": []}
            self._orchestrator = self._create_orchestrator()
            self._running = True
            self._commit_state(self._empty_state())
            return self.snapshot()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def sensor_payload(self, sensor_id: int) -> Dict[str, Any]:
        with self._lock:
            return self._selected_sensor_payload(sensor_id)

    def apply_feedback(self, is_actual_anomaly: bool) -> Dict[str, Any]:
        with self._lock:
            result = self._orchestrator.handle_operator_feedback(is_actual_anomaly=is_actual_anomaly)
            self._last_feedback = self._jsonable({
                "status": result.get("status", "registered"),
                "is_actual_anomaly": bool(is_actual_anomaly),
                "feedback_count": result.get("feedback_count", len(self._orchestrator.cla.feedback_buffer)),
                "updated_at": self._now_iso(),
                "update_summary": result.get("update_summary"),
            })
            self._commit_state(self._compose_state(self._state.get("event", {})))
            return self.snapshot()
    def update_threshold(self, threshold: float) -> Dict[str, Any]:
        threshold = self._safe_float(threshold, 2.5)
        threshold = max(0.5, min(10.0, threshold))
        with self._lock:
            self._orchestrator.ada.update_threshold(threshold)
            event = dict(self._state.get("event", {}))
            event["threshold"] = threshold
            self._commit_state(self._compose_state(event))
            return self.snapshot()

    def set_running(self, should_run: bool) -> Dict[str, Any]:
        if should_run:
            self.start()
        else:
            self.stop()
        return self.snapshot()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                should_run = self._running
            if should_run:
                try:
                    self._step_once()
                except Exception as exc:
                    with self._lock:
                        self._running = False
                        error_state = {
                            **self._state,
                            "running": False,
                            "error": str(exc),
                            "updated_at": self._now_iso(),
                        }
                        self._commit_state(error_state)
            time.sleep(self.interval_sec)

    def _step_once(self) -> None:
        with self._lock:
            step = self._sequence + 1
            buffers, scenario = self._make_sensor_buffers(step)
            raw_event = self._orchestrator.process_realtime_frame(buffers)
            event = self._normalize_event(raw_event, step)
            self._sequence = step
            self._scenario = scenario
            self._history.append({
                "step": event["step"],
                "score": event["anomaly_score"],
                "threshold": event["threshold"],
                "drift": event["drift_value"],
                "alarm": event["is_anomaly"],
                "scenario": scenario["type"],
            })
            if len(self._history) > 180:
                self._history = self._history[-180:]
            self._commit_state(self._compose_state(event))

    def _commit_state(self, state: Dict[str, Any]) -> None:
        self._revision += 1
        committed = dict(state)
        committed["revision"] = self._revision
        committed["updated_at"] = self._now_iso()
        self._state = self._jsonable(committed)

    def _compose_state(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event = self._normalize_event(event, self._sequence)
        step = event["step"]
        score = event["anomaly_score"]
        threshold = event["threshold"]
        drift = event["drift_value"]
        top_k = event["top_k_faults"]
        top_ids = {item["sensor_id"] for item in top_k if item["sensor_id"] >= 0}
        scenario_ids = set(self._scenario.get("sensors", []))
        sensors = self._build_sensor_tiles(top_ids, scenario_ids, score, threshold)
        counts = {
            "normal": sum(1 for item in sensors if item["status"] == "normal"),
            "warning": sum(1 for item in sensors if item["status"] == "warning"),
            "alarm": sum(1 for item in sensors if item["status"] == "alarm"),
        }
        selected_id = top_k[0]["sensor_id"] if top_k else 0
        return {
            "app": "A_EYE Real-Time Sensor Health Inspector",
            "revision": self._revision,
            "running": self._running,
            "updated_at": self._now_iso(),
            "step": step,
            "source": "orchestrator-sim",
            "event": event,
            "anomaly_score": score,
            "threshold": threshold,
            "drift_value": drift,
            "active_alarm": event["active_alarm"],
            "scenario": self._jsonable(self._scenario),
            "counts": counts,
            "sensors": sensors,
            "top_k_faults": top_k,
            "diagnostic_report": event["diagnostic_report"],
            "history": list(self._history),
            "selected_sensor": self._selected_sensor_payload(selected_id),
            "feedback": {
                "last": self._last_feedback,
                "count": len(self._orchestrator.cla.feedback_buffer),
                "trigger_limit": self._orchestrator.cla.feedback_trigger_limit,
            },
            "model": {
                "status": self._orchestrator.ada.status,
                "model_ready": bool(self._orchestrator.ada.model_ready),
                "online_buffer_count": len(self._orchestrator.cla.online_data_buffer),
                "online_buffer_limit": self._orchestrator.cla.max_online_buffer_size,
                "baseline_model_error": self._safe_float(self._orchestrator.ada.baseline_model_error, 0.0),
            },
            "error": None,
        }

    def _empty_state(self) -> Dict[str, Any]:
        event = {
            "step": 0,
            "anomaly_score": 0.0,
            "threshold": 2.5,
            "is_anomaly": False,
            "active_alarm": False,
            "drift_value": 0.0,
            "top_k_faults": [],
            "diagnostic_report": "",
        }
        return self._compose_state(event)

    def _normalize_event(self, raw_event: Any, fallback_step: int) -> Dict[str, Any]:
        event = raw_event if isinstance(raw_event, dict) else {}
        is_anomaly = self._safe_bool(event.get("is_anomaly", False))
        normalized = {
            "step": self._safe_int(event.get("step", fallback_step), fallback_step),
            "anomaly_score": self._safe_float(event.get("anomaly_score", 0.0), 0.0),
            "threshold": self._safe_float(event.get("threshold", self._orchestrator.ada.threshold), self._orchestrator.ada.threshold),
            "is_anomaly": is_anomaly,
            "active_alarm": self._safe_bool(event.get("active_alarm", is_anomaly)),
            "drift_value": self._safe_float(event.get("drift_value", 0.0), 0.0),
            "top_k_faults": [],
            "diagnostic_report": str(event.get("diagnostic_report", "") or ""),
        }
        faults = event.get("top_k_faults", []) or []
        if isinstance(faults, dict):
            faults = [faults]
        for fault in faults:
            normalized_fault = self._normalize_fault(fault)
            if normalized_fault["sensor_id"] >= 0:
                normalized["top_k_faults"].append(normalized_fault)
        return normalized

    def _normalize_fault(self, fault: Any) -> Dict[str, Any]:
        item = fault if isinstance(fault, dict) else {}
        return {
            "sensor_id": self._safe_int(item.get("sensor_id", -1), -1),
            "recon_error": self._safe_float(item.get("recon_error", 0.0), 0.0),
            "anomaly_type": str(item.get("anomaly_type", "unknown") or "unknown"),
            "confidence": self._safe_float(item.get("confidence", 0.0), 0.0),
            "rate_hz": self._safe_int(item.get("rate_hz", 0), 0),
        }

    def _create_orchestrator(self) -> AnomalyInspectionOrchestrator:
        config = {
            "reference_rate": 200,
            "window_sec": 1.0,
            "default_rate": 200,
            "log_path": "results/dashboard_inspection_log.jsonl",
            "ada_cfg": {
                "initial_threshold": 2.5,
                "fast_reset_threshold": 1.5,
                "dataset_mode": "sim",
                "persistence_window": 7,
                "min_trigger_count": 5,
            },
            "cla_cfg": {
                "drift_threshold": 0.35,
                "feedback_trigger_limit": 5,
                "max_online_buffer_size": 30,
            },
            "updater_cfg": {
                "epochs_online": 1,
                "epochs_feedback": 2,
                "threshold_upper": 10.0,
            },
        }
        orchestrator = AnomalyInspectionOrchestrator(config, device="cpu")
        for spec in self._sensor_specs:
            orchestrator.register_sensor(spec.sensor_id, initial_rate=spec.raw_rate, type_hint=spec.sensor_type)
        orchestrator.cla.fit_baseline([1.1, 1.2, 1.3, 1.25, 1.18, 1.22, 1.28, 1.24])
        return orchestrator

    def _build_sensor_specs(self, sensor_count: int) -> List[SensorSpec]:
        rates = [2000, 200, 50, 20, 5, 1]
        specs = []
        for sensor_id in range(sensor_count):
            raw_rate = rates[sensor_id % len(rates)]
            if sensor_id % 10 in (0, 7):
                sensor_type = "bits"
            elif sensor_id % 10 in (3, 8):
                sensor_type = "2word_lsb"
            else:
                sensor_type = "1word"
            specs.append(SensorSpec(
                sensor_id=sensor_id,
                raw_rate=raw_rate,
                sensor_type=sensor_type,
                phase=(sensor_id % 17) * 0.37,
                frequency=0.15 + (sensor_id % 9) * 0.04,
            ))
        return specs

    def _make_sensor_buffers(self, step: int) -> tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        scenario = self._scenario_for_step(step)
        buffers: Dict[int, np.ndarray] = {}
        active_ids = set(scenario["sensors"])
        for spec in self._sensor_specs:
            length = max(1, int(spec.raw_rate))
            sample_t = (step - 1) + np.arange(length) / max(spec.raw_rate, 1)
            if spec.sensor_type == "bits":
                bit = ((np.sin(sample_t * 0.8 + spec.phase) > 0.4).astype(np.int64) << (spec.sensor_id % 12))
                raw = bit + ((np.arange(length) + spec.sensor_id + step) % 19 == 0).astype(np.int64)
            elif spec.sensor_type == "2word_lsb":
                raw = (24000 + spec.sensor_id * 97 + step * 37 + np.arange(length) * 11) & 0xFFFF
                raw = raw.astype(np.float64)
            else:
                wave = 31000 + 4200 * np.sin(2 * math.pi * spec.frequency * sample_t + spec.phase)
                harmonic = 900 * np.sin(2 * math.pi * spec.frequency * 3 * sample_t)
                noise = self._rng.normal(0, 35, length)
                raw = wave + harmonic + noise

            if spec.sensor_id in active_ids:
                raw = self._inject_fault(raw.astype(np.float64), scenario, spec, step)
            else:
                raw = raw.astype(np.float64) + self._rng.normal(0, 8, length)

            clipped = np.clip(raw, 0, 65535).astype(np.int64)
            buffers[spec.sensor_id] = clipped
            self._last_buffers[spec.sensor_id] = clipped
        return buffers, scenario

    def _scenario_for_step(self, step: int) -> Dict[str, Any]:
        cycle = (step - 1) % 180
        scenarios = [
            (24, 44, "stuck", [15, 30, 45, 66, 69, 95, 115, 128, 158, 170]),
            (62, 78, "spike", [6, 24, 63, 66, 70, 132, 144, 175, 193]),
            (102, 128, "drift", [12, 38, 51, 84, 107, 121, 146, 168]),
            (146, 164, "noiseburst", [19, 37, 72, 87, 109, 157, 178, 192]),
        ]
        for start, end, fault_type, sensors in scenarios:
            if start <= cycle < end:
                return {
                    "type": fault_type,
                    "sensors": [sid for sid in sensors if sid < self.sensor_count],
                    "cycle_start": start,
                    "cycle_end": end,
                    "progress": (cycle - start) / max(1, end - start - 1),
                }
        return {"type": "normal", "sensors": [], "cycle_start": None, "cycle_end": None, "progress": 0.0}

    def _inject_fault(self, raw: np.ndarray, scenario: Dict[str, Any], spec: SensorSpec, step: int) -> np.ndarray:
        fault_type = scenario["type"]
        progress = float(scenario.get("progress", 0.0))
        if raw.size == 0:
            return raw
        if fault_type == "stuck":
            return np.full_like(raw, raw[0]) + self._rng.normal(0, 1.5, raw.size)
        if fault_type == "spike":
            pulse = np.exp(-np.linspace(0, 4.0, raw.size)) * (9000 + (spec.sensor_id % 5) * 1300)
            return raw + pulse
        if fault_type == "drift":
            return raw + 8500 * progress
        if fault_type == "noiseburst":
            return raw + self._rng.normal(0, 4600, raw.size)
        return raw

    def _build_sensor_tiles(self, top_ids: set[int], scenario_ids: set[int], score: float, threshold: float) -> List[Dict[str, Any]]:
        tiles = []
        for spec in self._sensor_specs:
            status = "normal"
            if spec.sensor_id in top_ids:
                status = "alarm"
            elif spec.sensor_id in scenario_ids and score > threshold * 0.55:
                status = "warning"
            tiles.append({
                "id": spec.sensor_id,
                "rate": spec.raw_rate,
                "type": spec.sensor_type,
                "status": status,
                "label": f"S{spec.sensor_id:03d}",
            })
        return tiles

    def _selected_sensor_payload(self, sensor_id: int) -> Dict[str, Any]:
        if not self._sensor_specs:
            return {"id": 0, "rate": 0, "type": "unknown", "series": []}
        spec = self._sensor_specs[min(max(sensor_id, 0), len(self._sensor_specs) - 1)]
        raw = self._last_buffers.get(spec.sensor_id, np.array([], dtype=np.int64))
        if raw.size > 80:
            idx = np.linspace(0, raw.size - 1, 80).astype(int)
            series = raw[idx].astype(float).tolist()
        else:
            series = raw.astype(float).tolist()
        return {
            "id": spec.sensor_id,
            "rate": spec.raw_rate,
            "type": spec.sensor_type,
            "series": series,
        }

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        return parsed if math.isfinite(parsed) else float(default)

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "y")
        return bool(value)

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else 0.0
        return value

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


