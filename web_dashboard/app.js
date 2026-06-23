// A_EYE 대시보드의 실시간 상태 스트림과 인터랙션을 관리합니다.
const els = {
  connectionState: document.querySelector('#connectionState'),
  runState: document.querySelector('#runState'),
  stepBadge: document.querySelector('#stepBadge'),
  sourceLabel: document.querySelector('#sourceLabel'),
  sensorGrid: document.querySelector('#sensorGrid'),
  normalCount: document.querySelector('#normalCount'),
  warningCount: document.querySelector('#warningCount'),
  alarmCount: document.querySelector('#alarmCount'),
  scoreValue: document.querySelector('#scoreValue'),
  thresholdValue: document.querySelector('#thresholdValue'),
  scoreChart: document.querySelector('#scoreChart'),
  alarmBanner: document.querySelector('#alarmBanner'),
  scenarioBadge: document.querySelector('#scenarioBadge'),
  faultList: document.querySelector('#faultList'),
  diagnosticReport: document.querySelector('#diagnosticReport'),
  driftValue: document.querySelector('#driftValue'),
  driftMeter: document.querySelector('#driftMeter'),
  bufferCount: document.querySelector('#bufferCount'),
  feedbackCount: document.querySelector('#feedbackCount'),
  modelReady: document.querySelector('#modelReady'),
  detailTitle: document.querySelector('#detailTitle'),
  selectedType: document.querySelector('#selectedType'),
  selectedRate: document.querySelector('#selectedRate'),
  sensorChart: document.querySelector('#sensorChart'),
  lastFeedback: document.querySelector('#lastFeedback'),
  startBtn: document.querySelector('#startBtn'),
  stopBtn: document.querySelector('#stopBtn'),
  resetBtn: document.querySelector('#resetBtn'),
  trueAlarmBtn: document.querySelector('#trueAlarmBtn'),
  falseAlarmBtn: document.querySelector('#falseAlarmBtn'),
  thresholdSlider: document.querySelector('#thresholdSlider'),
  thresholdOutput: document.querySelector('#thresholdOutput'),
  applyThresholdBtn: document.querySelector('#applyThresholdBtn'),
};

const endpoints = {
  state: `${window.location.origin}/api/state`,
  sensor: (sensorId) => `${window.location.origin}/api/sensor?id=${encodeURIComponent(sensorId)}`,
  events: `${window.location.origin}/api/events`,
  feedback: `${window.location.origin}/api/feedback`,
  threshold: `${window.location.origin}/api/threshold`,
  control: `${window.location.origin}/api/control`,
};

const sensorButtons = new Map();
let latestState = null;
let selectedSensorId = 0;
let selectedSensorPayload = null;
let eventSource = null;
let pollTimer = null;
let reconnectTimer = null;
let lastRevision = -1;
let thresholdEditing = false;
let thresholdDirty = false;

function setConnection(label, mode) {
  els.connectionState.textContent = label;
  els.connectionState.classList.remove('status-live', 'status-wait', 'status-alert');
  els.connectionState.classList.add(mode);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function fetchState() {
  const response = await fetch(endpoints.state, { cache: 'no-store' });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function fetchSensor(sensorId) {
  const response = await fetch(endpoints.sensor(sensorId), { cache: 'no-store' });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function getRevision(state) {
  return Number(state.revision ?? state.step ?? 0);
}

function acceptState(state, force = false) {
  const revision = getRevision(state);
  if (!force && revision < lastRevision) return;
  lastRevision = Math.max(lastRevision, revision);
  latestState = state;
  renderState(state);
}

function connectEvents() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(endpoints.events);
  eventSource.onopen = () => {
    stopPollingFallback();
    setConnection('실시간 연결', 'status-live');
  };
  eventSource.addEventListener('state', (event) => {
    acceptState(JSON.parse(event.data));
  });
  eventSource.onerror = () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    setConnection('재연결 중', 'status-wait');
    startPollingFallback();
    scheduleReconnect();
  };
}

function startPollingFallback() {
  if (pollTimer) return;
  pollTimer = window.setInterval(async () => {
    try {
      acceptState(await fetchState());
      setConnection('폴링 연결', 'status-wait');
    } catch (error) {
      setConnection('연결 실패', 'status-alert');
    }
  }, 1000);
}

function stopPollingFallback() {
  if (!pollTimer) return;
  window.clearInterval(pollTimer);
  pollTimer = null;
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connectEvents();
  }, 2500);
}

function renderState(state) {
  els.sourceLabel.textContent = state.source || 'orchestrator';
  els.runState.textContent = state.running ? '모니터링 중' : '일시 정지';
  els.runState.classList.toggle('status-live', Boolean(state.running));
  els.stepBadge.textContent = `step ${state.step}`;
  els.scoreValue.textContent = Number(state.anomaly_score || 0).toFixed(2);
  els.thresholdValue.textContent = `threshold ${Number(state.threshold || 0).toFixed(2)}`;
  if (!thresholdEditing && !thresholdDirty) {
    syncThresholdControl(state.threshold || 2.5);
  }

  const counts = state.counts || { normal: 0, warning: 0, alarm: 0 };
  els.normalCount.textContent = counts.normal || 0;
  els.warningCount.textContent = counts.warning || 0;
  els.alarmCount.textContent = counts.alarm || 0;

  const scenario = state.scenario || { type: 'normal' };
  els.scenarioBadge.textContent = scenario.type || 'normal';
  els.alarmBanner.textContent = state.active_alarm ? '비정상 알람 활성화' : '정상 감시 중';
  els.alarmBanner.classList.toggle('hot', Boolean(state.active_alarm));
  els.alarmBanner.classList.toggle('idle', !state.active_alarm);
  els.alarmBanner.setAttribute('role', state.active_alarm ? 'alert' : 'status');

  renderSensors(state.sensors || []);
  renderFaults(state.top_k_faults || []);
  els.diagnosticReport.textContent = state.diagnostic_report || 'No active diagnostic report.';

  const drift = Number(state.drift_value || 0);
  els.driftValue.textContent = drift.toFixed(2);
  els.driftMeter.style.width = `${Math.max(0, Math.min(100, drift * 100))}%`;

  const model = state.model || {};
  const feedback = state.feedback || {};
  els.bufferCount.textContent = `${model.online_buffer_count || 0} / ${model.online_buffer_limit || 0}`;
  els.feedbackCount.textContent = `${feedback.count || 0} / ${feedback.trigger_limit || 0}`;
  els.modelReady.textContent = model.model_ready ? 'on' : 'off';
  els.lastFeedback.textContent = feedback.last?.status || 'none';

  drawScoreChart(state.history || []);
  if (state.selected_sensor && selectedSensorId === Number(state.selected_sensor.id || 0)) {
    renderSelectedSensor(state.selected_sensor);
  }
}

function renderSensors(sensors) {
  const seen = new Set();
  for (const sensor of sensors) {
    const id = Number(sensor.id);
    seen.add(id);
    let button = sensorButtons.get(id);
    if (!button) {
      button = document.createElement('button');
      button.type = 'button';
      button.className = 'sensor-cell normal';
      button.textContent = String(id).padStart(3, '0');
      button.addEventListener('click', () => selectSensor(id));
      sensorButtons.set(id, button);
      els.sensorGrid.appendChild(button);
    }
    updateSensorButton(button, sensor);
  }

  for (const [id, button] of sensorButtons.entries()) {
    if (!seen.has(id)) {
      button.remove();
      sensorButtons.delete(id);
    }
  }
}

function updateSensorButton(button, sensor) {
  const id = Number(sensor.id);
  const status = sensor.status || 'normal';
  const selected = id === selectedSensorId;
  button.className = `sensor-cell ${status}`;
  button.classList.toggle('selected', selected);
  button.title = `${sensor.label} · ${sensor.type} · ${sensor.rate}Hz`;
  button.setAttribute('aria-label', `${sensor.label}, ${statusLabel(status)}, ${sensor.type}, ${sensor.rate}Hz`);
  button.setAttribute('aria-pressed', selected ? 'true' : 'false');
}

function statusLabel(status) {
  if (status === 'alarm') return '이상';
  if (status === 'warning') return '경고';
  return '정상';
}

function renderFaults(faults) {
  if (!faults.length) {
    els.faultList.replaceChildren();
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const fault of faults.slice(0, 5)) {
    const item = document.createElement('li');
    const sid = document.createElement('strong');
    sid.textContent = `S${String(fault.sensor_id).padStart(3, '0')}`;
    const type = document.createElement('span');
    type.textContent = `${fault.anomaly_type || 'unknown'} · ${fault.rate_hz || 0}Hz`;
    const score = document.createElement('span');
    score.textContent = Number(fault.recon_error || 0).toFixed(1);
    item.append(sid, type, score);
    fragment.appendChild(item);
  }
  els.faultList.replaceChildren(fragment);
}

async function selectSensor(sensorId) {
  selectedSensorId = sensorId;
  if (latestState) renderSensors(latestState.sensors || []);
  try {
    const payload = await fetchSensor(sensorId);
    renderSelectedSensor(payload);
  } catch (error) {
    console.error(error);
  }
}

function renderSelectedSensor(sensor) {
  selectedSensorPayload = sensor;
  els.detailTitle.textContent = `S${String(sensor.id).padStart(3, '0')} waveform`;
  els.selectedType.textContent = sensor.type || 'unknown';
  els.selectedRate.textContent = `${sensor.rate || 0}Hz`;
  drawSensorChart(sensor.series || []);
}

function syncThresholdControl(value) {
  const formatted = Number(value || 2.5).toFixed(2);
  els.thresholdSlider.value = formatted;
  els.thresholdOutput.value = formatted;
  els.thresholdOutput.textContent = formatted;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(300, Math.floor(rect.width * ratio));
  const height = Math.max(160, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawScoreChart(history) {
  const { ctx, width, height } = setupCanvas(els.scoreChart);
  ctx.clearRect(0, 0, width, height);
  drawGrid(ctx, width, height);
  if (!history.length) return;
  const scores = history.map((d) => Number(d.score || 0));
  const thresholds = history.map((d) => Number(d.threshold || 0));
  const maxY = Math.max(1, ...scores, ...thresholds) * 1.12;
  drawSeries(ctx, scores, maxY, width, height, '#2d6cdf', 2.4);
  drawSeries(ctx, thresholds, maxY, width, height, '#d8463d', 2, [6, 5]);
  for (let i = 0; i < history.length; i += 1) {
    if (history[i].alarm) {
      const x = (i / Math.max(1, history.length - 1)) * width;
      ctx.fillStyle = 'rgba(216, 70, 61, 0.10)';
      ctx.fillRect(x, 0, Math.max(2, width / history.length), height);
    }
  }
}

function drawSensorChart(series) {
  const { ctx, width, height } = setupCanvas(els.sensorChart);
  ctx.clearRect(0, 0, width, height);
  drawGrid(ctx, width, height);
  if (!series.length) return;
  const minY = Math.min(...series);
  const maxY = Math.max(...series);
  const span = Math.max(1, maxY - minY);
  const normalized = series.map((value) => (value - minY) / span);
  drawSeries(ctx, normalized, 1, width, height, '#0f9f8f', 2.2);
}

function drawGrid(ctx, width, height) {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#d7dfdd';
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    const y = (height / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 1; i < 8; i += 1) {
    const x = (width / 8) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function drawSeries(ctx, values, maxY, width, height, color, lineWidth, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash);
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * width;
    const y = height - (Number(value || 0) / maxY) * (height - 18) - 9;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

els.thresholdSlider.addEventListener('input', () => {
  thresholdEditing = true;
  thresholdDirty = true;
  syncThresholdControl(els.thresholdSlider.value);
});

els.thresholdSlider.addEventListener('pointerdown', () => {
  thresholdEditing = true;
});

els.thresholdSlider.addEventListener('change', () => {
  thresholdEditing = false;
});

els.applyThresholdBtn.addEventListener('click', async () => {
  thresholdEditing = false;
  thresholdDirty = false;
  acceptState(await postJson(endpoints.threshold, { threshold: Number(els.thresholdSlider.value) }), true);
});

els.trueAlarmBtn.addEventListener('click', async () => {
  acceptState(await postJson(endpoints.feedback, { is_actual_anomaly: true }), true);
});

els.falseAlarmBtn.addEventListener('click', async () => {
  acceptState(await postJson(endpoints.feedback, { is_actual_anomaly: false }), true);
});

els.startBtn.addEventListener('click', async () => {
  acceptState(await postJson(endpoints.control, { action: 'start' }), true);
});

els.stopBtn.addEventListener('click', async () => {
  acceptState(await postJson(endpoints.control, { action: 'stop' }), true);
});

els.resetBtn.addEventListener('click', async () => {
  selectedSensorId = 0;
  selectedSensorPayload = null;
  acceptState(await postJson(endpoints.control, { action: 'reset' }), true);
});

window.addEventListener('resize', () => {
  if (!latestState) return;
  drawScoreChart(latestState.history || []);
  if (selectedSensorPayload) renderSelectedSensor(selectedSensorPayload);
});

(async function boot() {
  try {
    const state = await fetchState();
    selectedSensorId = Number(state.selected_sensor?.id || 0);
    acceptState(state, true);
    connectEvents();
  } catch (error) {
    console.error(error);
    setConnection('연결 실패', 'status-alert');
    startPollingFallback();
    scheduleReconnect();
  }
})();
