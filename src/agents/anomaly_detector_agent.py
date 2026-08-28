# 실시간 센서 윈도우에서 이상 점수와 국소화 점수를 계산하는 에이전트
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.agents.base_agent import BaseAgent
from src.utils.model_inputs import build_bit_context
from main_anomaly_contrastive import MultiModalAnomalyModel

class AnomalyDetectorAgent(BaseAgent):
    """
    실시간 센서 윈도우 데이터 스트림을 분석하여 이상치(Anomaly Score)를 계산하고
    임계치와 비교해 실시간 알람 판단을 내리는 에이전트 (ADA).
    """
    def __init__(self, name: str, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__(name, config, device)
        self.reference_rate = config.get("reference_rate", 200) # Hz
        self.window_sec = config.get("window_sec", 1.0)
        self.dataset_mode = config.get("dataset_mode", "sim") # 'sim', 'nasa', 'swat'
        swat_mode = self.dataset_mode == 'swat'
        default_threshold = 4.0 if swat_mode else 3.0
        default_fast_reset = 1.2 if swat_mode else 1.5
        default_global_model_weight = 0.3 if swat_mode else 0.5
        default_local_model_weight = 0.05 if swat_mode else 0.5
        self.threshold = config.get("initial_threshold", default_threshold) # 기본 임계치
        self.fast_reset_threshold = config.get("fast_reset_threshold", default_fast_reset) # Fast-Reset 임계치
        self.score_policy = config.get("score_policy", {})
        self.global_model_weight = float(self.score_policy.get("global_model_weight", default_global_model_weight))
        self.local_model_weight = float(self.score_policy.get("local_model_weight", default_local_model_weight))
        self.warmup_limit = int(config.get("warmup_limit", 80 if self.dataset_mode == 'swat' else 15))
        self.warmup_fit_start = int(config.get("warmup_fit_start", 50 if self.dataset_mode == 'swat' else 10))
        self.baseline_ema_alpha = float(config.get("baseline_ema_alpha", 0.4 if self.dataset_mode == 'swat' else 0.05))
        self.correlation_rules = config.get("correlation_rules", self._default_correlation_rules())
        self.score_history = []
        self.persistence_window = config.get("persistence_window", 7) # 최근 7초 이력
        self.min_trigger_count = config.get("min_trigger_count", 7 if swat_mode else 5) # 기준 횟수 초과 시 최종 알람
        self.model = None
        self.model_ready = False  # 모델이 1회 이상 학습되기 전까지 앙상블에서 제외
        self._init_model()
        self.baseline_means = {}  # 센서별 정상 상태 평균값 트래킹
        self.baseline_stds = {}  # 센서별 정상 상태 표준편차 트래킹
        self.baseline_counts = {}  # 센서별 평균값 피팅 횟수
        self.sensor_score_history = {}  # 센서별 스코어 히스토리 (임계값 피팅용)
        self.sensor_score_fit_count = {}  # 센서별 스코어 피팅 횟수
        self.sensor_thresholds = {}  # 센서별 동적 임계치
        self.evt_k = config.get("evt_k", 4.5)  # 임계치 배수 감도 k
        self.baseline_model_error = 0.05  # 모델 재구성 오차 베이스라인 초기값
        self.baseline_model_count = 0
        self.status = "initialized"

    def _default_correlation_rules(self) -> List[Dict[str, Any]]:
        """기본 시뮬레이션 상관관계 규칙을 설정 객체로 제공합니다."""
        if self.dataset_mode != 'sim':
            return []
        return [{
            'name': 'valve_pressure_stuck',
            'condition_sensor_id': 112,
            'target_sensor_id': 107,
            'condition_bit': 8,
            'max_target_std': 3.0,
            'boost': 3000.0,
            'base_error': 1.0,
        }]

    def _apply_correlation_rules(self, recon_errors: Dict[int, float], sensor_ids: List[int], sensor_windows: List[Dict[str, Any]]):
        """설정된 센서 간 규칙을 적용해 국소화 점수를 보강합니다."""
        for rule in self.correlation_rules:
            condition_id = rule.get('condition_sensor_id')
            target_id = rule.get('target_sensor_id')
            if condition_id not in sensor_ids or target_id not in sensor_ids:
                continue

            condition_signal = sensor_windows[sensor_ids.index(condition_id)]['raw_signal']
            target_signal = sensor_windows[sensor_ids.index(target_id)]['raw_signal']
            condition_bit = rule.get('condition_bit')
            if condition_bit is None:
                condition_active = bool(np.all(condition_signal))
            else:
                condition_active = bool(np.all((condition_signal.astype(np.int64) & (1 << int(condition_bit))) != 0))

            target_std = float(np.std(target_signal))
            if condition_active and target_std < float(rule.get('max_target_std', 3.0)):
                base_error = float(rule.get('base_error', 1.0))
                recon_errors[target_id] = recon_errors.get(target_id, base_error) + float(rule.get('boost', 3000.0))

    def _score_components(self, heuristic_score: float, model_recon_error: float) -> Dict[str, float]:
        """최종 이상 점수를 구성하는 세부 점수와 가중치를 반환합니다."""
        model_score = float(np.log1p(model_recon_error)) if model_recon_error > 0 else 0.0
        model_weight = self.global_model_weight if (model_recon_error > 0 and self.model_ready) else 0.0
        model_weight = min(max(float(model_weight), 0.0), 1.0)
        heuristic_weight = 1.0 - model_weight
        anomaly_score = heuristic_weight * heuristic_score + model_weight * model_score
        return {
            'heuristic_score': float(heuristic_score),
            'model_score': float(model_score),
            'heuristic_weight': float(heuristic_weight),
            'model_weight': float(model_weight),
            'anomaly_score': float(anomaly_score),
        }
    def _init_model(self):
        """가상 혹은 실제 다중 모달 아노말리 탐지 모델 인스턴스 생성"""
        model_config = {
            'high_ch': 1,
            'mid_ch': 1,
            'low_dim': 1,
            'bit_len': 16,
            'enc_dim': 64,
            'proj_dim': 32,
            'recon_dim': 64,
            'pred_steps': 1,
            'tau': 0.07,
            'lambda_cpc': 1.0,
            'time_tokens': 4
        }
        # 외부 설정이 있을 경우 덮어쓰기
        if "model_config" in self.config:
            model_config.update(self.config["model_config"])
            
        self.model = MultiModalAnomalyModel(model_config).to(self.device)
        self.model.eval()

    def load_model_weights(self, checkpoint_path: str):
        """학습된 모델 가중치를 로드합니다."""
        if os.path.exists(checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.status = f"model_loaded: {os.path.basename(checkpoint_path)}"
            except Exception as e:
                print(f"[ADA] 모델 로드 실패: {e}")
                self.status = "load_failed"

    def update_threshold(self, new_threshold: float):
        """RL 또는 EVT 최적화 알고리즘에 의해 임계치를 동적으로 갱신합니다."""
        self.threshold = new_threshold
        self.status = f"threshold_updated: {new_threshold:.4f}"

    def align_signal(self, raw_signal: np.ndarray, current_rate: int) -> np.ndarray:
        """
        Zero-Order Hold 방식을 활용하여 이기종 주파수의 데이터를
        기준 주파수(예: 200Hz) 크기로 시간축 정렬(Alignment)합니다.
        """
        target_len = int(self.window_sec * self.reference_rate)
        if len(raw_signal) == 0:
            return np.zeros(target_len)
            
        current_len = len(raw_signal)
        if current_rate == self.reference_rate:
            if current_len == target_len:
                return raw_signal
            elif current_len > target_len:
                return raw_signal[:target_len]
            else:
                # 패딩
                pad_len = target_len - current_len
                return np.pad(raw_signal, (0, pad_len), mode='edge')
                
        # Zero-order hold interpolation
        t_original = np.arange(current_len) / current_rate
        t_reference = np.arange(target_len) / self.reference_rate
        
        # 최신 값 유지를 위해 1D interpolation 적용
        aligned = np.interp(t_reference, t_original, raw_signal, 
                            left=raw_signal[0], right=raw_signal[-1])
        return aligned

    def process(self, sensor_windows: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        다양한 주파수를 지닌 센서들의 실시간 최근 윈도우 데이터를 받아서
        정렬 후 모델에 인입시켜 이상 탐지를 판단합니다.
        """
        self.status = "detecting"
        raw_model_recon_error = 0.0
        
        # 1. 주파수별 그룹화 및 정렬
        aligned_sensors = []
        sensor_ids = []
        sensor_types = []
        
        for s in sensor_windows:
            aligned = self.align_signal(s['raw_signal'], s['raw_rate'])
            aligned_sensors.append(aligned)
            sensor_ids.append(s['id'])
            sensor_types.append(s.get('type', '1word'))
            s['aligned_signal'] = aligned  # 캐시 저장
            
        if len(aligned_sensors) == 0:
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'recon_errors': {},
                'score_components': {
                    'heuristic_score': 0.0,
                    'model_score': 0.0,
                    'heuristic_weight': 1.0,
                    'model_weight': 0.0,
                    'anomaly_score': 0.0,
                },
                'threshold_info': {}
            }
            
        # N x T shape matrix 생성
        X = np.stack(aligned_sensors, axis=0) # (num_sensors, target_len)
        X_tensor = torch.from_numpy(X).float().to(self.device) # (N, T)
        
        # 2. 모델 추론: MultiModalAnomalyModel을 실제로 통과시켜 재구성 오차를 취득
        self.model.eval()
        with torch.no_grad():
            # --- [C-1] 딥러닝 모델 기반 재구성 오차 계산 ---
            model_recon_error = 0.0
            try:
                if self.model_ready:
                    # [C-1] 정규화: sim 모드일 때만 0~65535 범위를 [0, 1]로 스케일링
                    if self.dataset_mode == 'sim':
                        X_norm_scaled = X_tensor / 65535.0
                    else:
                        X_norm_scaled = X_tensor  # 이미 외부에서 정규화 완료됨
                    
                    high_input = X_norm_scaled.unsqueeze(-1)  # (N, T, 1)
                    mid_input = X_norm_scaled.unsqueeze(-1)   # (N, T, 1)
                    low_input = X_norm_scaled.mean(dim=1, keepdim=True)  # (N, 1)
                    
                    # 모델의 bit_len을 동적으로 감지하여 bits_input 생성
                    bit_len = self.model.bit_enc.fc[0].in_features
                    bit_context = build_bit_context(
                        sensor_windows,
                        bit_len,
                        context_bits=kwargs.get('context_bits'),
                    )
                    bits_input = torch.from_numpy(bit_context).to(self.device)
                    bits_input = bits_input.unsqueeze(0).expand(X_norm_scaled.shape[0], -1)
                    
                    model_out = self.model({
                        'high': high_input,
                        'mid': mid_input,
                        'low': low_input,
                        'bits': bits_input
                    })
                    
                    # 재구성 오차: 모델 디코더 출력과 정규화된 입력의 MSE
                    recon = model_out['recon']  # (N, recon_dim)
                    target_for_recon = X_norm_scaled[:, :recon.shape[1]]  # 동일 차원으로 슬라이싱
                    
                    # 센서별 MSE 재구성 오차 계산
                    recon_errors_per_sensor = torch.mean((recon - target_for_recon) ** 2, dim=1)  # (N,)
                    # 전체 센서 평균 대신 상위 5개 센서의 평균 에러를 최종 모델 재구성 오차로 사용하여 개별 센서 이상 감지력 극대화
                    top_k_val, _ = torch.topk(recon_errors_per_sensor, k=min(5, len(recon_errors_per_sensor)))
                    raw_model_recon_error = float(top_k_val.mean().item())
                    
                    # 모델 베이스라인 오차 축적 및 피팅
                    if self.baseline_model_count < 15:
                        self.baseline_model_error = (self.baseline_model_error * self.baseline_model_count + raw_model_recon_error) / (self.baseline_model_count + 1)
                        self.baseline_model_count += 1
                    
                    # 베이스라인 대비 상대 오차로 정규화 (정상 시 1.0 수준으로 스케일 보정)
                    model_recon_error = raw_model_recon_error / (self.baseline_model_error + 1e-6)
                else:
                    model_recon_error = 0.0
                    raw_model_recon_error = 0.0
            except Exception:
                # 모델 추론 실패 시 heuristic 단독으로 fallback
                model_recon_error = 0.0
                raw_model_recon_error = 0.0
            
            # --- Heuristic 기반 센서별 이상 탐지 ---
            recon_errors = {}
            total_recon_error = 0.0
            
            for idx, s_id in enumerate(sensor_ids):
                # 원래 센서 스트림 정보 가져오기
                s_info = sensor_windows[idx]
                raw_sig = s_info['raw_signal']
                raw_rate = s_info['raw_rate']
                
                # --- 1) 16-bit Rollover Unwrapping (보간 뭉개짐 방지를 위해 원본 단에서 먼저 언랩 수행) ---
                raw_sig_clean = raw_sig.copy().astype(np.float64)
                # 오직 2word_lsb 센서에 대해서만 안전하게 언랩핑 적용 (rate >= 10Hz)
                if sensor_types[idx] == '2word_lsb' and raw_rate >= 10 and len(raw_sig_clean) > 1:
                    raw_diff = np.diff(raw_sig_clean)
                    # 65535 -> 0 rollover 보정 (half-range threshold = 32768)
                    raw_diff[raw_diff > 32768] -= 65536
                    raw_diff[raw_diff < -32768] += 65536
                    raw_sig_clean = np.concatenate([[raw_sig_clean[0]], raw_sig_clean[0] + np.cumsum(raw_diff)])
                
                std_raw = float(np.std(raw_sig_clean)) if len(raw_sig_clean) > 1 else 0.0
                current_mean = float(np.mean(raw_sig_clean))
                
                # --- [Phase 4] 센서별 정상 평균 및 표준편차 베이스라인 피팅 (초기 15초 누적) ---
                if s_id not in self.sensor_score_history:
                    if s_id not in self.baseline_means:
                        self.baseline_means[s_id] = 0.0
                        self.baseline_counts[s_id] = 0
                    if s_id not in self.baseline_stds:
                        self.baseline_stds[s_id] = std_raw
                    self.sensor_score_history[s_id] = []
                    self.sensor_score_fit_count[s_id] = 0
                    self.sensor_thresholds[s_id] = self.threshold # 기본 글로벌 임계값으로 임시 세팅
                
                if self.baseline_counts[s_id] < 15:
                    self.baseline_means[s_id] = (self.baseline_means[s_id] * self.baseline_counts[s_id] + current_mean) / (self.baseline_counts[s_id] + 1)
                    self.baseline_stds[s_id] = (self.baseline_stds[s_id] * self.baseline_counts[s_id] + std_raw) / (self.baseline_counts[s_id] + 1)
                    self.baseline_counts[s_id] += 1
                
                # --- 2) 이미 언랩되어 깨끗한 신호를 기준 주파수로 정렬 ---
                aligned_clean = self.align_signal(raw_sig_clean, raw_rate)
                sig_clean = torch.from_numpy(aligned_clean).float().to(self.device)
                
                # 차분 계산 (언랩 및 보간 정렬이 완비된 깨끗한 신호 기준)
                diff = torch.abs(torch.diff(sig_clean))
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                # 기본 정상 에러 수준 (1.0 근처)
                err = 1.0
                
                if self.dataset_mode == 'sim':
                    # 1) Stuck 고장 감지: 모든 센서 중 주기가 10Hz 이상이고 std_raw가 극히 낮은 경우 (고정 상태)
                    # [Phase 4] 미세한 잔류 떨림이 있는 부분 고착(partial stuck)을 감지하기 위해 임계치를 0.5 -> 3.0으로 완화
                    if raw_rate >= 10 and std_raw < 3.0:
                        err += 1000.0
                        
                    # 2) Spike 고장 감지: 1word 및 2word_lsb 센서에 대해 적용 (오경보 방지를 위해 주기가 10Hz 이상인 센서만)
                    # [Phase 4] 미세/중형 스파이크 감지와 정상 진동 오경보 차단의 균형을 맞춘 임계치 설정
                    elif sensor_types[idx] in ['1word', '2word_lsb'] and raw_rate >= 10:
                        # 주파수별 스파이크 정규화 및 절대 임계치 미세 분기
                        if raw_rate >= 2000:
                            th_spike_norm = 4.5
                            th_spike_abs = 15000.0
                        elif raw_rate == 200:
                            th_spike_norm = 4.0
                            th_spike_abs = 18000.0
                        elif raw_rate == 50:
                            th_spike_norm = 4.1    # 50Hz 정상 최대 정규화 4.21 부근 고려
                            th_spike_abs = 21500.0  # 50Hz 정상 최대 Abs Diff 22159 부근 고려
                        else:
                            th_spike_norm = 3.8
                            th_spike_abs = 600.0
                        
                        # std_raw가 5.0보다 크면 안전하게 정규화 기반 탐지 수행
                        # 보간으로 인한 스파이크 뭉개짐(smoothing)을 방지하기 위해 보간 전 raw_sig_clean 상에서 차분 계산
                        raw_diff_sig = np.abs(np.diff(raw_sig_clean))
                        max_raw_diff = float(np.max(raw_diff_sig)) if len(raw_diff_sig) > 0 else 0.0
                        
                        is_spike = False
                        if std_raw > 5.0:
                            # Near-zero std 상황에서 노이즈 증폭으로 인한 오경보(FAR 상승)를 원천 차단하기 위해 분모 안정화 오프셋 30.0 추가
                            max_diff_norm = max_raw_diff / (std_raw + 30.0)
                            if max_diff_norm > th_spike_norm:
                                is_spike = True
                        else:
                            if max_raw_diff > th_spike_abs:
                                is_spike = True
                                
                        # 첨도(Kurtosis) 기반 보조 스파이크 감지 (1word 센서 대상)
                        # 정상 sine 파형의 첨도는 ~1.5 이며, 단발성 스파이크가 튄 윈도우는 첨도가 급증함
                        if not is_spike and sensor_types[idx] == '1word' and std_raw > 10.0:
                            mean_sig = np.mean(raw_sig_clean)
                            kurt = np.mean((raw_sig_clean - mean_sig) ** 4) / (std_raw ** 4 + 1e-6)
                            if kurt > 12.0:
                                is_spike = True
                                
                        if is_spike:
                            err += 5000.0
                            
                    # 3) Drift 고장 감지: 오직 1word 센서에만 적용 (2word_lsb는 32-bit counter이므로 언랩핑 시 선형적으로 계속 값이 증가하므로 static baseline 평균 편차 방식 적용 시 상시 오경보 유발)
                    # [Phase 4] 동적 EMA 베이스라인 평균과의 누적 편차(drift_deviation)를 비교하여 드리프트 탐지
                    # 저주파 1word 센서(5Hz, 1Hz)의 드리프트 및 갑작스러운 값 도약을 포착하기 위해 raw_rate >= 1 로 확장
                    if err == 1.0 and sensor_types[idx] == '1word' and raw_rate >= 1:
                        if self.baseline_counts.get(s_id, 0) >= 15:
                            drift_deviation = abs(current_mean - self.baseline_means[s_id])
                            
                            # 50Hz 센서는 정상 평균 편차가 110 미만이므로 th_drift=250.0으로 정밀 튜닝하여 subtle spike(평균 약 300 상승) 검출
                            if raw_rate == 50:
                                th_drift = 250.0
                            elif raw_rate >= 10:
                                th_drift = 3000.0
                            else:
                                th_drift = 1000.0
                                
                            if drift_deviation > th_drift:
                                err += 2000.0
                                
                        # [Phase 4] 정상 상태의 자연적인 완만한 slow drift와 핑크 노이즈는 EMA가 매끄럽게 추종하도록 동적으로 업데이트 (메모리 계수 0.96)
                        # 고장으로 판별된(err > 1.0) 센서의 경우 베이스라인을 동결하여 괴리를 유지하게 함
                        if err == 1.0 and self.baseline_counts.get(s_id, 0) >= 15:
                            self.baseline_means[s_id] = 0.96 * self.baseline_means[s_id] + 0.04 * current_mean
                                
                    # 4) Noiseburst 고장 감지: [Phase 4] 1word 센서의 신호 분산(std)의 급격한 증가를 명시적으로 탐지
                    # 5Hz 센서의 표준편차 기반 감지를 위해 raw_rate >= 5 로 완화
                    if sensor_types[idx] == '1word' and raw_rate >= 5:
                        # 정상 1word std는 800 미만이므로, 약한 노이즈버스트(1500) 및 강한 노이즈버스트(5000)를 잡기 위해 800.0 설정
                        if err == 1.0 and std_raw > 800.0:
                            err += 3000.0
                else:
                    # NASA/SWaT 일반 모드 Heuristic (표준화 & 센서별 임계치)
                    z_drift = 0.0
                    z_spike = 0.0
                    z_stuck = 0.0
                    
                    if sensor_types[idx] != 'discrete':
                        # 안정적인 표준편차 스케일 (하한치 1e-4)
                        scale_std = max(self.baseline_stds.get(s_id, 0.0), 1e-4)
                        
                        # 1) Stuck 고장 감지: 변동이 거의 없는 고정 신호
                        # 만약 센서의 정상 표준편차가 원래 매우 작다면(1e-3 미만) 원래 고정된 센서이므로 Stuck으로 판정하지 않음
                        if len(raw_sig_clean) > 1 and std_raw < 0.01 * scale_std and scale_std > 1e-3:
                            z_stuck = 10.0
                        
                        # 2) Spike 고장 감지
                        if len(raw_sig_clean) > 1:
                            raw_diff_sig = np.abs(np.diff(raw_sig_clean))
                            max_raw_diff = float(np.max(raw_diff_sig)) if len(raw_diff_sig) > 0 else 0.0
                            # 분모 안정화
                            z_spike = max_raw_diff / (scale_std + 1e-4)
                        
                        # 3) Drift 고장 감지
                        if s_id in self.baseline_means and self.baseline_counts.get(s_id, 0) >= 15:
                            drift_deviation = abs(current_mean - self.baseline_means[s_id])
                            z_drift = drift_deviation / (scale_std + 1e-4)
                    
                    # 앙상블 시스템에 넣을 센서별 이상 점수 S_s 계산
                    heur_score_s = z_drift + z_spike + z_stuck
                    
                    if model_recon_error > 0 and self.model_ready:
                        # 모델 에러 표준화
                        model_score_s = float(recon_errors_per_sensor[idx].item() / (self.baseline_model_error + 1e-6))
                        local_model_weight = min(max(self.local_model_weight, 0.0), 1.0)
                        S_s = (1.0 - local_model_weight) * heur_score_s + local_model_weight * model_score_s
                    else:
                        S_s = heur_score_s
                        
                    # 센서별 동적 임계값 피팅 (데이터셋 모드에 따라 웜업 한계 가변화)
                    # SWaT과 같이 장기 테스트 세트에서는 80프레임 동안 신중하게 Calibration을 수행하며,
                    # NASA와 같이 길이가 짧고 이상이 초반에 시작하는 모드에서는 15프레임으로 웜업을 단축해 초기 누락을 방지합니다.
                    warmup_limit = self.warmup_limit
                    warmup_fit_start = self.warmup_fit_start
                    
                    warmup_cnt = self.sensor_score_fit_count.get(s_id, 0)
                    if warmup_cnt < warmup_limit:
                        self.sensor_score_fit_count[s_id] = warmup_cnt + 1
                        if warmup_cnt >= warmup_fit_start:
                            self.sensor_score_history[s_id].append(S_s)
                            if warmup_cnt == (warmup_limit - 1):
                                scores_arr = np.array(self.sensor_score_history[s_id])
                                mu_s = float(np.mean(scores_arr))
                                sigma_s = float(np.std(scores_arr))
                                # q2 피드백 반영: 표준편차 하한을 0.1로 주어 분포 부족 시 오경보 방지
                                self.sensor_thresholds[s_id] = mu_s + self.evt_k * max(sigma_s, 0.1)
                    
                    # 센서 s가 로컬 임계치를 넘었는지 판단
                    th_s = self.sensor_thresholds.get(s_id, self.threshold)
                    # 웜업 및 피팅 완료 후에만 실시간 판정 시작
                    is_sensor_anomaly = (S_s > th_s) if warmup_cnt >= warmup_limit else False
                    
                    # 센서가 정상 동작 상태일 때만 baseline_means 및 stds를 업데이트하여 드리프트 고장의 베이스라인 오염 방지 (로컬 동결 적용)
                    # 단, SWaT 데이터셋의 경우 공정 전환(천이) 시 베이스라인이 상시 추종해주어야 오경보 락인 데드락을 방지할 수 있으므로,
                    # swat 모드에서는 이상 여부와 상관없이 항상 업데이트를 허용하고, 그 외의 모드(NASA 등)에서는 로컬 동결을 엄격히 적용합니다.
                    if s_id in self.baseline_means and self.baseline_counts.get(s_id, 0) >= 15:
                        if self.dataset_mode == 'swat' or warmup_cnt < warmup_limit or not is_sensor_anomaly:
                            ema_alpha = self.baseline_ema_alpha
                            self.baseline_means[s_id] = (1 - ema_alpha) * self.baseline_means[s_id] + ema_alpha * current_mean
                            self.baseline_stds[s_id] = (1 - ema_alpha) * self.baseline_stds[s_id] + ema_alpha * std_raw
                    
                    # 기존 정적 임계값 로직과의 호환성을 위해 비율(R_s)을 err에 매핑하여 저장
                    if is_sensor_anomaly:
                        err = 1000.0 + float(S_s / (th_s + 1e-6))
                    else:
                        err = float(S_s / (th_s + 1e-6))
                    
                recon_errors[s_id] = err
                total_recon_error += err
            
            # --- 5) 다변량 상관관계 고장 감지 ---
            # 특정 센서 ID와 bit 조건은 config의 correlation_rules로 격리합니다.
            self._apply_correlation_rules(recon_errors, sensor_ids, sensor_windows)

            # 전체 200개 센서 중 가장 에러가 높은 상위 5개 센서의 평균 에러를 시스템의 아노말리 대표 오차로 설정
            # 이를 통해 극히 일부 센서의 결함도 신속하게 잡아낼 수 있습니다.
            sorted_errs = sorted(recon_errors.values(), reverse=True)
            top_mean_error = np.mean(sorted_errs[:5]) if len(sorted_errs) >= 5 else np.mean(sorted_errs)
            
            # [C-1] Anomaly Score 최종 산출: Heuristic + Model 앙상블
            # Heuristic score (log-scale)
            heuristic_score = float(np.log1p(top_mean_error))
            
            score_components = self._score_components(heuristic_score, model_recon_error)
            anomaly_score = score_components['anomaly_score']
            
            # 지속성 필터 (Temporal Persistence Filter) 및 Fast-Reset Rule 적용
            # 정상 수준 이하로 떨어지면 쿨다운 오검출(FP) 방지를 위해 즉시 이력을 초기화합니다.
            fast_reset_th = getattr(self, 'fast_reset_threshold', 1.5)
            if anomaly_score < fast_reset_th:
                self.score_history = []
            else:
                self.score_history.append(anomaly_score > self.threshold)
                if len(self.score_history) > self.persistence_window:
                    self.score_history.pop(0)
                
            # 최근 윈도우 동안 누적 초과 횟수가 기준치 이상일 때 최종 알람 트리거
            active_triggers = sum(self.score_history)
            is_anomaly = active_triggers >= self.min_trigger_count
            
            # 정상 상태일 때 모델 베이스라인 오차 EMA 업데이트
            if self.baseline_model_count >= 15:
                if anomaly_score < self.threshold and raw_model_recon_error > 0:
                    self.baseline_model_error = 0.99 * self.baseline_model_error + 0.01 * raw_model_recon_error
                    
        self.status = "idle"
        
        return {
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'recon_errors': recon_errors,
            'score_components': score_components,
            'threshold_info': {
                'global_threshold': float(self.threshold),
                'fast_reset_threshold': float(self.fast_reset_threshold),
                'warmup_limit': int(self.warmup_limit),
                'warmup_fit_start': int(self.warmup_fit_start),
                'sensor_thresholds': {int(k): float(v) for k, v in self.sensor_thresholds.items()},
            }
        }

