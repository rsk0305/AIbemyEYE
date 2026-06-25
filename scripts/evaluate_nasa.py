# NASA SMAP/MSL 채널에서 이상 탐지 성능을 평가하는 스크립트
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.evaluation import binary_detection_metrics, legacy_prefixed_metrics, standard_detection_report
from src.data.nasa_loader import get_nasa_channel_data, load_nasa_metadata
from src.orchestrator.orchestrator import AnomalyInspectionOrchestrator
from main_anomaly_contrastive import Trainer

def point_adjust(y_pred, y_true):
    adjusted_pred = y_pred.copy()
    in_anomaly = False
    start_idx = -1
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if not in_anomaly:
                in_anomaly = True
                start_idx = i
        else:
            if in_anomaly:
                if np.any(adjusted_pred[start_idx:i] == 1):
                    adjusted_pred[start_idx:i] = 1
                in_anomaly = False
                
    if in_anomaly:
        if np.any(adjusted_pred[start_idx:] == 1):
            adjusted_pred[start_idx:] = 1
            
    return adjusted_pred

def evaluate_channel(chan_id, device='cpu'):
    print(f"\nEvaluating NASA Channel: {chan_id}...")
    window_size = 100
    
    # 1. Load data
    train_ds, test_ds, y_true, bit_len = get_nasa_channel_data(chan_id, window_size=window_size)
    
    # 2. Build model and train
    model_config = {
        'high_ch': 1,
        'mid_ch': 1,
        'low_dim': 1,
        'bit_len': bit_len,
        'enc_dim': 64,
        'proj_dim': 32,
        'recon_dim': 64,
        'pred_steps': 1,
        'tau': 0.07,
        'lambda_cpc': 1.0,
        'lr': 1e-3
    }
    
    orchestrator_config = {
        "reference_rate": 1,
        "window_sec": 100,
        "default_rate": 1,
        "log_path": f"results/nasa_inspection_{chan_id}.jsonl",
        "ada_cfg": {
            "initial_threshold": 3.0,
            "fast_reset_threshold": 1.5,
            "dataset_mode": "nasa",
            "model_config": model_config,
            "persistence_window": 3,
            "min_trigger_count": 2
        },
        "cla_cfg": {
            "drift_threshold": 100.0,
            "feedback_trigger_limit": 1000,
            "max_online_buffer_size": 100
        }
    }
    
    orchestrator = AnomalyInspectionOrchestrator(orchestrator_config, device=device)
    
    # Pre-train
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    trainer = Trainer(orchestrator.ada.model, torch.device(device), model_config)
    trainer.pretrain_contrastive(train_loader, epochs=5)
    orchestrator.ada.model_ready = True
    
    # Fit ADA baseline
    print(f"[{chan_id}] Fitting ADA baseline...")
    orchestrator.ada.baseline_means[0] = float(np.mean(train_ds.telemetry))
    orchestrator.ada.baseline_stds[0] = float(np.std(train_ds.telemetry))
    orchestrator.ada.baseline_counts[0] = 15
    
    fit_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    fit_errors = []
    orchestrator.ada.model.eval()
    with torch.no_grad():
        for batch in fit_loader:
            high = batch['high'].to(device)
            mid = batch['mid'].to(device)
            low = batch['low'].to(device)
            bits = batch['bits'].to(device)
            
            out = orchestrator.ada.model({
                'high': high,
                'mid': mid,
                'low': low,
                'bits': bits
            })
            recon = out['recon']
            target = high[:, :recon.shape[1]].squeeze(-1)
            recon_errors_per_sensor = torch.mean((recon - target) ** 2, dim=1)
            fit_errors.extend(recon_errors_per_sensor.cpu().numpy().tolist())
            
    orchestrator.ada.baseline_model_error = float(np.mean(fit_errors))
    orchestrator.ada.baseline_model_count = 15
    
    # 3. Fast inference directly on ADA.process
    print(f"[{chan_id}] Running fast inference...")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    y_pred = []
    y_pred.extend([0] * (window_size - 1))
    
    for batch in test_loader:
        t_window = batch['high'][0].squeeze().numpy()
        
        # Call ADA process directly, bypassing diagnostic and CL loggers
        res = orchestrator.ada.process([
            {'id': 0, 'raw_signal': t_window, 'raw_rate': 1, 'type': '1word'}
        ])
        y_pred.append(1 if res['is_anomaly'] else 0)
        
    y_pred = np.array(y_pred)
    
    # 4. Calculate metrics
    point_metrics = binary_detection_metrics(y_true, y_pred)
    y_pred_pa = point_adjust(y_pred, y_true)
    point_adjusted_metrics = binary_detection_metrics(y_true, y_pred_pa)
    
    print(f"[{chan_id}] Pt-F1: {point_metrics['f1']:.4f} | PA-F1: {point_adjusted_metrics['f1']:.4f}")
    
    metrics = {}
    metrics.update(legacy_prefixed_metrics('pt', point_metrics))
    metrics.update(legacy_prefixed_metrics('pa', point_adjusted_metrics))
    metrics['report'] = standard_detection_report(
        'NASA SMAP/MSL',
        point_metrics,
        point_adjusted_metrics,
        extra={'channel': chan_id},
    )
    return metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    metadata = load_nasa_metadata()
    selected_channels = ['A-1', 'D-1', 'M-1']
    
    results = {}
    for chan in selected_channels:
        if chan in metadata:
            try:
                metrics = evaluate_channel(chan, device=device)
                results[chan] = metrics
            except Exception as e:
                print(f"Failed to evaluate {chan}: {e}")
                import traceback
                traceback.print_exc()
                
    avg_pt_f1 = np.mean([r['pt_f1'] for r in results.values()])
    avg_pa_f1 = np.mean([r['pa_f1'] for r in results.values()])
    
    report = {
        'channels': results,
        'average_point_f1': float(avg_pt_f1),
        'average_pa_f1': float(avg_pa_f1),
        'report': {
            'schema_version': 1,
            'dataset': 'NASA SMAP/MSL',
            'metrics': {
                'average_point_f1': float(avg_pt_f1),
                'average_point_adjusted_f1': float(avg_pa_f1),
            },
            'channels': selected_channels,
        }
    }
    
    os.makedirs("results", exist_ok=True)
    report_path = "results/nasa_performance.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*60)
    print("NASA SMAP/MSL Fast Evaluation Complete")
    print(f"Saved results to {report_path}")
    print(f"Average Point-level F1: {avg_pt_f1:.4f}")
    print(f"Average Point-Adjusted F1: {avg_pa_f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

