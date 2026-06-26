# SWaT 벤치마크에서 이상 탐지 성능을 평가하는 스크립트
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.evaluation import binary_detection_metrics, legacy_prefixed_metrics, standard_detection_report
from src.utils.device import resolve_torch_device
from src.data.swat_loader import get_swat_data
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

def swat_collate_fn(batch_list):
    high_list = []
    mid_list = []
    low_list = []
    bits_list = []

    for item in batch_list:
        h = item['high'].t().unsqueeze(-1)
        m = item['mid'].t().unsqueeze(-1)
        l = item['low'].unsqueeze(-1)
        b = item['bits'].unsqueeze(0).repeat(h.shape[0], 1)

        high_list.append(h)
        mid_list.append(m)
        low_list.append(l)
        bits_list.append(b)

    return {
        'high': torch.cat(high_list, dim=0),
        'mid': torch.cat(mid_list, dim=0),
        'low': torch.cat(low_list, dim=0),
        'bits': torch.cat(bits_list, dim=0)
    }

def evaluate_swat(device='cpu'):
    print("\nEvaluating SWaT Dataset...")
    # Remove single thread restriction to utilize multicore CPU speedup
    # torch.set_num_threads(1)
    window_size = 100

    # 1. Load sampled windows for faster benchmark calibration.
    train_window_stride = 3000
    test_window_stride = 500
    train_ds, test_ds, y_true, num_sensors, num_bits = get_swat_data(
        window_size=window_size,
        train_subsample_step=train_window_stride,
        test_subsample_step=test_window_stride,
    )

    # 2. Build model and train
    model_config = {
        'high_ch': 1,
        'mid_ch': 1,
        'low_dim': 1,
        'bit_len': num_bits,
        'enc_dim': 64,
        'proj_dim': 32,
        'recon_dim': window_size,
        'pred_steps': 1,
        'tau': 0.07,
        'lambda_cpc': 1.0,
        'time_tokens': int(os.environ.get('A_EYE_TIME_TOKENS', '4')),
        'lr': 1e-3
    }

    orchestrator_config = {
        "reference_rate": 1,
        "window_sec": window_size,
        "default_rate": 1,
        "log_path": "results/swat_inspection.jsonl",
        "ada_cfg": {
            "initial_threshold": 2.2,
            "fast_reset_threshold": 1.2,
            "dataset_mode": "swat",
            "model_config": model_config,
            "persistence_window": 10,
            "min_trigger_count": 5
        },
        "cla_cfg": {
            "drift_threshold": 100.0,
            "feedback_trigger_limit": 1000,
            "max_online_buffer_size": 100
        }
    }

    orchestrator = AnomalyInspectionOrchestrator(orchestrator_config, device=device)

    # Pre-train
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=swat_collate_fn)
    model = orchestrator.ada.model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=model_config.get('lr', 1e-3), weight_decay=1e-4)
    from main_anomaly_contrastive import info_nce_loss

    print("Training model on SWaT normal dataset...")
    model.train()
    import time
    t0 = time.time()
    for epoch in range(1):
        for batch_idx, batch in enumerate(train_loader):
            t_batch_start = time.time()
            batch_a = {}
            batch_b = {}
            for k, v in batch.items():
                x_a = v.clone()
                x_b = v.clone()
                if x_a.is_floating_point():
                    x_a = x_a + 0.005 * torch.randn_like(x_a)
                    x_b = x_b + 0.005 * torch.randn_like(x_b)
                batch_a[k] = x_a.to(device)
                batch_b[k] = x_b.to(device)

            out_a = model(batch_a)
            out_b = model(batch_b)
            z_a = out_a['proj']
            z_b = out_b['proj']

            loss = info_nce_loss(z_a, z_b, temperature=model_config.get('tau', 0.07))

            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Time/Batch: {time.time() - t_batch_start:.3f}s")
                sys.stdout.flush()

    print(f"Pre-train finished in {time.time() - t0:.2f}s")
    orchestrator.ada.model_ready = True

    # 3. Fit ADA baseline
    print("Fitting ADA baseline on normal train dataset...")
    for s_id in range(num_sensors):
        type_hint = 'discrete' if s_id in train_ds.bit_col_indices else '1word'
        orchestrator.register_sensor(s_id, initial_rate=1, type_hint=type_hint)
        orchestrator.ada.baseline_means[s_id] = float(np.mean(train_ds.values[:, s_id]))
        orchestrator.ada.baseline_stds[s_id] = float(np.std(train_ds.values[:, s_id]))
        orchestrator.ada.baseline_counts[s_id] = 15

    fit_loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=swat_collate_fn)
    orchestrator.ada.model.eval()
    fit_errors = []

    with torch.no_grad():
        for i, batch in enumerate(fit_loader):
            if i >= 10:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = orchestrator.ada.model(batch)
            recon = out['recon']
            target = batch['high'][:, :recon.shape[1]].squeeze(-1)
            recon_errors_per_sensor = torch.mean((recon - target) ** 2, dim=1)
            fit_errors.extend(recon_errors_per_sensor.cpu().numpy().tolist())

    orchestrator.ada.baseline_model_error = float(np.mean(fit_errors))
    orchestrator.ada.baseline_model_count = 15
    print(f"ADA Baseline model reconstruction error fitted: {orchestrator.ada.baseline_model_error:.6f}")

    # 4. Fast inference on SWaT test dataset
    test_pre_sampled = len(y_true) == len(test_ds)
    eval_step = 1 if test_pre_sampled else 500
    print(f"Running fast inference on SWaT test dataset (loader-sampled={test_pre_sampled}, eval step={eval_step})...")
    test_ds_len = len(test_ds)

    y_pred_eval = []
    y_true_eval = []

    for idx in range(0, test_ds_len, eval_step):
        if idx % 100000 == 0:
            print(f"  Progress: {idx}/{test_ds_len} windows evaluated...")

        item = test_ds[idx]

        # Build sensor windows to pass directly to ADA
        sensor_windows = []
        for s_id in range(num_sensors):
            s_type = 'discrete' if s_id in test_ds.bit_col_indices else '1word'
            sensor_windows.append({
                'id': s_id,
                'raw_signal': item['high'][:, s_id].numpy(),
                'raw_rate': 1,
                'type': s_type
            })

        # Call ADA process directly, bypassing diagnostic printing and CL loggers
        res = orchestrator.ada.process(sensor_windows)
        y_pred_eval.append(1 if res['is_anomaly'] else 0)

        gt_idx = idx if test_pre_sampled else idx + window_size - 1
        y_true_eval.append(y_true[gt_idx])

    y_pred_eval = np.array(y_pred_eval)
    y_true_eval = np.array(y_true_eval)

    # 5. Calculate metrics
    point_metrics = binary_detection_metrics(y_true_eval, y_pred_eval)
    y_pred_pa = point_adjust(y_pred_eval, y_true_eval)
    point_adjusted_metrics = binary_detection_metrics(y_true_eval, y_pred_pa)

    print("\n" + "="*60)
    print("SWaT Fast Evaluation Complete")
    print(
        f"Pt-F1: {point_metrics['f1']:.4f} (P: {point_metrics['precision']:.4f}, R: {point_metrics['recall']:.4f}) | "
        f"PA-F1: {point_adjusted_metrics['f1']:.4f} (P: {point_adjusted_metrics['precision']:.4f}, R: {point_adjusted_metrics['recall']:.4f})"
    )
    print("="*60)

    metrics = {}
    metrics.update(legacy_prefixed_metrics('pt', point_metrics))
    metrics.update(legacy_prefixed_metrics('pa', point_adjusted_metrics))
    metrics['report'] = standard_detection_report(
        'SWaT',
        point_metrics,
        point_adjusted_metrics,
        extra={
            'train_window_stride': train_window_stride,
            'test_window_stride': test_window_stride if test_pre_sampled else None,
            'evaluated_windows': int(len(y_true_eval)),
        },
    )
    return metrics

def main():
    device = resolve_torch_device()
    print(f"Using device: {device}")

    try:
        metrics = evaluate_swat(device=device)

        os.makedirs("results", exist_ok=True)
        report_path = "results/swat_performance.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"Failed to evaluate SWaT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

