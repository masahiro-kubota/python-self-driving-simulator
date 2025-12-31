#!/usr/bin/env python3
# Usage:
#   uv run scripts/system_identification/estimate_steering_dynamics.py train scripts/system_identification/data/rosbag2_autoware_0.mcap
#   uv run scripts/system_identification/estimate_steering_dynamics.py eval scripts/system_identification/data/rosbag2_autoware_0.mcap --load-params scripts/system_identification/results/params.json
#   uv run scripts/system_identification/estimate_steering_dynamics.py --help
#
# Description:
#   MCAPからステアリング制御入力と車両ステータスを抽出し、FOPDT/SOPDTモデルのパラメータ推定を行います。

import argparse
import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from core.utils.mcap_utils import read_messages

def extract_data(mcap_path):
    print(f"Extracting data from {mcap_path}...")
    cmd_times = []
    cmd_vals = []
    status_times = []
    status_vals = []
    vel_times = []
    vel_vals = []
    
    cmd_topic = "/control/command/control_cmd"
    status_topic = "/vehicle/status/steering_status"
    vel_topic = "/localization/kinematic_state"
    
    for topic, msg, timestamp_ns in read_messages(mcap_path, [cmd_topic, status_topic, vel_topic]):
        t = timestamp_ns / 1e9
        if topic == cmd_topic:
            try: 
                val = msg.lateral.steering_tire_angle
                cmd_times.append(t)
                cmd_vals.append(val)
            except AttributeError: pass
        elif topic == status_topic:
            try: 
                val = msg.steering_tire_angle
                status_times.append(t)
                status_vals.append(val)
            except AttributeError: pass
        elif topic == vel_topic:
            try: 
                val = msg.twist.twist.linear.x
                vel_times.append(t)
                vel_vals.append(val)
            except AttributeError: pass

    return (np.array(cmd_times), np.array(cmd_vals), 
            np.array(status_times), np.array(status_vals),
            np.array(vel_times), np.array(vel_vals))

def simulate_fopdt(params, u_interp, t_span, u_times):
    # params: [K, tau, L]
    K, tau, L = params
    dt = np.mean(np.diff(t_span))
    y_sim = np.zeros_like(t_span)
    
    # Initial condition
    y_sim[0] = u_interp(t_span[0] - L) * K # Approximation
    
    alpha = np.exp(-dt / tau) if tau > 1e-4 else 0.0
    
    for i in range(1, len(t_span)):
        t_delayed = t_span[i] - L
        u_val = u_interp(t_delayed) if t_delayed >= u_times[0] else u_interp(u_times[0])
        
        if tau > 1e-4:
            # Discrete LPF: y[k] = alpha*y[k-1] + (1-alpha)*K*u[k]
            y_sim[i] = alpha * y_sim[i-1] + (1 - alpha) * K * u_val
        else:
            y_sim[i] = K * u_val
            
    return y_sim

def cost_fopdt(params, u_interp, t_span, y_meas, u_times):
    # Constraints can be handled by bounds in minimize, but here we penalize
    K, tau, L = params
    if tau < 0 or L < 0:
        return 1e9
    
    y_sim = simulate_fopdt(params, u_interp, t_span, u_times)
    return np.mean((y_sim - y_meas)**2)

def simulate_gain_delay(params, u_interp, t_span, u_times):
    # params: [K, L]
    K, L = params
    y_sim = np.zeros_like(t_span)
    for i, t in enumerate(t_span):
        t_delayed = t - L
        u_val = u_interp(t_delayed) if t_delayed >= u_times[0] else u_interp(u_times[0])
        y_sim[i] = K * u_val
    return y_sim

def cost_gain_delay(params, u_interp, t_span, y_meas, u_times):
    K, L = params
    if L < 0:
        return 1e9
    y_sim = simulate_gain_delay(params, u_interp, t_span, u_times)
    return np.mean((y_sim - y_meas)**2)

def simulate_sopdt(params, u_interp, t_span, u_times, max_rate=None):
    # params: [K, zeta, omega_n, L]
    K, zeta, omega_n, L = params
    dt = np.mean(np.diff(t_span))
    y_sim = np.zeros_like(t_span)
    
    # State variables: y (position), v (velocity)
    y = u_interp(t_span[0] - L) * K
    v = 0.0
    
    y_sim[0] = y
    
    # Pre-calc max delta if rate limited
    max_delta = max_rate * dt if max_rate is not None else float('inf')
    
    # Calculate discrete delay steps (Match simulator logic)
    delay_steps = max(1, int(L / dt))
    
    for i in range(1, len(t_span)):
        # Discrete delay
        t_delayed_discrete = t_span[i] - (delay_steps * dt)
        u_val = u_interp(t_delayed_discrete) if t_delayed_discrete >= u_times[0] else u_interp(u_times[0])
        
        # Dynamics: y'' + 2*zeta*wn*y' + wn^2*y = K*wn^2*u
        # v' = K*wn^2*u - 2*zeta*wn*v - wn^2*y
        target = K * u_val
        
        dv = (omega_n**2) * (target - y) - 2 * zeta * omega_n * v
        v_next = v + dv * dt
        
        # Rate Limiting (Match simulator/dynamics.py)
        delta = v_next * dt
        if abs(delta) > max_delta:
            delta = np.copysign(max_delta, delta)
            v_next = delta / dt
            
        y += delta
        v = v_next
        
        y_sim[i] = y
            
    return y_sim

def cost_sopdt(params, u_interp, t_span, y_meas, u_times):
    K, zeta, omega_n, L = params
    if zeta < 0 or omega_n < 0 or L < 0:
        return 1e9
    
    y_sim = simulate_sopdt(params, u_interp, t_span, u_times)
    return np.mean((y_sim - y_meas)**2)

import json

def save_params(params_dict, filepath):
    with open(filepath, 'w') as f:
        json.dump(params_dict, f, indent=4)
    print(f"Parameters saved to {filepath}")

def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_optimization(u_interp, status_t, status_v, cmd_t):
    # Optimize FOPDT
    x0_fopdt = [1.0, 0.1, 0.05]
    bounds_fopdt = [(0.5, 2.0), (0.01, 2.0), (0.0, 1.0)] # K, tau, L
    
    print("Optimizing FOPDT model...")
    res_fopdt = minimize(cost_fopdt, x0_fopdt, args=(u_interp, status_t, status_v, cmd_t), 
                         bounds=bounds_fopdt, method='L-BFGS-B')
    
    # Optimize SOPDT
    x0_sopdt = [1.0, 0.7, 5.0, 0.05]
    bounds_sopdt = [(0.5, 2.0), (0.1, 2.0), (0.1, 20.0), (0.0, 1.0)]
    
    print("Optimizing SOPDT model...")
    res_sopdt = minimize(cost_sopdt, x0_sopdt, args=(u_interp, status_t, status_v, cmd_t),
                         bounds=bounds_sopdt, method='L-BFGS-B')
    
    # Optimize Gain + Delay
    x0_gd = [1.0, 0.05]
    bounds_gd = [(0.5, 2.0), (0.0, 1.0)]
    print("Optimizing Gain+Delay model...")
    res_gd = minimize(cost_gain_delay, x0_gd, args=(u_interp, status_t, status_v, cmd_t),
                      bounds=bounds_gd, method='L-BFGS-B')
                      
    return {
        "fopdt": {
            "K": res_fopdt.x[0],
            "tau": res_fopdt.x[1],
            "L": res_fopdt.x[2]
        },
        "sopdt": {
            "K": res_sopdt.x[0],
            "zeta": res_sopdt.x[1],
            "omega_n": res_sopdt.x[2],
            "L": res_sopdt.x[3]
        },
        "gain_delay": {
            "K": res_gd.x[0],
            "L": res_gd.x[1]
        }
    }

def evaluate_models(models_params, u_interp, status_t, status_v, cmd_t):
    results = {}
    
    # FOPDT
    p_dict = models_params["fopdt"]
    p = [p_dict["K"], p_dict["tau"], p_dict["L"]] # Convert back to list for simulation
    y_fopdt = simulate_fopdt(p, u_interp, status_t, cmd_t)
    rmse_fopdt = np.sqrt(np.mean((y_fopdt - status_v)**2))
    results["fopdt"] = {"y": y_fopdt, "rmse": rmse_fopdt, "params": p_dict}

    # SOPDT
    p_dict = models_params["sopdt"]
    p = [p_dict["K"], p_dict["zeta"], p_dict["omega_n"], p_dict["L"]]
    y_sopdt = simulate_sopdt(p, u_interp, status_t, cmd_t, max_rate=0.937)
    rmse_sopdt = np.sqrt(np.mean((y_sopdt - status_v)**2))
    results["sopdt"] = {"y": y_sopdt, "rmse": rmse_sopdt, "params": p_dict}
    
    # Gain + Delay
    if "gain_delay" in models_params:
        p_dict = models_params["gain_delay"]
        p = [p_dict["K"], p_dict["L"]]
        y_gd = simulate_gain_delay(p, u_interp, status_t, cmd_t)
        rmse_gd = np.sqrt(np.mean((y_gd - status_v)**2))
        results["gain_delay"] = {"y": y_gd, "rmse": rmse_gd, "params": p_dict}
        
    return results

def print_results(results):
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    
    r = results["fopdt"]
    p = r["params"]
    print(f"Model: FOPDT (First Order)")
    print(f"  params: K={p['K']:.4f}, tau={p['tau']:.4f}, L={p['L']:.4f}")
    print(f"  RMSE: {r['rmse']:.6f}")
    print("-" * 20)
    
    r = results["sopdt"]
    p = r["params"]
    print(f"Model: SOPDT (Second Order)")
    print(f"  params: K={p['K']:.4f}, zeta={p['zeta']:.4f}, omega_n={p['omega_n']:.4f}, L={p['L']:.4f}")
    print(f"  RMSE: {r['rmse']:.6f}")
    print("-" * 20)
    
    if "gain_delay" in results:
        r = results["gain_delay"]
        p = r["params"]
        print(f"Model: Gain + Delay")
        print(f"  params: K={p['K']:.4f}, L={p['L']:.4f}")
        print(f"  RMSE: {r['rmse']:.6f}")

def plot_results(mcap_path, vel_t, vel_v, status_t, status_v, u_interp, results, mode_title="Identification"):
    y_fopdt = results["fopdt"]["y"]
    y_sopdt = results["sopdt"]["y"]
    rmse_fopdt = results["fopdt"]["rmse"]
    rmse_sopdt = results["sopdt"]["rmse"]
    
    # Determine output directory (same dir as this script)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    base_name = Path(mcap_path).stem
    
    # Plotly visualization
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=("Vehicle Velocity", f"Steering Angle Response ({mode_title})", "Model Error"),
                            row_heights=[0.2, 0.5, 0.3])

        # Row 1: Velocity
        if len(vel_t) > 0:
            fig.add_trace(go.Scatter(x=vel_t, y=vel_v, mode='lines', name='Velocity', line=dict(color='orange', width=2)), row=1, col=1)

        # Row 2: Steering
        fig.add_trace(go.Scatter(x=status_t, y=status_v, mode='lines', name='Measured', line=dict(color='black', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=status_t, y=u_interp(status_t), mode='lines', name='Command', line=dict(color='gray', dash='dash', width=1)), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=status_t, y=y_fopdt, mode='lines', name=f'FOPDT (RMSE={rmse_fopdt:.4f})', line=dict(color='red', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=status_t, y=y_sopdt, mode='lines', name=f'SOPDT (RMSE={rmse_sopdt:.4f})', line=dict(color='green', width=1.5)), row=2, col=1)

        # Row 3: Error
        fig.add_trace(go.Scatter(x=status_t, y=y_fopdt - status_v, mode='lines', name='Error (FOPDT)', line=dict(color='red', width=1), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=status_t, y=y_sopdt - status_v, mode='lines', name='Error (SOPDT)', line=dict(color='green', width=1), showlegend=False), row=3, col=1)

        fig.update_layout(
            title_text=f"Steering Dynamics {mode_title}",
            xaxis3_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            yaxis2_title="Steering Angle (rad)",
            yaxis3_title="Error (rad)",
            height=900,
            hovermode="x unified"
        )
        
        # output_html = output_dir / f"{base_name}.{mode_title.lower()}.html"
        # fig.write_html(str(output_html))
        # print(f"\nInteractive plot saved to {output_html}")
        
    except ImportError:
        print("\nPlotly not found. Skipping interactive plot.")
        
    # Matplotlib fallback
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3, 1.5]})
    
    # Top: Velocity
    if len(vel_t) > 0:
        ax1.plot(vel_t, vel_v, 'orange', label='Velocity')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.legend()
        ax1.grid(True)
    
    # Middle: Steering
    ax2.plot(status_t, status_v, 'k-', label='Measured', alpha=0.6)
    ax2.plot(status_t, u_interp(status_t), 'k--', label='Command', alpha=0.3)
    ax2.plot(status_t, y_fopdt, 'r-', label=f'FOPDT (RMSE={rmse_fopdt:.4f})')
    ax2.plot(status_t, y_sopdt, 'g-', label=f'SOPDT (RMSE={rmse_sopdt:.4f})')
    
    ax2.legend()
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title(f"Steering Dynamics {mode_title}")
    ax2.grid(True)
    
    # Bottom: Error
    ax3.plot(status_t, y_fopdt - status_v, 'r-', label='Error (FOPDT)')
    ax3.plot(status_t, y_sopdt - status_v, 'g-', label='Error (SOPDT)')
    ax3.set_ylabel('Error (rad)')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    
    plt.tight_layout()
    
    output_png = output_dir / f"{base_name}.{mode_title.lower()}.png"
    plt.savefig(output_png)
    print(f"Static plot saved to {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Estimate or Evaluate steering dynamics.")
    subparsers = parser.add_subparsers(dest='command', help='Mode: train or eval')
    
    # Train Parser
    parser_train = subparsers.add_parser('train', help='Estimate parameters from MCAP')
    parser_train.add_argument('file', help='Input MCAP file')
    parser_train.add_argument('--save-params', help='Path to save estimated parameters (JSON). Defaults to output dir.')
    
    # Eval Parser
    parser_eval = subparsers.add_parser('eval', help='Evaluate existing parameters on MCAP')
    parser_eval.add_argument('file', help='Input MCAP file')
    parser_eval.add_argument('--load-params', required=True, help='Path to load parameters from (JSON)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    mcap_path = args.file
    
    # 1. Extract Data
    cmd_t, cmd_v, status_t, status_v, vel_t, vel_v = extract_data(mcap_path)
    if len(cmd_t) == 0 or len(status_t) == 0:
        print("Error: No steering data found.")
        return

    # Normalize time
    t0 = min(cmd_t[0], status_t[0])
    if len(vel_t) > 0:
        t0 = min(t0, vel_t[0])
    cmd_t -= t0
    status_t -= t0
    if len(vel_t) > 0:
        vel_t -= t0
    
    # Sort
    idx = np.argsort(cmd_t)
    cmd_t = cmd_t[idx]
    cmd_v = cmd_v[idx]
    
    # Prepend a point before t=0 to represent the "initial state" avoid backward extrapolation of step input
    # We assume the command before simulation start was equal to the initial steering status (or 0)
    # This captures the "Step" nature of the input at t=0
    pre_t = cmd_t[0] - 1.0
    pre_v = status_v[0] if len(status_v) > 0 else 0.0
    
    cmd_t = np.insert(cmd_t, 0, pre_t)
    cmd_v = np.insert(cmd_v, 0, pre_v)
    
    u_interp = interp1d(cmd_t, cmd_v, kind='linear', fill_value=(pre_v, cmd_v[-1]), bounds_error=False)
    
    models_params = {}
    
    if args.command == 'train':
        print("\n--- Training Mode ---")
        models_params = run_optimization(u_interp, status_t, status_v, cmd_t)
        
        save_path = args.save_params
        if not save_path:
             output_dir = Path(__file__).parent / "results"
             output_dir.mkdir(exist_ok=True)
             save_path = output_dir / "params.json"

        save_params(models_params, save_path)
            
        results = evaluate_models(models_params, u_interp, status_t, status_v, cmd_t)
        print_results(results)
        plot_results(mcap_path, vel_t, vel_v, status_t, status_v, u_interp, results, mode_title="Training")
        
    elif args.command == 'eval':
        print("\n--- Evaluation Mode ---")
        if not Path(args.load_params).exists():
            print(f"Error: Parameter file not found: {args.load_params}")
            sys.exit(1)
            
        models_params = load_params(args.load_params)
        print(f"Loaded parameters from {args.load_params}")
        
        results = evaluate_models(models_params, u_interp, status_t, status_v, cmd_t)
        print_results(results)
        plot_results(mcap_path, vel_t, vel_v, status_t, status_v, u_interp, results, mode_title="Evaluation")

if __name__ == "__main__":
    main()
