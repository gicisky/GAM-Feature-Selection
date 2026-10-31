#!/usr/bin/env python3
"""GAM 复验守护进程：定时健康检查、异常自动重启、增量备份、动态 ETA。

不修改 source/数据/结果；只读 status + 检测进程 + rsync 备份 + 必要时重启协调器。
"""
import json, os, subprocess, time, sys, fcntl
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT.parent.parent / 'results' / 'revalidation_v2'
PY = '/opt/miniconda3/bin/python3'
CHECK_INTERVAL = 120      # 健康检查间隔（秒）
BACKUP_INTERVAL = 600     # 备份间隔（秒）
RESTART_WINDOW = 1800     # 30 分钟内
MAX_RESTARTS = 3          # 连续重启上限，超过则停止自动重启
LOG = ROOT / 'monitor_daemon.log'

ONE_FOLD = {'E7', 'lambda', 'convergence', 'independent', 'filter_guided'}

def log(msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

def fold_count():
    n = 0
    for d in (ROOT / 'results', ROOT / 'component_ablation' / 'results'):
        if d.exists():
            n += len(list(d.glob('*_fold*.json')))
    return n

def task_done_count():
    n = 0
    for d in (ROOT / 'results', ROOT / 'component_ablation' / 'results'):
        if d.exists():
            for p in d.glob('*.json'):
                if '_fold' not in p.name and 'error' not in p.name:
                    n += 1
    return n

def error_count():
    n = 0
    for d in (ROOT / 'results', ROOT / 'component_ablation' / 'results'):
        if d.exists():
            n += len(list(d.glob('*.error.json')))
    return n

def total_folds():
    plan = read_json(ROOT / 'plan.json')
    if isinstance(plan, list):
        n = sum(1 if t.get('experiment') in ONE_FOLD else 5 for t in plan)
    else:
        n = 0
    n += 60 * 5  # component ablation 60 tasks x 5 folds
    return n

def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False

def lock_free():
    try:
        f = open(ROOT / 'run.lock', 'w')
    except OSError:
        return True
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        f.close()

def restart():
    if not lock_free():
        log("  ⚠️ run.lock 仍被占用，可能有其他实例，跳过重启")
        return False
    log("  🔄 启动 run_fast.py --workers 10")
    try:
        logfile = open(ROOT / 'fast_revalidation.log', 'a')
        subprocess.Popen([PY, str(ROOT / 'run_fast.py'), '--workers', '10'],
                         stdout=logfile, stderr=subprocess.STDOUT,
                         cwd=str(ROOT), start_new_session=True)
        return True
    except Exception as e:
        log(f"  ❌ 重启失败: {e}")
        return False

def backup():
    try:
        BACKUP.mkdir(parents=True, exist_ok=True)
        for src, name in [(ROOT / 'results', 'results'),
                          (ROOT / 'component_ablation' / 'results', 'component_results')]:
            dst = BACKUP / name
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', '--delete', str(src) + '/', str(dst) + '/'],
                           capture_output=True, timeout=120)
        # 备份关键元数据
        for f in ['plan.json', 'status.json', 'execution_manifest.json',
                  'component_ablation/manifest.json', 'retained_unchanged_bare.jsonl']:
            p = ROOT / f
            if p.exists():
                subprocess.run(['cp', str(p), str(BACKUP / p.name)], capture_output=True)
        log(f"  💾 增量备份完成 → {BACKUP}")
    except Exception as e:
        log(f"  ❌ 备份失败: {e}")

def eta_str(remaining_folds, rate_per_min):
    if rate_per_min <= 0:
        return "速率待定（尚无新折）"
    minutes = remaining_folds / rate_per_min
    eta = datetime.now() + timedelta(minutes=minutes)
    return f"{minutes/60:.1f} 小时后（约 {eta.strftime('%m-%d %H:%M')}）"

def main():
    log("=" * 60)
    log("守护进程启动：健康检查 + 自动重启 + 增量备份")
    last_folds = fold_count()
    last_errors = error_count()
    restarts = 0
    last_restart_ts = 0.0
    last_backup_ts = time.time()
    total = total_folds()
    # 立即做一次备份，确保初始状态有副本
    backup()
    last_backup_ts = time.time()
    log(f"初始: 总折={total} 已完成折={last_folds} 剩余折={total-last_folds} 已完成任务={task_done_count()} 错误={last_errors}")
    while True:
        time.sleep(CHECK_INTERVAL)
        status = read_json(ROOT / 'status.json')
        pid = status.get('pid')
        state = status.get('state', 'unknown')
        folds = fold_count()
        errors = error_count()
        delta = folds - last_folds

        # 完成判断
        if state == 'complete' and not alive(pid):
            log("✅ 全部实验完成（state=complete 且协调器已退出），守护进程退出")
            backup()
            break

        # 1) 协调器健康
        if not alive(pid):
            log(f"⚠️ 协调器 pid={pid} 停止 (state={state}, 错误={errors})")
            now = time.time()
            if now - last_restart_ts < RESTART_WINDOW and restarts >= MAX_RESTARTS:
                msg = (f"{datetime.now()}\n协调器在 {RESTART_WINDOW//60} 分钟内重启 {restarts} 次，已停止自动重启。"
                       f"\n请检查 fast_revalidation.log 与 *.error.json。")
                (ROOT / 'MANUAL_ATTENTION_REQUIRED.txt').write_text(msg)
                log("🛑 达到重启上限，停止自动重启，需人工介入（已写 MANUAL_ATTENTION_REQUIRED.txt）")
                break
            restart()
            restarts += 1
            last_restart_ts = now
            time.sleep(30)
            continue

        # 2) 错误文件告警
        if errors > last_errors:
            log(f"⚠️ 错误文件从 {last_errors} 增至 {errors}")
            for d in (ROOT / 'results', ROOT / 'component_ablation' / 'results'):
                if d.exists():
                    for e in d.glob('*.error.json'):
                        log(f"   error: {e.name}")

        # 3) 备份
        if time.time() - last_backup_ts >= BACKUP_INTERVAL:
            backup()
            last_backup_ts = time.time()

        # 4) 进度 + ETA
        rate = delta / (CHECK_INTERVAL / 60.0)  # 折/分钟
        remaining = total - folds
        log(f"进度: 折={folds}(+{delta}) 任务={task_done_count()} 错误={errors} | "
            f"速率≈{rate:.1f}折/分 | 预计 {eta_str(remaining, rate)}")

        last_folds = folds
        last_errors = errors

if __name__ == '__main__':
    main()
