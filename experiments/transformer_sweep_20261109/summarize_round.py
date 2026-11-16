"""Summarize successful production runs; preserve historical-comparison limitations."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
audit = json.loads((here/'remote_status_latest.json').read_text(encoding='utf-8-sig'))
plan = json.loads((here/'experiment_plan.json').read_text(encoding='utf-8-sig'))
observables = ['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
runs = {}
for item in audit['runs']:
    if item['run_id'].startswith('smoke_') or item.get('status',{}).get('status') != 'complete':
        continue
    if not item.get('metrics') or not item.get('prediction_file_bytes'):
        raise ValueError('Incomplete artifacts: '+item['run_id'])
    if item['run_id'] in runs:
        raise ValueError('Duplicate successful configuration: '+item['run_id'])
    runs[item['run_id']] = item
assert set(runs) == {x['run_id'] for x in plan['runs']}
assert {r['status']['train']['rows'] for r in runs.values()} == {4692288}
assert {r['status']['val']['rows'] for r in runs.values()} == {260190}
for split in ['train','val']:
    assert len({r['status'][split]['sha256'] for r in runs.values()}) == 1
best = {}
for group, selected in [('all',list(runs)),('regression',[x for x in runs if x.startswith('regression_')]),('part',[x for x in runs if x.startswith('part_')])]:
    best[group] = {}
    for obs in observables:
        best[group][obs] = {}
        for metric in ['mae','rmse','w1']:
            name = min(selected,key=lambda x:runs[x]['metrics'][obs][metric])
            best[group][obs][metric] = {'run_id':name,'value':runs[name]['metrics'][obs][metric]}
historical_path = here.parent/'本轮新结果_20261109/02_评估结果/metrics.json'
historical = json.loads(historical_path.read_text(encoding='utf-8'))['metrics']
record = dict(checked_utc=audit['checked_utc'],completed=26,architectures={'regression':13,'part':13},
    train_events=4692288,validation_events=260190,test_evaluated=False,
    successful_gpu_job_hours=sum(r['status']['elapsed_seconds'] for r in runs.values())/3600,
    best_by_metric=best,runs=runs,historical_metrics_as_reported=historical,
    limitations=[
        'Single seed; training time capped, not equal achieved epochs or convergence.',
        'Historical results used a slightly different numerical truth representation. Cross-round numbers are preliminary until all predictions are evaluated against identical truth.',
        'Per-observable minima can belong to different models; they do not describe one achievable combined model.',
        'Warmup/cosine learning rate reaches zero at its fixed 20-epoch horizon, even if more epochs execute.',
        'Historical failed attempts retained running in status.json; only completed replacement results are included here.',
    ])
(here/'round1_summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines = [
    '第一轮 Transformer 实验汇总',
    '检查时间 UTC：'+audit['checked_utc'],
    '26/26 成功：普通 Transformer 13 组，ParT 13 组。',
    '全量训练 4,692,288 事件；验证 260,190 事件；尚未做测试集评估。',
    '成功任务合计 %.3f GPU-job 小时（不包含失败尝试和排队）。'%record['successful_gpu_job_hours'],
    '15 个原失败任务因 node872 的不可纠正 ECC 错误，排除该节点后补跑全部成功。',
    '', '各观测量分别最优（不同指标可能对应不同模型）',
]
for obs in observables:
    mae=best['all'][obs]['mae']; w1=best['all'][obs]['w1']
    lines.append('%s: MAE %.8f [%s]; W1 %.8f [%s]'%(obs,mae['value'],mae['run_id'],w1['value'],w1['run_id']))
lines += ['', '历史结果的初步数值对照',
    'ParT + 0.1 EMD 的质量 MAE 为 46.31923 GeV；旧 MSE 为 46.53418 GeV，旧 MSE+0.1EMD 为 46.75304 GeV。',
    '其余三个观测量的本轮最佳 MAE 均未超过旧的两个基线。',
    '新旧训练预算不同、仅一个种子；truth 有微小数值差异，需统一真值重算后再作正式比较，不能据此宣称显著改善。',
    '本轮没有在逐事件误差和所有边缘分布指标上同时占优的配置。',
    '', '全部配置的验证指标（MAE / W1）']
for cfg in plan['runs']:
    item=runs[cfg['run_id']]
    pairs=['%s=%.6f/%.6f'%(o,item['metrics'][o]['mae'],item['metrics'][o]['w1']) for o in observables]
    lines.append(cfg['run_id']+' | '+' | '.join(pairs))
lines += ['', '后续比较注意：warmup/cosine 在固定 20 epoch 后学习率归零，这两组不适合直接据此判断学习率调度无用。',
    '来源：remote_status_latest.json；历史来源：本轮新结果_20261109/02_评估结果/metrics.json。']
(here/'round1_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
print('Saved round1_summary.json and round1_summary.txt; 26 unique successful configurations verified.')
