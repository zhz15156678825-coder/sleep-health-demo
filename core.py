"""Offline demonstration only. No diagnosis, causal inference or private mental inference."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 20261028
FEATURES = ['sleep_hours', 'efficiency', 'awakenings', 'stress', 'energy', 'sleep_mean3', 'energy_mean3']
LABELS = {'sleep_hours': '当日睡眠时长（小时）', 'efficiency': '当日睡眠效率（%）',
          'awakenings': '当日夜醒次数', 'stress': '当日主观压力（0–10）',
          'energy': '当日主观精力（0–10）', 'sleep_mean3': '截至当日3天平均睡眠',
          'energy_mean3': '截至当日3天平均精力'}
CDC = 'https://www.cdc.gov/sleep/about/index.html'
NIH = 'https://www.nhlbi.nih.gov/health/insomnia/treatment'
NOTICE = '全部记录均为固定种子生成的模拟数据；预测、状态推演和实验指标仅用于软件演示，不能用于诊断、治疗或证明真实健康效果。'


def generate_data(seed=SEED):
    rng = np.random.default_rng(seed)
    records = []
    for user in range(1, 13):
        baseline = rng.uniform(6.1, 8.0)
        mood = rng.uniform(-0.5, 0.5)
        prev_energy = 6.0
        for date in pd.date_range('2026-09-01', periods=30):
            stress = float(np.clip(rng.normal(4.5, 1.7), 0, 10))
            bedtime = float(np.clip(rng.normal(23.7 + (user % 3) * .3, .65), 21, 27))
            hours = float(np.clip(baseline - .11 * stress + .3 * (date.dayofweek >= 5) + rng.normal(0, .6), 3.5, 10))
            awake = int(rng.poisson(1.5))
            efficiency = float(np.clip(95 - awake * 2.2 + rng.normal(0, 2), 65, 99))
            # Synthetic autoregressive energy; this is a data-generating assumption, not clinical truth.
            energy = float(np.clip(1.8 + .45 * hours + .3 * prev_energy - .2 * stress - .12 * awake + mood + rng.normal(0, .55), 0, 10))
            records.append(dict(user_id=f'U{user:02}', date=date, sleep_hours=round(hours, 2),
                                bedtime_hour=round(bedtime, 2), efficiency=round(efficiency, 1),
                                awakenings=awake, stress=round(stress, 1), energy=round(energy, 1),
                                synthetic=True))
            prev_energy = energy
    return pd.DataFrame(records)


def make_features(data):
    d = data.sort_values(['user_id', 'date']).copy()
    d['date'] = pd.to_datetime(d.date)
    group = d.groupby('user_id', sort=False)
    for source, name in [('sleep_hours', 'sleep_mean3'), ('energy', 'energy_mean3')]:
        d[name] = group[source].transform(lambda s: s.rolling(3, min_periods=1).mean())
    d['target'] = group.energy.shift(-1)
    d['target_date'] = group.date.shift(-1)
    return d


def fit_model(data):
    """Fixed chronological holdout. All preprocessing fits training rows only."""
    features = make_features(data)
    pairs = features.dropna(subset=['target', 'target_date'])
    # Labels through Sep 22 train; Sep 23–30 hold out. Feature date <= label date - 1.
    train = pairs[pairs.target_date <= pd.Timestamp('2026-09-22')].copy()
    test = pairs[pairs.target_date > pd.Timestamp('2026-09-22')].copy()
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
    pipe.fit(train[FEATURES], train.target)
    prediction = pipe.predict(test[FEATURES])
    test['prediction'] = prediction
    test['persistence'] = test.energy
    test['train_mean'] = float(train.target.mean())
    metrics = []
    for name, column in [('Ridge（完整）', 'prediction'), ('沿用当日精力', 'persistence'), ('训练集均值', 'train_mean')]:
        metrics.append({'模型': name, 'MAE': mean_absolute_error(test.target, test[column]),
                        'RMSE': np.sqrt(mean_squared_error(test.target, test[column])), '测试样本': len(test)})
    # Ablation uses identical split, preprocessing and fixed alpha.
    no_energy = [x for x in FEATURES if x not in ('energy', 'energy_mean3')]
    ablation = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
    ablation.fit(train[no_energy], train.target)
    ap = ablation.predict(test[no_energy])
    metrics.append({'模型': 'Ridge（移除历史精力）', 'MAE': mean_absolute_error(test.target, ap),
                    'RMSE': np.sqrt(mean_squared_error(test.target, ap)), '测试样本': len(test)})
    return dict(pipeline=pipe, model=pipe.named_steps['ridge'], train=train, test=test,
                features=features, metrics=pd.DataFrame(metrics))


def explain(result, row):
    pipe = result['pipeline']
    x = pd.DataFrame([row[FEATURES]], columns=FEATURES)
    z = pipe.named_steps['scaler'].transform(pipe.named_steps['imputer'].transform(x))
    contributions = z[0] * result['model'].coef_
    terms = pd.DataFrame({'特征': [LABELS[x] for x in FEATURES], '原始值': x.iloc[0].to_numpy(),
                          '标准化系数': result['model'].coef_, 'contribution': contributions})
    return float(pipe.predict(x)[0]), terms


def default_state(user_id):
    return dict(user_id=user_id, goal='规律作息', preference='简短提醒', available_minutes=15,
                night_shift=False, stress=4, belief='尚不清楚哪些习惯与精力相关',
                intention='先记录，再选择一个小行动', emotion='未报告',
                social_constraint='无已报告约束', source='默认演示值，待用户确认', version=0)


def safety_check(text):
    # Keyword routing is deliberately conservative and cannot perform clinical triage.
    emergency = ['胸痛', '呼吸困难', '昏厥', '开车', '驾驶', '自伤', '自杀']
    clinical = ['安眠药', '剂量', '用药', '停药', '褪黑素', '诊断', '呼吸暂停', '窒息', '失眠', '打鼾', '连续睡不着']
    hits = [word for word in emergency + clinical if word in text]
    urgent = any(word in text for word in emergency)
    if urgent:
        message = '此输入涉及可能的即时安全风险。若困倦，请停止驾驶或操作设备；若有急性身体危险或可能伤害自己的风险，请立即联系当地急救或可信任的人寻求现场帮助。平台不能评估紧急程度。'
    elif hits:
        message = '此问题超出生活习惯演示范围：不提供诊断、药物或剂量建议。持续睡眠困扰、明显日间困倦或夜间呼吸异常，请咨询合格医疗人员。'
    else:
        message = '仅提供一般生活习惯信息。关键词检查可能漏检，不能排除疾病或判断紧急程度。'
    return dict(blocked=bool(hits), hits=hits, urgent=urgent, message=message)


def run_agents(history, state, query='', use_context=True):
    """Three deterministic logical agents; emits inspectable artifacts, not hidden reasoning."""
    state = deepcopy(state)
    latest = history.sort_values('date').tail(7)
    physical = dict(window_days=len(latest), sleep_mean=round(float(latest.sleep_hours.mean()), 2),
                    energy_mean=round(float(latest.energy.mean()), 2),
                    sleep_variability=round(float(latest.sleep_hours.std(ddof=0)), 2),
                    as_of=str(pd.to_datetime(latest.date.max()).date()), source='模拟睡眠记录')
    safety = safety_check(query)
    # Physical and mental carriers are deliberately explicit.
    candidates = [
        dict(id='routine', action='选择可坚持的睡前与起床时间，先记录一周。', minutes=10,
             mental_action='支持规律作息的意图', goal='规律作息', source=CDC),
        dict(id='winddown', action='睡前安排短暂安静放松，减少电子设备干扰。', minutes=5,
             mental_action='降低执行负担，尊重放松偏好', goal='减轻睡前负担', source=CDC),
        dict(id='log', action='今天先记录精力与睡眠感受，等条件允许再选择习惯调整。', minutes=0,
             mental_action='保留自主选择权，不推断未知心理状态', goal='了解精力变化', source='工程演示规则 R03'),
    ]
    evaluated = []
    for item in candidates:
        branch = deepcopy(item)
        veto = bool(safety['blocked'])
        reason = []
        if use_context and item['minutes'] > state['available_minutes']:
            veto = True
            reason.append('超过用户可用时间')
        if use_context and state['night_shift'] and item['id'] == 'routine':
            veto = True
            reason.append('夜班情境下不直接套用常规日间作息')
        # Demonstration ordinal scores, not learned utilities, probabilities or health gains.
        physical_score = 3 if item['id'] == 'routine' else 2
        if physical['sleep_variability'] > .7 and item['id'] == 'routine':
            physical_score = 4
        mental_score = 2
        social_score = 2
        if use_context:
            mental_score += 3 * (state['goal'] == item['goal'])
            mental_score += int(state['preference'] == '简短提醒' and item['id'] == 'winddown')
            mental_score += int(state['stress'] >= 7 and item['id'] == 'winddown')
            social_score = 3 if item['minutes'] <= state['available_minutes'] else 0
        if safety['blocked']:
            reason.append('安全规则优先，转人工咨询')
        branch.update(physical_score=physical_score, mental_score=mental_score, social_score=social_score,
                      veto=veto, reason='；'.join(reason) or '符合演示约束',
                      score=0 if veto else physical_score + mental_score + social_score,
                      next_state_hypothesis=dict(physical='若用户采纳，将新增一条行动记录；睡眠变化未知。',
                                                 mental=f"可能形成意图：{item['mental_action']}；需用户确认，实际情绪变化未知。"))
        evaluated.append(branch)
    eligible = [x for x in evaluated if not x['veto']]
    selected = sorted(eligible, key=lambda x: (-x['score'], x['id']))[0] if eligible else None
    observation = dict(physical=physical, mental=state if use_context else '移除心理情境，仅保留安全检查',
                       unknown=['未报告情绪不作推断', '未知的他人信念与关系', '真实行动后的健康效果'])
    trace = [
        dict(step=1, agent='数据分析智能体', inputs={'rows': len(history), 'user': state['user_id']},
             outputs=physical, depends_on=[], rule='D01：仅按已知日期聚合最近7天'),
        dict(step=2, agent='知识规则智能体', inputs={'query': query, 'analysis_step': 1},
             outputs={'safety': safety, 'candidate_ids': [x['id'] for x in candidates]}, depends_on=[1],
             rule='K01：生活习惯规则；K02：安全否决优先；来源 CDC / 工程规则'),
        dict(step=3, agent='情境反馈智能体', inputs={'observation': observation, 'knowledge_step': 2},
             outputs={'branches': evaluated, 'selected_id': selected['id'] if selected else None}, depends_on=[1, 2],
             rule='C01：三项分数直接相加；安全/时间/夜班否决；同分按ID排序')]
    payload = dict(user_id=state['user_id'], state_version=state['version'], query=query, use_context=use_context,
                   observation=observation, safety=safety, candidates=evaluated, selected=selected, trace=trace)
    payload['run_id'] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    payload['notice'] = NOTICE
    return payload


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    from pathlib import Path
    dest = Path(__file__).parent / 'data'
    dest.mkdir(exist_ok=True)
    generate_data().to_csv(dest / 'sleep_demo.csv', index=False, encoding='utf-8-sig')
    (dest / 'profiles_demo.json').write_text(dumps({f'U{i:02}': default_state(f'U{i:02}') for i in range(1, 13)}), encoding='utf-8')
    print('已生成12名模拟用户 × 连续30天，共360行；种子', SEED)
