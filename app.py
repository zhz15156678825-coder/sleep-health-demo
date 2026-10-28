from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import altair as alt
import pandas as pd
import streamlit as st
import core

st.set_page_config(page_title='眠境 · 睡眠健康研究原型', page_icon='🌙', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1350px}
[data-testid="stMetric"]{background:white;border:1px solid #deebe9;border-radius:14px;padding:16px}
h1,h2,h3{letter-spacing:-.02em} .stCaption{color:#56717a}
</style>''', unsafe_allow_html=True)


@st.cache_data
def load_data():
    path = Path(__file__).parent / 'data' / 'sleep_demo.csv'
    return pd.read_csv(path, parse_dates=['date']) if path.exists() else core.generate_data()


@st.cache_resource
def train():
    return core.fit_model(load_data())


data = load_data()
result = train()
if 'profiles' not in st.session_state:
    st.session_state.profiles = {u: core.default_state(u) for u in sorted(data.user_id.unique())}
    st.session_state.runs = {}
    st.session_state.audit = []

with st.sidebar:
    st.title('🌙 眠境')
    st.caption('SLEEP CONTEXT LAB · 离线研究原型')
    user = st.selectbox('模拟用户', sorted(data.user_id.unique()))
    page = st.radio('工作台', ['睡眠仪表盘', '预测与解释', '情境与智能体', '实验对照', '数据与理论'])
    st.divider()
    st.caption('12 位模拟用户 / 连续 30 天\n\n2026.09.01 — 09.30\n\n种子：20261028')
    st.info('本地运行 · 无付费 API\n\n所有状态默认只保留在当前浏览器会话，可下载导出。')

history = data[data.user_id == user].sort_values('date')
state = st.session_state.profiles[user]
st.caption(f'眠境研究工作台 / {user} / {page}')
st.title(page)
st.warning(core.NOTICE, icon='⚠️')


if page == '睡眠仪表盘':
    st.write('从睡眠记录到个体情境，查看过去 30 天的模拟变化。')
    a, b, c, d = st.columns(4)
    last = history.iloc[-1]
    a.metric('近7天睡眠 / 小时', f'{history.tail(7).sleep_hours.mean():.1f}')
    b.metric('最近精力 / 10', f'{last.energy:.1f}', f'{last.energy - history.iloc[-2].energy:+.1f} 较前一天')
    c.metric('近7天睡眠效率', f'{history.tail(7).efficiency.mean():.1f}%')
    d.metric('记录完整度', '30 / 30 天')
    span = st.slider('展示最近天数', 7, 30, 30)
    view = history.tail(span)
    l, r = st.columns(2)
    with l:
        st.subheader('睡眠时长趋势')
        chart = alt.Chart(view).mark_line(point=True, color='#168777').encode(
            x=alt.X('date:T', title='日期'), y=alt.Y('sleep_hours:Q', title='小时', scale=alt.Scale(zero=False)),
            tooltip=['date:T', 'sleep_hours:Q', 'user_id:N']).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
    with r:
        st.subheader('主观精力与压力')
        long = view.melt(id_vars=['date'], value_vars=['energy', 'stress'], var_name='指标', value_name='分数')
        long['指标'] = long['指标'].map({'energy': '精力', 'stress': '压力'})
        st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X('date:T', title='日期'), y=alt.Y('分数:Q', scale=alt.Scale(domain=[0, 10])),
            color='指标:N', tooltip=['date:T', '指标:N', '分数:Q']).properties(height=280), use_container_width=True)
    st.caption('夜晚按起床日期归档。bedtime_hour=24.5 表示该夜跨午夜后的 00:30；这些字段不是穿戴设备实测。')
    with st.expander('查看当前用户原始记录'):
        st.dataframe(history, hide_index=True, use_container_width=True)
        st.download_button('下载该用户 CSV', history.to_csv(index=False).encode('utf-8-sig'), f'{user}_sleep.csv', 'text/csv')

elif page == '预测与解释':
    st.write('任务：截至第 t 天晚间的睡眠、压力与精力记录 → 预测第 t+1 天精力评分。')
    st.info('所有用户按目标日期统一切分：09.02–09.22 训练（252 条），09.23–09.30 测试（96 条）。缺失值填补与标准化仅在训练集拟合；Ridge α=10 事先固定，没有使用测试集调参。')
    mode = st.radio('解释对象', ['测试日期回顾', '下一天演示预测'], horizontal=True)
    if mode == '测试日期回顾':
        options = result['test'][result['test'].user_id == user]
        target = st.selectbox('目标日期', options.target_date.dt.strftime('%Y-%m-%d').tolist())
        row = options[options.target_date == pd.Timestamp(target)].iloc[0]
        truth = float(row.target)
    else:
        row = result['features'][result['features'].user_id == user].iloc[-1]
        truth = None
        st.caption('使用 09.30 记录预测 10.01；没有 10.01 真实标签，不能计算该次误差。模型仍只使用截至 09.22 的训练标签。')
    prediction, terms = core.explain(result, row)
    a, b, c = st.columns(3)
    a.metric('预测精力 / 10', f'{prediction:.2f}')
    b.metric('模拟观察值', '未知' if truth is None else f'{truth:.2f}')
    c.metric('训练集中心截距', f"{result['model'].intercept_:.2f}")
    st.caption('展示原始回归输出，不截断以保持加和解释；若超出 0–10 则属于模型越界。未计算经过校准的个体置信区间。')
    st.subheader('本次预测如何组成')
    st.write('预测 = 训练集中心截距 + 各项贡献。正负贡献仅表示模型内部的加减关系，不代表因果效果。')
    st.altair_chart(alt.Chart(terms).mark_bar().encode(
        y=alt.Y('特征:N', sort='-x'), x=alt.X('contribution:Q', title='对精力预测的贡献（分）'),
        color=alt.condition(alt.datum.contribution >= 0, alt.value('#168777'), alt.value('#dc8969')),
        tooltip=['特征:N', '原始值:Q', 'contribution:Q']).properties(height=260), use_container_width=True)
    st.dataframe(terms.rename(columns={'contribution': '局部贡献'}), hide_index=True, use_container_width=True)
    st.subheader('独立留出集上的模拟误差')
    st.dataframe(result['metrics'], hide_index=True, use_container_width=True)
    st.caption('测试期内允许使用当日已经观察到的精力，属于逐日单步预测；不是从 09.22 一次预测未来八天。相同用户出现在训练与测试中，结果不代表对新用户的泛化。')

elif page == '情境与智能体':
    st.write('先确认自己的目标与约束，再比较可执行的反馈。这里的心理变量是用户声明或演示默认值。')
    with st.form(f'state_{user}'):
        l, r = st.columns(2)
        with l:
            goals = ['规律作息', '减轻睡前负担', '了解精力变化']
            goal = st.selectbox('目标', goals, index=goals.index(state['goal']))
            preference = st.selectbox('反馈偏好', ['简短提醒', '详细解释'], index=int(state['preference'] == '详细解释'))
            belief = st.text_input('信念 / 对睡眠的当前理解', state['belief'], key=f'belief_{user}', max_chars=300)
            intention = st.text_input('当前意图', state['intention'], key=f'intent_{user}', max_chars=300)
            social = st.text_input('社会 / 家庭约束', state['social_constraint'], key=f'social_{user}', max_chars=300)
        with r:
            minutes = st.slider('今晚可用时间（分钟）', 0, 60, state['available_minutes'])
            stress = st.slider('当前主观压力', 0, 10, state['stress'])
            shift = st.checkbox('当前为夜班情境', state['night_shift'])
            emotion = st.text_input('情绪（可写“未报告”）', state['emotion'], key=f'emotion_{user}', max_chars=100)
        if st.form_submit_button('保存状态'):
            before = deepcopy(state)
            state = dict(user_id=user, goal=goal, preference=preference, available_minutes=minutes,
                         night_shift=shift, stress=stress, belief=belief, intention=intention,
                         emotion=emotion, social_constraint=social, source='用户明确填写', version=state['version'] + 1)
            st.session_state.profiles[user] = state
            st.session_state.audit.append({'user_id': user, 'time_utc': datetime.now(timezone.utc).isoformat(),
                                           'event': 'state_update', 'before': before, 'after': deepcopy(state)})
            st.success(f"已保存 {user} 的状态版本 {state['version']}")
    with st.expander('联合状态与部分可见信息', expanded=False):
        st.json({'mental_state': state, 'physical_state': history.tail(1).to_dict('records')[0],
                 'observation_boundary': '只使用该用户模拟记录及明确输入；不读取其他用户状态；不推断他人心理。',
                 'unknown': '未知的情绪、信念及真实干预结果保持未知。'})
    query = st.text_area('情境输入或问答', placeholder='例如：今晚加班，只有5分钟，能先做什么？或：为什么建议记录精力？', key=f'query_{user}', max_chars=2000)
    st.caption('离线结构化问答：按关键词路由安全提示，并以已保存的目标/时间/压力/夜班状态比较建议。自由文本不会自动修改状态；时间、偏好等变化请在上方保存。')
    if st.button('运行三个智能体', key=f'run_{user}', type='primary'):
        run = core.run_agents(history, state, query)
        run['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
        st.session_state.runs.setdefault(user, []).append(run)
    runs = st.session_state.runs.get(user, [])
    if runs:
        run = runs[-1]
        if run['state_version'] != state['version'] or run['query'] != query:
            st.warning('下方为历史运行，状态或输入已改变。再次运行以更新建议。')
        st.caption(f"运行 {run['run_id']} · 状态版本 {run['state_version']} · 用户 {run['user_id']}")
        if run['safety']['blocked']:
            st.error(run['safety']['message'])
        elif run['selected']:
            st.success('建议：' + run['selected']['action'])
            if run['observation']['mental']['preference'] == '详细解释':
                st.write(f"原因：{run['selected']['reason']}；目标匹配与可用时间计入评分。评分为人工规则，不是效果概率。")
            st.info(run['safety']['message'])
        st.subheader('候选反馈比较')
        table = pd.DataFrame(run['candidates'])[['id', 'action', 'minutes', 'physical_score', 'mental_score', 'social_score', 'score', 'veto', 'reason']]
        st.dataframe(table.rename(columns={'id': '候选', 'action': '行动', 'minutes': '耗时', 'physical_score': '物理合理性',
                                          'mental_score': '心理一致性', 'social_score': '情境适宜性', 'score': '总分', 'veto': '否决', 'reason': '理由'}), hide_index=True, use_container_width=True)
        st.caption('每项为工程设定的序数分：总分=三项之和；否决后总分归零且不可选，同分按ID字典序。文本社会约束仅供人工审阅，自动约束检查只覆盖可用时间与夜班。')
        st.subheader('可追溯协同流程')
        for step in run['trace']:
            with st.expander(f"{step['step']} · {step['agent']} · {step['rule']}"):
                st.json(step)
        st.download_button('导出本次完整协同记录 JSON', core.dumps(run), f"{user}_{run['run_id']}.json", 'application/json')
        with st.form(f'feedback_{user}'):
            choice = st.selectbox('你的反馈', ['愿意尝试', '暂不采纳', '需要调整'])
            note = st.text_input('补充说明', max_chars=500)
            if st.form_submit_button('记录反馈'):
                st.session_state.audit.append(dict(user_id=user, event='feedback', run_id=run['run_id'],
                                                   choice=choice, note=note, time_utc=datetime.now(timezone.utc).isoformat()))
                st.success('已记录自报反馈；不会将“愿意尝试”当成已执行或健康改善。')
    st.download_button('导出该用户状态与更新/反馈历史', core.dumps({'profile': state, 'audit': [x for x in st.session_state.audit if x['user_id'] == user]}), f'{user}_state_audit.json', 'application/json')

elif page == '实验对照':
    st.subheader('A · 预测模型对照')
    st.write('同一批留出记录，比较固定 Ridge、均值基线、沿用当日精力和移除历史精力的消融模型。')
    st.dataframe(result['metrics'], hide_index=True, use_container_width=True)
    test = result['test'][result['test'].user_id == user]
    plot = test[['target_date', 'target', 'prediction', 'persistence']].melt(id_vars='target_date', var_name='系列', value_name='精力')
    plot['系列'] = plot['系列'].map({'target': '模拟观察值', 'prediction': 'Ridge预测', 'persistence': '沿用当日精力'})
    st.altair_chart(alt.Chart(plot).mark_line(point=True).encode(x=alt.X('target_date:T', title='目标日期'),
                         y=alt.Y('精力:Q'), color='系列:N', tooltip=['target_date:T', '系列:N', '精力:Q']).properties(height=280), use_container_width=True)
    st.download_button('下载逐样本留出预测 CSV', result['test'].to_csv(index=False).encode('utf-8-sig'), 'holdout_predictions.csv', 'text/csv')
    st.subheader('B · 情境反馈消融')
    st.write('对同一用户、同一睡眠历史、同一安全规则，比较完整情境与移除心理情境。此对照只展示规则决策变化，不能证明 MWM 改善睡眠。')
    scenario = st.selectbox('对照情境', ['当前保存状态', '忙碌夜班（零可用时间）', '高压力，优先减轻负担'])
    s = deepcopy(state)
    if scenario.startswith('忙碌'):
        s.update(available_minutes=0, night_shift=True)
    elif scenario.startswith('高压力'):
        s.update(goal='减轻睡前负担', stress=9, available_minutes=5)
    for col, contextual in zip(st.columns(2), [False, True]):
        run = core.run_agents(history, s, '', use_context=contextual)
        with col:
            st.markdown('**' + ('完整情境' if contextual else '移除心理情境') + '**')
            st.write(run['selected']['action'] if run['selected'] else run['safety']['message'])
            st.dataframe(pd.DataFrame(run['candidates'])[['id', 'score', 'veto', 'reason']], hide_index=True)
            with st.expander('查看分支与假设后继状态'):
                st.json(run)
    st.caption('移除心理情境组故意忽略时间与夜班，可能产生不适配建议；这是消融示范，不能当作推荐策略。两组均保留安全否决。')
    st.subheader('C · 后续真实研究方案（尚未开展）')
    st.write('预注册结局与切分 → 伦理审查及知情同意 → 接入真实数据并审计缺失/时序 → 按用户分组外部验证 → 随机分配规则反馈与情境反馈 → 盲评可执行性、约束违反率与采纳率；健康效果需独立随访。')
    st.warning('没有招募受试者，没有真实临床验证，没有显著性或健康改善结论。页面数字全部由本地模拟数据即时计算。')

else:
    st.subheader('设计依据与实现边界')
    st.dataframe(pd.DataFrame([
        ['MWM §3.3 / §5.3', '联合物理—心理状态', '睡眠记录 + 用户声明的目标、偏好、信念、意图、情绪、社会约束'],
        ['MWM §5.3', '目标个体的部分观察', '只访问当前用户历史与其明确输入；未知变量不推断'],
        ['MWM §5.4–5.5', '行动分解、后继状态与分支评价', '显式物理行动/心理意图；定性假设后继状态；人工评分与安全否决'],
        ['SCPS §2.3.1 / §2.3.4', '以人为中心、知识与数据协同', '用户可修改状态；轻量模型 + 独立生活习惯规则'],
        ['SCPS §4', '持续质量保障和运行监测', '时序隔离测试、协同日志、安全路由、用户反馈审计']
    ], columns=['论文定位', '理论概念', '本原型映射']), hide_index=True, use_container_width=True)
    st.info('这是受两篇论文启发的领域化工程原型，不是 MENTIS 复现，也不是完整心理世界模型或真实 SCPS 部署。后继状态是人工定性假设，评分未经人类标注校准。三个智能体是 Python 逻辑模块，不使用大语言模型。')
    st.markdown('参考：Fei & Zhao, *Mental World Modeling*, arXiv:2607.27201v1（用户上传版本），§3.3、§5.3–5.5。\n\nBelbachir et al., *Towards socio-cyber-physical systems: a software perspective amid AI breakthroughs*, Sci China Inf Sci 69, 193101 (2026)，[DOI](https://doi.org/10.1007/s11432-026-5037-1)，§2.3、§4。')
    st.markdown(f'生活习惯规则来源：[CDC About Sleep]({core.CDC})。就医与药物边界参考：[NHLBI Insomnia Treatment]({core.NIH})。文献访问日期：2026-10-28；规则ID与工程假设详见 docs/THEORY.md。')
    st.subheader('示例数据字典')
    dictionary = pd.read_csv(Path(__file__).parent / 'data' / 'dictionary.csv')
    st.dataframe(dictionary, hide_index=True, use_container_width=True)
    st.download_button('下载完整模拟数据（360行）', data.to_csv(index=False).encode('utf-8-sig'), 'sleep_demo.csv', 'text/csv')
    st.caption('无需网络即可运行。首次安装开源依赖需要联网；运行时不访问外部 API、不发送个人数据。请勿在此原型中输入真实敏感信息。')
