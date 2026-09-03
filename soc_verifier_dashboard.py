import streamlit as st
import pandas as pd
import json, os
from datetime import datetime

BASE_DIR = os.environ.get('SOC_BASE_DIR', '/content/SOC-Audit-v7')
st.set_page_config(page_title='SOC v9 Verifier', layout='wide')
st.title('v9  Verifier Flags · Abstention Queue · Compliance Gap Map')

report_path = os.path.join(BASE_DIR, 'reports', 'final_soc_report.json')
gap_path    = os.path.join(BASE_DIR, 'reports', 'compliance_gap_map.json')
fb_path     = os.path.join(BASE_DIR, 'feedback', 'analyst_labels.csv')

results   = json.load(open(report_path)) if os.path.exists(report_path) else []
flagged   = [r for r in results if not r.get('_verifier', {}).get('passed', True)]
abstained = [r for r in results if r.get('_abstention', {}).get('abstained', False)]
clean     = [r for r in results if r not in flagged and r not in abstained]

c1, c2, c3 = st.columns(3)
c1.metric('Verifier Flags', len(flagged))
c2.metric('Abstentions', len(abstained))
c3.metric('Clean Outputs', len(clean))
st.divider()

# -- Verifier Flags --
st.subheader('Verifier-Flagged Outputs')
if flagged:
    for item in flagged:
        v   = item.get('_verifier', {})
        fid = str(item.get('finding_id', '?'))
        sev = str(item.get('severity', '?'))
        with st.expander(f'{fid}  [{sev}]  — {v.get("flag_count", 0)} flag(s)'):
            for e in v.get('errors', []):
                st.error(f'[ERROR] {e.get("field","?")} — {str(e.get("rule",[]))[:120]}')
            for w in v.get('warnings', []):
                st.warning(f'[WARN]  {w.get("field","?")} — {str(w.get("rule",[]))[:120]}')
            st.json(item.get('triage', {}))
else:
    st.success('All verifier checks passed — no flagged outputs.')
st.divider()

# -- Abstention Queue --
st.subheader('Abstention Queue (Manual Review Required)')
if abstained:
    for item in abstained:
        ab  = item.get('_abstention', {})
        fid = str(item.get('finding_id', '?'))
        sev = str(item.get('severity', '?'))
        with st.expander(f'{fid}  [{sev}]  — {len(ab.get("reasons", []))} reason(s)'):
            for t in ab.get('reason_text', ab.get('reasons', [])):
                st.write(f'- {t}')
            if st.button(f'Approve {fid} for training', key=f'approve_{fid}'):
                if os.path.exists(fb_path):
                    fb = pd.read_csv(fb_path)
                    if fid in fb['id'].astype(str).values:
                        fb.loc[fb['id'].astype(str) == fid, 'review_status'] = 'approved'
                    else:
                        fb = pd.concat([fb, pd.DataFrame([{
                            'id': fid, 'review_status': 'approved',
                            'analyst_id': 'dashboard',
                            'adjusted_severity': item.get('severity', 'UNKNOWN'),
                            'feedback_time': str(datetime.now())}])], ignore_index=True)
                    fb.to_csv(fb_path, index=False)
                st.success(f'{fid} marked as approved for training')
else:
    st.success('No abstentions — all findings had sufficient evidence.')
st.divider()

# -- Compliance Gap Map --
st.subheader('Compliance Gap Map')
if os.path.exists(gap_path):
    gdata = json.load(open(gap_path))
    gdf   = pd.DataFrame(gdata.get('gap_map', []))
    if not gdf.empty:
        st.dataframe(gdf[['control_id','framework','finding_count',
                           'worst_severity','is_shared_gap']],
                     use_container_width=True)
    else:
        st.info('Gap map is empty.')
else:
    st.info('Run the Control Deduplication cell first to generate the gap map.')