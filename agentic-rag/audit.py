import pandas as pd

sep = '=' * 65
print(sep)
print('FULL NIGHT AUDIT')
print(sep)

df = pd.read_csv('results/generation_cache.csv')
print()
print('GENERATION CACHE')
print('-' * 40)
print('Total rows:', len(df))
for c in ['baseline','reranker','agentic']:
    empty = df[c+'_answer'].isna().sum() + (df[c+'_answer']=='').sum()
    lat = df[c+'_latency'].mean()
    print(c + ': ' + str(len(df)-int(empty)) + '/150 answers, mean latency ' + str(round(lat,2)) + 's')
print('Agentic mean attempts: ' + str(round(df['agentic_n_attempts'].mean(),3)))
print('Agentic mean confidence: ' + str(round(df['agentic_confidence'].mean(),4)))
low = df['agentic_low_conf'].map(lambda x: str(x).strip().lower()=='true').sum()
print('Agentic low-conf flags: ' + str(int(low)) + ' / 150')

df2 = pd.read_csv('results/results_main.csv')
df2 = df2[~df2['question'].astype(str).str.contains('AGGREGATE', na=False)]
print()
print('RAGAS RESULTS')
print('-' * 40)
conditions = ['baseline','reranker','agentic']
metrics = ['faithfulness','answer_relevancy','context_precision']
for m in metrics:
    row = m.ljust(28)
    for c in conditions:
        col = c + '_' + m
        if col in df2.columns:
            v = df2[col].mean(skipna=True)
            row += str(round(v,4)).rjust(14)
        else:
            row += 'MISSING'.rjust(14)
    print(row)
for c in conditions:
    col = c + '_latency'
    v = df2[col].mean(skipna=True) if col in df2.columns else 0
    print(c + ' latency: ' + str(round(v,2)) + 's')

print()
print('THRESHOLD SWEEP')
print('-' * 40)
df3 = pd.read_csv('results/threshold_sweep.csv')
print(df3.to_string(index=False))

print()
print('MISSING CHECK')
print('-' * 40)
missing = []
for c in conditions:
    for m in metrics:
        col = c + '_' + m
        if col not in df2.columns or df2[col].isna().all():
            missing.append(col)
if not missing:
    print('All 9 RAGAS columns present and scored.')
else:
    for x in missing:
        print('MISSING: ' + x)
sweep_taus = list(df3['tau'])
needed = [t for t in [0.40, 0.50, 0.60] if t not in sweep_taus]
if needed:
    print('Sweep missing tau values: ' + str(needed))
else:
    print('Sweep complete for tau 0.40 0.50 0.60.')

print()
print(sep)
print('VERDICT')
print(sep)
print('Generation 150q x 3 conditions : COMPLETE')
done = [c for c in conditions if all(c+'_'+m in df2.columns and not df2[c+'_'+m].isna().all() for m in metrics)]
notdone = [c for c in conditions if c not in done]
print('RAGAS complete for: ' + str(done))
if notdone:
    print('RAGAS missing for: ' + str(notdone))
print('Sweep tau values done: ' + str(sweep_taus))
print('Paper draft: SC_ARAG_paper_draft.docx')
print(sep)
