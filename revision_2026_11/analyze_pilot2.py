import json,collections,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REV
R=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else REV/'pilot_readouts.jsonl')]
side=lambda v,t: (v>0.5) if t=='A' else (v<0.5)
trunc=lambda r: r['stop_reason'] in ('length','max_tokens')
g=collections.defaultdict(list)
for r in R: g[(r['cell'],r['truth'],'r0' if r['round']==0 else 'post')].append(r)
print(f"{'cell':12}{'T':2}{'rnd':>5}{'n':>4}{'trunc%':>7}{'okFINAL':>8}{'ok_right':>9}{'trunc_legacy_right':>19}{'logged_side_agree':>18}{'comp_tokens':>12}")
for k,v in sorted(g.items()):
    n=len(v); t=sum(map(trunc,v)); ok=[r for r in v if r['strict_status']=='ok']
    tr=[r for r in v if trunc(r)]
    okr=sum(side(r['strict_parse'],r['truth']) for r in ok)
    trr=sum(side(r['legacy_parse'],r['truth']) for r in tr)
    sa=sum(side(r['legacy_parse'],r['truth'])==side(r['logged_belief'],r['truth']) for r in v)
    ct=sum(r['usage'].get('completion_tokens',0) for r in v)/n
    print(f"{k[0]:12}{k[1]:2}{k[2]:>5}{n:4d}{100*t/n:6.0f}%{len(ok):8d}{okr:>5}/{len(ok):<3}{trr:>11}/{len(tr):<7}{sa:>10}/{n:<7}{ct:12.0f}")
print('ok responses that were also truncated:',sum(trunc(r) and r['strict_status']=='ok' for r in R))
print('non-truncated without FINAL:',sum((not trunc(r)) and r['strict_status']!='ok' for r in R))
