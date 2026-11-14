import json,collections,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REV
R=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else REV/'pilot_readouts.jsonl')]
print('records',len(R),'no_response',sum(r['text'] is None for r in R),'max attempts',max(r['attempts'] for r in R))
print('models returned',collections.Counter(r.get('model_returned') for r in R))
g=collections.defaultdict(list)
for r in R: g[(r['cell'],r['truth'])].append(r)
side=lambda v,t: None if v is None else ((v>0.5) if t=='A' else (v<0.5))
print(f"{'cell':12}{'T':2}{'n':>4}{'trunc':>6}{'strict_ok':>10}{'legacy=logged':>14}{'leg!=strict':>12}{'sideflip':>9}{'right_logged':>13}{'right_legacy':>13}{'right_strict':>13}")
for k,v in sorted(g.items()):
    n=len(v); tr=sum(r['stop_reason'] in('length','max_tokens') for r in v)
    ok=sum(r['strict_status']=='ok' for r in v)
    agree=sum(r['legacy_parse'] is not None and abs(r['legacy_parse']-r['logged_belief'])<1e-9 for r in v)
    dis=sum(r['strict_status']=='ok' and abs(r['legacy_parse']-r['strict_parse'])>1e-12 for r in v)
    flip=sum(r['strict_status']=='ok' and side(r['legacy_parse'],r['truth'])!=side(r['strict_parse'],r['truth']) for r in v)
    post=[r for r in v if r['round']>=1]
    rl=sum(bool(side(r['logged_belief'],r['truth'])) for r in post)/len(post)
    rg=sum(bool(side(r['legacy_parse'],r['truth'])) for r in post)/len(post)
    sp=[r for r in post if r['strict_status']=='ok']; rs=sum(bool(side(r['strict_parse'],r['truth'])) for r in sp)/max(1,len(sp))
    print(f"{k[0]:12}{k[1]:2}{n:4d}{tr:6d}{ok:10d}{agree:14d}{dis:12d}{flip:9d}{rl:13.2f}{rg:13.2f}{rs:13.2f}")
print('strict status',collections.Counter(r['strict_status'] for r in R))
print('stop reasons',collections.Counter(r['stop_reason'] for r in R))
ex=[r for r in R if r['strict_status']=='ok' and abs(r['legacy_parse']-r['strict_parse'])>1e-12][:4]
for r in ex: print('---',r['key'],'legacy',r['legacy_parse'],'strict',r['strict_parse'],'\n',r['text'][-300:])
bad=[r for r in R if r['strict_status'] not in('ok',)][:4]
for r in bad: print('=== INVALID',r['key'],r['strict_status'],r['stop_reason'],'\n',(r['text'] or '')[-300:])
