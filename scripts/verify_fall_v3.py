#!/usr/bin/env python
"""Compare V2/V3 on already-seen development clips. Not an independent test."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from pfv.config import load_config
from pfv.evaluation.cache import build_clip_data
from pfv.evaluation.simulate import simulate_clip
from pfv.fall.state_machine import FallStateMachine


def main() -> int:
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-dir',type=Path,default=root/'data/interim/caucafall/pose_v3_regression')
    parser.add_argument('--out',type=Path,default=root/'reports/fall_v3_regression.json')
    args=parser.parse_args()
    before=load_config(root/'config/fall_robust.yaml');after=load_config(root/'config/fall_robust_v3.yaml')
    paths=sorted(args.pose_dir.glob('*.npz'))
    required={'s04_sit_down','s09_fall_sitting','s08_walk'}
    if not required.issubset({p.stem for p in paths}):
        parser.error('Regression cache must contain s04_sit_down, s09_fall_sitting and s08_walk.')
    rows=[]
    for path in paths:
        clip=build_clip_data(path,after);a=simulate_clip(clip,before.fall);b=simulate_clip(clip,after.fall)
        states=Counter()
        for seq in clip.tracks:
            machine=FallStateMachine(after.fall)
            for t,static,temporal in zip(seq.times,seq.static,seq.temporal):
                states[machine.update(t,static,temporal).state.value]+=1
        rows.append({'v3_states':dict(states),'clip':clip.name,'is_fall':clip.is_fall,'lying_s':clip.lying_s,'v2_alert':a.alerted,'v3_alert':b.alerted,'v2_alert_s':a.first_alert_s,'v3_alert_s':b.first_alert_s})
    def summarize(key):
        return {'detected_falls':sum(r[key] for r in rows if r['is_fall']),'falls':sum(r['is_fall'] for r in rows),'false_alarm_clips':[r['clip'] for r in rows if not r['is_fall'] and r[key]],'negative_clips':sum(not r['is_fall'] for r in rows)}
    lost=[r['clip'] for r in rows if r['is_fall'] and r['v2_alert'] and not r['v3_alert']]
    indexed={r['clip']:r for r in rows}
    walk_upright=set(indexed['s08_walk']['v3_states'])=={'UPRIGHT'}
    passed=walk_upright and not lost and indexed['s09_fall_sitting']['v3_alert'] and not indexed['s04_sit_down']['v3_alert'] and not indexed['s08_walk']['v3_alert']
    report={'protocol':'already-seen development regression; not independent test','v2':summarize('v2_alert'),'v3':summarize('v3_alert'),'lost_falls':lost,'subject8_walk_all_upright':walk_upright,'passed':passed,'clips':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print('DEVELOPMENT REGRESSION (already-seen clips)')
    print(json.dumps({k:v for k,v in report.items() if k!='clips'},indent=2))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
