# Fall Robust V3 — development correction

## Latest full-dataset audit

The completed 100-clip fresh-inference audit supersedes the smaller development
checks below as the current summary: **42/50 fall clips detected, 1/50 normal
clips with an alarm, and 27/50 normal clips showing provisional FALLING**.
Eight falls are missed; Subject 8 Fall left emits duplicate alarms on different
track IDs. The module is not correct across every subject/activity.

See [the human-readable audit summary](ALL_SUBJECTS_AUDIT.md) and
[the complete structured record](../eval/development/all_subjects_v3_20261028.json).
These are already-seen development data, not a new independent final test.


V2 sometimes let a short projected trunk plus a low hip position classify sitting
or walking as lying. In s04_sit_down, the trunk stayed near 20 degrees from
vertical while its normalized length was about 0.24–0.27. The weak short-trunk
vote and hip/ankle vote accumulated long enough to trigger at 8.25 s.

## Change

- Compute the maximum knee angle only when both complete legs are valid.
- For a vertically extended skeleton (spread >= 0.59), reject the short-trunk
  vote when both knees are bent (maximum <= 135 degrees), or a tall box
  (height/width >= 1.4) has a trunk below the lying-angle threshold (<= 45 degrees).
- This rejects only the weak short-trunk cue. Direct trunk-angle or compactness
  evidence can still establish a bent-leg fall.
- While FALLING, preserve short-trunk support briefly after direct lying
  evidence, for less than the existing 3-second persistence interval. This
  bridges camera-axis foreshortening without enabling cold-start walking.
- Emit a new alarm only on a current LYING observation. Historical fractions
  alone no longer emit an alarm on an UPRIGHT/UNKNOWN frame in profile V3.

These are projected-pose heuristics. The numerical limits need validation on
other cameras. Missing knee joints never establish sitting. The unreliable velocity-only FALLING hint is disabled in the corrected V3
profile; persistence remains 3 seconds.

## Subject 8 state-label correction

The first V3 revision removed the alarm but still showed FALLING on 139 of
219 tracked observations in s08_walk. At 0.05 s, a two-frame hip-velocity
estimate triggered suspicion while the posture was UPRIGHT. Later, projected
boxes around ratio 1.445 and trunk tilt 31 degrees fell outside the original
upright guard (ratio 2.4, tilt 30), so short-trunk evidence retriggered it.

The corrected profile disables `falling_velocity` and uses ratio 1.4 / tilt
45 for the upright guard, retaining the vertical-spread check and temporal
bridge for actual falls. This is a development configuration adjustment based
on already-seen data, not a universally validated human-proportion threshold.
All 219 cached observations now remain UPRIGHT, with zero FALLING/FALLEN.
The regression command checks the actual state histogram, not only alert count.

## Compatibility

`config/fall_robust_v3.yaml` opts into the changes. New guards and the current
LYING requirement default to disabled, retaining V1/V2 decision behavior for
historical profiles. V1 frozen models, gate, result and run log are unchanged.
Webcam and demo launchers now use V3. Static feature output contains eight
fields; the historical seven-feature availability script still measures its
explicit original feature list.

## Earlier verification snapshots

An earlier development subset contained 30 clips: 13 fall and 17 ordinary
activities. On the same current pose detections, both profiles detect 12/13
falls; false-alarm clips drop from 2/17 to 0/17. No previously detected fall in
this subset is lost. Subject 9 Fall sitting still alarms at 7.3 s. Subject 4
Sit down and Subject 8 Walk no longer alarm.

Subject 1 Fall left is missed by both profiles on current detections: the clip
contains only 2.15 seconds of labeled lying, shorter than 3-second persistence.
This is a remaining detection limit, not a claim of complete accuracy.

An additional replay of the 100 legacy single-candidate pose caches changes
43/50 detected falls and 2/50 false-alarm clips (V2) to 41/50 and 0/50 (V3).
Two legacy detections are lost: s01_fall_left and s05_fall_forward. Fresh
rotation inference detects s05_fall_forward with both profiles (V3 6.5 s).
The legacy results are preserved in the development report and must not be
presented as results of the current rotation pipeline over all 100 clips.

574 automated tests pass. All these clips have already been inspected; this is
development/regression evidence, not a new independent final test.

## Run

```bash
sh run_demo_sitting.sh
sh run_webcam.sh
.venv/bin/pfv fall --config config/fall_robust_v3.yaml --clip "Subject.4/Sit down"
.venv/bin/python scripts/verify_fall_v3.py
```

The default regression cache contains the 30 already-seen clips in
`data/interim/caucafall/pose_v3_regression` (gitignored, generated locally).
The compact recorded results are versioned under `eval/development/`.
