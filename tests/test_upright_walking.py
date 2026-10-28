from pathlib import Path
import pytest
from pfv.config import load_config
from pfv.fall.features import StaticFeatures,TemporalFeatures
from pfv.fall.state_machine import FallStateMachine,FallState

# Rounded observations from s08_walk at 0.05, 0.7, 1.0, 7.3 and 8.9 seconds.
OBSERVATIONS=[(2.564,9.002,.258,.159,.294,.678,176.08),(1.445,31.230,.261,.159,.295,.629,176.09),(1.738,18.744,.272,.153,.280,.647,176.34),(2.367,1.577,.256,.173,.338,.699,171.85),(1.980,2.489,.272,.185,.325,.679,173.14)]

def profile():return load_config(Path(__file__).parents[1]/'config/fall_robust_v3.yaml').fall

def observed(row):
    aspect,angle,trunk,knee,ankle,spread,knee_angle=row
    return StaticFeatures(bbox_aspect_ratio=aspect,trunk_angle_deg=angle,trunk_length_norm=trunk,hip_knee_dy_norm=knee,hip_ankle_dy_norm=ankle,keypoint_spread_y_norm=spread,max_knee_angle_deg=knee_angle)

@pytest.mark.parametrize('row',OBSERVATIONS)
def test_observed_walking_and_startup_velocity_keep_upright(row):
    m=FallStateMachine(profile());f=observed(row)
    assert m.update(0,f).state is FallState.UPRIGHT
    d=m.update(.05,f,TemporalFeatures(hip_velocity_norm=1.608,window_s=.05,n_samples=2,contiguous=True))
    assert d.state is FallState.UPRIGHT
    assert not d.alert

def test_normal_walk_then_actual_fall_changes_real_internal_state():
    m=FallStateMachine(profile())
    for i in range(100):assert m.update(i/20,observed(OBSERVATIONS[i%5])).state is FallState.UPRIGHT
    lying=StaticFeatures(trunk_angle_deg=80,trunk_length_norm=.2,keypoint_spread_y_norm=.3,hip_knee_dy_norm=.05,hip_ankle_dy_norm=.1,max_knee_angle_deg=140)
    outputs=[m.update(i/20,lying) for i in range(100,180)]
    assert outputs[0].state is FallState.FALLING
    assert sum(d.alert for d in outputs)==1
    assert outputs[-1].state is FallState.FALLEN

def test_legacy_profile_retains_velocity_hint():
    root=Path(__file__).parents[1]
    old=load_config(root/'config/fall_robust.yaml')
    assert old.fall.timing.falling_velocity==.6
    assert load_config(root/'config/fall_robust_v3.yaml').fall.timing.falling_velocity is None
