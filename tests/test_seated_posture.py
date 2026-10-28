from dataclasses import replace
import numpy as np
import pytest
from pfv.config import FallConfig,PostureConfig
from pfv.fall.features import StaticFeatures,max_knee_angle_deg
from pfv.fall.posture import classify_posture,Posture
from pfv.fall.state_machine import FallStateMachine
from pfv.perception.base import PersonPose

def legs(left='bent',right='bent'):
    kp=np.zeros((17,3),np.float32)
    for side,kind in [(0,left),(1,right)]:
        hip,knee,ankle=11+side,13+side,15+side;x=10+side*20
        kp[hip]=[x,10,.9];kp[knee]=[x+10,10,.9]
        kp[ankle]=[x+10,30,.9] if kind=='bent' else [x+30,10,.9]
    return kp

def pose(kp):return PersonPose((0,0,100,100),kp,.9,keypoint_conf=.5)

def weak_short_trunk(**overrides):
    return replace(StaticFeatures(trunk_angle_deg=20,trunk_length_norm=.25,keypoint_spread_y_norm=.65,hip_ankle_dy_norm=.32,hip_knee_dy_norm=.13,max_knee_angle_deg=125),**overrides)

def test_bilateral_bent_knees_have_right_angles():
    assert max_knee_angle_deg(pose(legs()))==pytest.approx(90)

def test_one_straight_leg_prevents_bilateral_bend_evidence():
    assert max_knee_angle_deg(pose(legs(right='straight')))==pytest.approx(180)

@pytest.mark.parametrize('joint',[11,12,13,14,15,16])
def test_any_missing_leg_joint_returns_unknown(joint):
    kp=legs();kp[joint,2]=.49
    assert max_knee_angle_deg(pose(kp)) is None

def test_coincident_joint_returns_unknown():
    kp=legs();kp[13]=kp[11]
    assert max_knee_angle_deg(pose(kp)) is None

def test_knee_angle_is_invariant_to_scale_translation_and_rotation():
    a=legs();b=a.copy();b[:,:2]=b[:,:2]@np.array([[0,-1],[1,0]])*3+[100,200]
    assert max_knee_angle_deg(pose(a))==pytest.approx(max_knee_angle_deg(pose(b)))

def test_opt_in_guard_rejects_only_weak_short_trunk_cue():
    legacy=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34)
    guarded=legacy.model_copy(update={'seated_knee_angle_max_deg':135})
    assert classify_posture(weak_short_trunk(),legacy).posture is Posture.LYING
    result=classify_posture(weak_short_trunk(),guarded)
    assert result.posture is Posture.UPRIGHT
    assert result.votes['trunk_length'] is False

@pytest.mark.parametrize('extra',[{'trunk_angle_deg':80},{'keypoint_spread_y_norm':.30}])
def test_bent_legs_do_not_veto_direct_lying_orientation(extra):
    cfg=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,seated_knee_angle_max_deg=135)
    assert classify_posture(weak_short_trunk(**extra),cfg).posture is Posture.LYING

@pytest.mark.parametrize('angle',[None,160])
def test_missing_or_straight_knees_do_not_establish_sitting(angle):
    cfg=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,seated_knee_angle_max_deg=135)
    assert classify_posture(weak_short_trunk(max_knee_angle_deg=angle),cfg).posture is Posture.LYING

def test_compact_camera_axis_pose_keeps_short_trunk_evidence():
    cfg=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,seated_knee_angle_max_deg=135)
    result=classify_posture(weak_short_trunk(keypoint_spread_y_norm=.55),cfg)
    assert result.votes['trunk_length'] is True
    assert result.posture is Posture.LYING

def test_long_sitting_produces_no_alarm_but_following_real_fall_does():
    cfg=FallConfig(posture=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,seated_knee_angle_max_deg=135))
    machine=FallStateMachine(cfg);alerts=[]
    for i in range(200):
        t=i/20
        feature=weak_short_trunk() if t<5 else weak_short_trunk(trunk_angle_deg=80,keypoint_spread_y_norm=.3)
        if machine.update(t,feature).alert:alerts.append(t)
    assert len(alerts)==1
    assert alerts[0]>=8

@pytest.mark.parametrize('limit',[0,-1,180,181])
def test_bad_knee_threshold_rejected(limit):
    with pytest.raises(ValueError):PostureConfig(seated_knee_angle_max_deg=limit)

def test_tall_upright_walking_pose_does_not_use_short_trunk_vote():
    cfg=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,upright_aspect_ratio_min=2)
    result=classify_posture(weak_short_trunk(bbox_aspect_ratio=2.8,max_knee_angle_deg=175),cfg)
    assert result.posture is Posture.UPRIGHT
    assert result.votes['trunk_length'] is False

def test_axial_fall_with_narrower_box_keeps_short_trunk_vote():
    cfg=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,upright_aspect_ratio_min=2)
    result=classify_posture(weak_short_trunk(bbox_aspect_ratio=1.7,max_knee_angle_deg=175),cfg)
    assert result.posture is Posture.LYING

@pytest.mark.parametrize('current',['upright','unknown'])
def test_alarm_waits_for_a_current_lying_frame(current):
    from pfv.config import TimingConfig
    cfg=FallConfig(timing=TimingConfig(require_current_lying_for_alert=True))
    machine=FallStateMachine(cfg)
    lying=StaticFeatures(trunk_angle_deg=80,trunk_length_norm=.2,keypoint_spread_y_norm=.3,hip_ankle_dy_norm=.1,hip_knee_dy_norm=.05)
    upright=StaticFeatures(trunk_angle_deg=0,trunk_length_norm=.4,keypoint_spread_y_norm=.9,hip_ankle_dy_norm=.5,hip_knee_dy_norm=.3)
    for i in range(60):assert not machine.update(i/20,lying).alert
    last=machine.update(3.,upright if current=='upright' else StaticFeatures())
    assert not last.alert
    assert last.reason=='awaiting_current_lying'
    assert machine.update(3.05,lying).alert

def bridge_config():
    from pfv.config import TimingConfig
    return FallConfig(posture=PostureConfig(trunk_length_max=.28,hip_ankle_dy_max=.34,seated_knee_angle_max_deg=135,upright_aspect_ratio_min=2.4),timing=TimingConfig(require_current_lying_for_alert=True))

def ambiguous_axial_pose():
    return weak_short_trunk(bbox_aspect_ratio=2.7,max_knee_angle_deg=170)

def test_recent_direct_lying_bridges_camera_axis_ambiguity():
    machine=FallStateMachine(bridge_config());alerts=[]
    for i in range(80):
        f=weak_short_trunk(trunk_angle_deg=80,keypoint_spread_y_norm=.3) if i<10 else ambiguous_axial_pose()
        if machine.update(i/20,f).alert:alerts.append(i/20)
    assert len(alerts)==1
    assert alerts[0]>=3

def test_one_direct_glimpse_does_not_bridge_forever():
    machine=FallStateMachine(bridge_config())
    assert not machine.update(0,weak_short_trunk(trunk_angle_deg=80,keypoint_spread_y_norm=.3)).alert
    assert not any(machine.update(i/20,ambiguous_axial_pose()).alert for i in range(1,120))

def test_recovery_clears_direct_lying_memory():
    machine=FallStateMachine(bridge_config())
    direct=weak_short_trunk(trunk_angle_deg=80,keypoint_spread_y_norm=.3)
    for i in range(80):machine.update(i/20,direct)
    normal=weak_short_trunk(trunk_length_norm=.5,hip_ankle_dy_norm=.5,hip_knee_dy_norm=.3,max_knee_angle_deg=175)
    for i in range(80,160):machine.update(i/20,normal)
    assert machine._last_direct_lying is None
    assert not any(machine.update(i/20,ambiguous_axial_pose()).alert for i in range(160,240))
