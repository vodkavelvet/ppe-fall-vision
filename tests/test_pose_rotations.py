import numpy as np
import pytest
from pfv.perception.base import PersonPose, PoseError
from pfv.perception.yolo_pose import YoloPoseEstimator, _restore_pose, _merge_poses
from pfv.config import PoseConfig

def pose(box=(10,20,50,80), conf=.8):
    kp=np.zeros((17,3),np.float32);kp[:]=[25,40,.9];kp[0]=0
    return PersonPose(bbox=box,keypoints=kp,confidence=conf,keypoint_conf=.5)

@pytest.mark.parametrize('angle,box,point',[
 (90,(120,10,180,50),(159,25)),
 (-90,(20,50,80,90),(40,74)),
 (180,(150,20,190,80),(174,59)),
])
def test_rotation_restores_box_joints_and_missing_points(angle,box,point):
    p=_restore_pose(pose(),angle,200,100)
    assert p.bbox==pytest.approx(box)
    assert p.keypoints[1,:2]==pytest.approx(point)
    assert p.keypoints[0].tolist()==[0,0,0]
    assert p.keypoints[1,2]==pytest.approx(.9)

def test_merging_removes_duplicate_and_keeps_other_person():
    p=pose();other=pose((120,20,170,80));best=pose((11,21,51,81),.95)
    result=_merge_poses([p,other,best],.5)
    assert len(result)==2
    assert result[0] is best
    assert result[1] is other

def test_estimate_maps_rotated_predictions_back_before_merging(monkeypatch):
    est=YoloPoseEstimator(rotations=[90]); est._device='cpu'
    class Model:
        def predict(self,image,**kwargs):return [image.shape[:2]]
    monkeypatch.setattr(est,'load',lambda:Model())
    monkeypatch.setattr(est,'_convert',lambda shape:[pose()] if shape==(100,200) else [])
    results=est.estimate(np.zeros((100,200,3),np.uint8))
    assert len(results)==1
    assert results[0].bbox==(10,20,50,80)

@pytest.mark.parametrize('angle',[45,0,270])
def test_bad_rotations_are_rejected(angle):
    with pytest.raises(PoseError):YoloPoseEstimator(rotations=[angle])
    with pytest.raises(ValueError):PoseConfig(rotations=[angle])

def test_v2_replay_preserves_two_people(tmp_path):
    from pfv.fall.replay import replay_clip
    a,b=pose(),pose((120,20,170,80))
    path=tmp_path/'s01_walk.npz'
    np.savez(path,detected=np.array([True]),keypoints=np.array([a.keypoints]),
             bboxes=np.array([a.bbox]),confidence=np.array([a.confidence]),fall_flags=np.array([False]),
             frame_offsets=np.array([0,2]),all_keypoints=np.array([a.keypoints,b.keypoints]),
             all_bboxes=np.array([a.bbox,b.bbox]),all_confidence=np.array([a.confidence,b.confidence]))
    steps=list(replay_clip(path))
    assert len(steps[0].tracked)==2

def test_extraction_keeps_all_detections_and_empty_frames(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('extract_pose_v2_test',Path(__file__).parents[1]/'scripts/extract_pose_caucafall.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a,b=pose(),pose((120,20,170,80),.95)
    class Source:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):
            return iter([SimpleNamespace(index=i,image=i) for i in range(3)])
    class Estimator:
        model_name='fake-pose.pt';conf=.5;imgsz=640;rotations=(-90,90);merge_iou=.5;rotation_conf=.5
        def estimate(self,image):return [[a,b],[],[a]][image]
    monkeypatch.setattr(module,'ImageSequenceSource',Source)
    result=module.extract_clip(SimpleNamespace(n_frames=3,image_paths=[],fall_flags=[False,True,True]),Estimator(),None)
    assert result['frame_offsets'].tolist()==[0,2,2,3]
    assert result['all_bboxes'].shape==(3,4)
    assert result['n_people'].tolist()==[2,0,1]
    assert result['detected'].tolist()==[True,False,True]
    np.testing.assert_allclose(result['all_bboxes'][0],a.bbox)
    np.testing.assert_allclose(result['all_bboxes'][1],b.bbox)


def test_original_view_is_preferred_over_higher_rotation_confidence():
    original=pose(conf=.7);rotated=pose((11,21,51,81),.95)
    other=pose((120,20,170,80),.9)
    result=_merge_poses([original,rotated,other],.5,primary_count=1)
    assert len(result)==2
    assert result[0] is original
    assert result[1] is other


def test_weak_chair_is_removed_without_discarding_supported_person():
    from pfv.perception.yolo_pose import _fuse_views
    person=pose(conf=.3);chair=pose((120,20,170,80),.35)
    rotated_person=pose((11,21,51,81),.9)
    result=_fuse_views([person,chair],[rotated_person],.5,.5)
    assert len(result)==1
    assert result[0] is person


def test_rotated_view_recovers_person_missing_from_primary():
    from pfv.perception.yolo_pose import _fuse_views
    recovered=pose(conf=.9)
    assert _fuse_views([], [recovered], .5, .5)==[recovered]
