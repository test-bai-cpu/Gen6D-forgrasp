import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imsave, imread

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d

import time

def load_all(params):
    cfg = load_cfg(params["cfg"])
    ref_database = parse_database_name(params["database"])
    
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')

    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))
    
    return estimator, object_bbox_3d


def get_pose_gen6d(estimator, object_bbox_3d, img, img_name, output):
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)

    (output_dir / 'images_raw').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_inter').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out_smooth').mkdir(exist_ok=True, parents=True)

    pose_init = None
        
    # generate a pseudo K
    h, w, _ = img.shape
    f=np.sqrt(h**2+w**2)
    K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

    estimator.cfg['refine_iter'] = 5
    
    pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init)

    pts, _ = project_points(object_bbox_3d, pose_pr, K)
    bbox_img = draw_bbox_3d(img, pts, (0,0,255))
    imsave(f'{str(output_dir)}/images_out/{img_name}-bbox.jpg', bbox_img)
    np.save(f'{str(output_dir)}/images_out/{img_name}-pose.npy', pose_pr)
    imsave(f'{str(output_dir)}/images_inter/{img_name}.jpg', visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d))
    
    return pose_pr

def test_main():
    object_type = "square_b"
    params = {
        "cfg": "configs/gen6d_pretrain.yaml",
        "database": f"custom/{object_type}",
        "output": f"data/custom/{object_type}/test",
        "ffmpeg": "ffmpeg",
        "num": 5,
        "std": 2.5
    }
    estimator, object_bbox_3d = load_all(params)
    
    for i in range(0, 8):
        img_name = f"color_image_{i}"
        output = f"data/custom/{object_type}/test"
        
        img = imread(f"data/custom/image/{img_name}.png")
        
        predict_pose = get_pose_gen6d(estimator, object_bbox_3d, img, img_name, output)
        print(predict_pose)

def main():
    object_type = "square_b"
    params = {
        "cfg": "configs/gen6d_pretrain.yaml",
        "database": f"custom/{object_type}",
        "output": f"data/custom/{object_type}/test",
        "ffmpeg": "ffmpeg",
        "num": 5,
        "std": 2.5
    }
    estimator, object_bbox_3d = load_all(params)
    
    
    img_name = f"color_image_0"
    output = f"data/custom/{object_type}/test"
    
    img = imread(f"data/custom/image/{img_name}.png")
    
    predict_pose = get_pose_gen6d(estimator, object_bbox_3d, img, img_name, output)
    print(predict_pose)

main()