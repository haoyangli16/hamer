import torch
import numpy as np
from pathlib import Path
import cv2
import os
from typing import List, Dict, Optional, Union
import time
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

# Try importing ViTPoseModel, assuming it's in the python path or current directory
try:
    from vitpose_model import ViTPoseModel
except ImportError:
    # If running from a different directory, we might need to adjust path or this might fail
    # But since we are in the same dir as demo.py, this should work if PYTHONPATH includes it
    import sys

    sys.path.append(str(Path(__file__).parent))
    from vitpose_model import ViTPoseModel

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


class HamerPredictor:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        body_detector: str = "vitdet",
        rescale_factor: float = 2.0,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the HaMeR predictor.
        Args:
            checkpoint_path: Path to the model checkpoint.
            body_detector: 'vitdet' or 'regnety'.
            rescale_factor: Factor for padding the bbox.
            batch_size: Batch size for HaMeR inference (number of hands processed in parallel per frame).
            device: Torch device.
        """
        self.checkpoint_path = checkpoint_path
        self.body_detector = body_detector
        self.rescale_factor = rescale_factor
        self.batch_size = batch_size

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        print(f"Loading HaMeR models on {self.device}...")

        # Download and load models
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(self.checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load detector
        print(f"Loading {self.body_detector} detector...")
        if self.body_detector == "vitdet":
            cfg_path = (
                Path(hamer.__file__).parent
                / "configs"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[
                    i
                ].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif self.body_detector == "regnety":
            from detectron2 import model_zoo
            from detectron2.config import get_cfg

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            raise ValueError(f"Unknown body detector: {self.body_detector}")

        # Keypoint detector
        print("Loading ViTPose...")
        self.cpm = ViTPoseModel(self.device)

        # Renderer
        print("Setting up renderer...")
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        print("Initialization complete.")

    def predict(
        self,
        rgb_input: Union[np.ndarray, List[np.ndarray]],
        full_frame: bool = True,
        side_view: bool = False,
        render_crops: bool = False,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Process input RGB image(s).

        Args:
            rgb_input: numpy array of shape (H, W, 3) or (T, H, W, 3). Values should be 0-255 (uint8).
                       Assumed to be RGB.
            full_frame: Boolean, whether to return full frame visualization.
            side_view: Boolean, whether to include side view in crop visualization.
            render_crops: Boolean, whether to render individual hand crops (like demo).
            verbose: Boolean, print progress.

        Returns:
            results: list of dicts (one per frame). Each dict contains:
                - 'frame_idx': index of the frame
                - 'hands': list of dicts, each containing:
                    - 'vertices': np.array (N, 3)
                    - 'cam_t': np.array (3,)
                    - 'is_right': bool
                    - 'box': np.array (4,) [x1, y1, x2, y2]
                    - 'pred_cam': np.array (3,)
                    - 'crop_visualization': np.ndarray (H, W, 3) (if render_crops=True)
                - 'visualization': np.ndarray (H, W, 3) or (H, W*2, 3) etc, RGB image (if full_frame=True)
        """
        # Normalize input to list of frames or (T, H, W, 3)
        frames = None
        if isinstance(rgb_input, np.ndarray):
            if len(rgb_input.shape) == 3:
                frames = rgb_input[None, ...]  # Add T dim
            elif len(rgb_input.shape) == 4:
                frames = rgb_input
            else:
                raise ValueError(
                    f"Input shape {rgb_input.shape} not supported. Expected (H, W, 3) or (T, H, W, 3)"
                )
        elif isinstance(rgb_input, list):
            frames = np.stack(rgb_input)
        else:
            raise ValueError("Input must be numpy array or list of numpy arrays.")

        results = []

        _frames_iter = (
            tqdm(enumerate(frames), total=len(frames), desc="Processing frames")
            if verbose
            else enumerate(frames)
        )

        for frame_idx, img_rgb in _frames_iter:
            # if verbose and frame_idx % 10 == 0:
            #     print(f"Processing frame {frame_idx}/{len(frames)}")

            # Detectron2 DefaultPredictor expects BGR usually, as it uses cv2.imread logic
            # The DefaultPredictor_Lazy calls model(input) where input is dict.
            # demo.py does: img_cv2 = cv2.imread(...) # BGR
            # det_out = detector(img_cv2)
            # So we convert RGB to BGR for detector
            img_bgr = img_rgb[:, :, ::-1].copy()

            # 1. Detect humans
            det_out = self.detector(img_bgr)

            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()

            # 2. Detect keypoints
            # cpm.predict_pose expects RGB (it uses ViTPose which typically expects RGB)
            # demo.py passes 'img' which is RGB (line 82 and 90)
            vitposes_out = self.cpm.predict_pose(
                img_rgb,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []
            is_right = []

            # 3. Filter hands based on keypoints
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes["keypoints"][-42:-21]
                right_hand_keyp = vitposes["keypoints"][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [
                        keyp[valid, 0].min(),
                        keyp[valid, 1].min(),
                        keyp[valid, 0].max(),
                        keyp[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(0)

                keyp = right_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [
                        keyp[valid, 0].min(),
                        keyp[valid, 1].min(),
                        keyp[valid, 0].max(),
                        keyp[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(1)

            frame_result = {"frame_idx": frame_idx, "hands": [], "visualization": None}

            if len(bboxes) == 0:
                # If full_frame requested, return original image as visualization
                if full_frame:
                    frame_result["visualization"] = img_rgb.copy()
                results.append(frame_result)
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # 4. Run HaMeR
            # ViTDetDataset expects BGR image (demo.py passes img_cv2 which is BGR)
            dataset = ViTDetDataset(
                self.model_cfg,
                img_bgr,
                boxes,
                right,
                rescale_factor=self.rescale_factor,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            all_verts = []
            all_cam_t = []
            all_right = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model(batch)

                multiplier = 2 * batch["right"] - 1
                pred_cam = out["pred_cam"]
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()

                scaled_focal_length = (
                    self.model_cfg.EXTRA.FOCAL_LENGTH
                    / self.model_cfg.MODEL.IMAGE_SIZE
                    * img_size.max()
                )
                pred_cam_t_full = (
                    cam_crop_to_full(
                        pred_cam, box_center, box_size, img_size, scaled_focal_length
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Store results per hand
                batch_size_curr = batch["img"].shape[0]
                for n in range(batch_size_curr):
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    is_right_hand = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
                    cam_t = pred_cam_t_full[n]

                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right_hand)

                    hand_info = {
                        "vertices": verts,
                        "cam_t": cam_t,
                        "is_right": bool(is_right_hand),
                        "pred_cam": pred_cam[n].detach().cpu().numpy(),
                    }

                    if render_crops:
                        # Render crop visualization like demo.py
                        # Prepare input patch (denormalize)
                        input_patch = batch["img"][n].cpu() * (
                            DEFAULT_STD[:, None, None] / 255
                        ) + (DEFAULT_MEAN[:, None, None] / 255)
                        input_patch = input_patch.permute(1, 2, 0).numpy()

                        regression_img = self.renderer(
                            out["pred_vertices"][n].detach().cpu().numpy(),
                            out["pred_cam_t"][n].detach().cpu().numpy(),
                            batch["img"][n],
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                        )

                        if side_view:
                            white_img = (
                                torch.ones_like(batch["img"][n]).cpu()
                                - DEFAULT_MEAN[:, None, None] / 255
                            ) / (DEFAULT_STD[:, None, None] / 255)
                            side_img = self.renderer(
                                out["pred_vertices"][n].detach().cpu().numpy(),
                                out["pred_cam_t"][n].detach().cpu().numpy(),
                                white_img,
                                mesh_base_color=LIGHT_BLUE,
                                scene_bg_color=(1, 1, 1),
                                side_view=True,
                            )
                            final_img = np.concatenate(
                                [input_patch, regression_img, side_img], axis=1
                            )
                        else:
                            final_img = np.concatenate(
                                [input_patch, regression_img], axis=1
                            )

                        # Convert back to uint8 (0-255)
                        hand_info["crop_visualization"] = (final_img * 255).astype(
                            np.uint8
                        )

                    frame_result["hands"].append(hand_info)

            # Assign boxes to hands (order is preserved)
            for i, hand_data in enumerate(frame_result["hands"]):
                hand_data["box"] = boxes[i]

            # 5. Visualization
            if full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )

                # renderer.render_rgba_multiple needs list of arrays
                cam_view = self.renderer.render_rgba_multiple(
                    all_verts,
                    cam_t=all_cam_t,
                    render_res=img_size[n],
                    is_right=all_right,
                    **misc_args,
                )

                # Overlay
                # input_img expects float 0-1
                input_img = (
                    img_bgr.astype(np.float32)[:, :, ::-1] / 255.0
                )  # BGR -> RGB, normalized
                input_img = np.concatenate(
                    [input_img, np.ones_like(input_img[:, :, :1])], axis=2
                )  # Add alpha

                # cam_view is (H,W,4)
                input_img_overlay = (
                    input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                    + cam_view[:, :, :3] * cam_view[:, :, 3:]
                )

                # Output is RGB
                vis_img = (input_img_overlay * 255).astype(np.uint8)

                frame_result["visualization"] = vis_img
            else:
                if full_frame:
                    frame_result["visualization"] = img_rgb.copy()

            results.append(frame_result)

        return results


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HaMeR Predictor Demo for Video/Image")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input video or image"
    )
    parser.add_argument(
        "--output", type=str, default="output_hamer", help="Path to output folder"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument(
        "--side_view",
        action="store_true",
        default=False,
        help="Render side view (for crops)",
    )
    parser.add_argument(
        "--render_crops", action="store_true", default=False, help="Render hand crops"
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = HamerPredictor(
        checkpoint_path=args.checkpoint, batch_size=args.batch_size
    )

    # Load input
    input_path = args.input
    if input_path.endswith((".mp4", ".avi", ".mov")):
        # Process video
        cap = cv2.VideoCapture(input_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # cv2 reads BGR, convert to RGB
            frames.append(frame[:, :, ::-1])
        cap.release()
        print(f"Loaded {len(frames)} frames from video.")

        if len(frames) == 0:
            print("No frames extracted.")
            exit(1)

        start_time = time.time()
        # Run prediction
        # Processing all at once might assume enough memory. For very long videos, chunking is better.
        # Here we pass all frames.
        video_input = np.stack(frames)  # (T, H, W, 3)
        results = predictor.predict(
            video_input,
            full_frame=True,
            render_crops=args.render_crops,
            side_view=args.side_view,
            verbose=True,
        )
        end_time = time.time()
        total_processing_time = end_time - start_time
        avg_fps = len(frames) / total_processing_time
        print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")

        # Save video
        os.makedirs(args.output, exist_ok=True)
        output_filename = os.path.join(
            args.output, f"out_{os.path.basename(input_path)}"
        )

        if len(results) > 0 and results[0]["visualization"] is not None:
            h, w, _ = results[0]["visualization"].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

            for res in results:
                vis = res["visualization"]
                # Save as BGR
                out.write(vis[:, :, ::-1])
            out.release()
            print(f"Saved output video to {output_filename}")

    elif input_path.endswith((".jpg", ".png", ".jpeg")):
        # Process single image
        img = cv2.imread(input_path)
        img_rgb = img[:, :, ::-1]
        results = predictor.predict(
            img_rgb,
            full_frame=True,
            render_crops=args.render_crops,
            side_view=args.side_view,
        )

        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, f"out_{os.path.basename(input_path)}")
        if results[0]["visualization"] is not None:
            cv2.imwrite(out_path, results[0]["visualization"][:, :, ::-1])
            print(f"Saved output image to {out_path}")
    else:
        print("Unsupported input format.")
