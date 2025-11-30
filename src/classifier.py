# src/classifier.py

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict
from collections import deque, Counter
import open_clip


def get_color_boosts(crop: np.ndarray) -> Dict[str, float]:
    if crop.size == 0:
        return {cls: 1.0 for cls in ['mechanic', 'cleaner', 'worker', 'driver', 'unknown']}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = max(1, crop.shape[0] * crop.shape[1])

    boosts = {'mechanic': 1.0, 'cleaner': 1.0, 'worker': 1.0, 'driver': 1.0, 'unknown': 1.0}

    orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
    orange_ratio = cv2.countNonZero(orange_mask) / total_pixels
    if orange_ratio > 0.10:
        boosts['cleaner'] *= 1.5

    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    white_ratio = cv2.countNonZero(white_mask) / total_pixels
    if white_ratio > 0.25:
        boosts['mechanic'] *= 1.3

    return boosts


def apply_zone_prior(probs: Dict[str, float], zone: str) -> Dict[str, float]:
    priors = {
        "RepairZone": {'mechanic': 2.0, 'worker': 1.2, 'driver': 0.3, 'cleaner': 0.3, 'unknown': 0.5},
        "PlatformZone": {'cleaner': 2.0, 'worker': 1.2, 'mechanic': 0.5, 'driver': 0.5, 'unknown': 0.8},
        "CrossingZone": {'worker': 1.5, 'cleaner': 1.2, 'mechanic': 0.8, 'driver': 0.5, 'unknown': 1.0},
        "DriverZone": {'driver': 5.0, 'mechanic': 0.2, 'worker': 0.2, 'cleaner': 0.2, 'unknown': 0.3},
        "out_of_zones": {'driver': 1.0, 'mechanic': 0.8, 'worker': 1.2, 'cleaner': 0.8, 'unknown': 1.5}
    }

    prior = priors.get(zone, {cls: 1.0 for cls in probs})
    adjusted = {cls: probs[cls] * prior.get(cls, 1.0) for cls in probs}
    total = sum(adjusted.values())
    return {cls: val / total for cls, val in adjusted.items()} if total > 0 else probs


class WorkerClassifier:
    """
    Worker classification using CLIP.

    Key changes vs old implementation:
    - Per-track temporal voting: collect up to `vote_len` per-frame top-1 predictions for a *new* track
      and then assign the most common one permanently to that track_id (to avoid reclassification/flicker).
    - After a track receives a confirmed class, classifier will skip reprocessing it unless `force_classify=True`.
    - Keeps backward-compatible outputs expected by `main` (same keys on tracks).
    """

    WORKER_CLASSES = {
        'mechanic': {
            'name': 'Mechanic',
            'prompts': [
                'person wearing white work coveralls',
                'mechanic in dirty white uniform',
                'worker in light grey industrial suit',
                'industrial worker in light colored clothes'
            ],
            'color': (220, 220, 220)
        },
        'cleaner': {
            'name': 'Cleaner',
            'prompts': [
                'person wearing dark blue uniform with orange stripes',
                'worker in blue uniform and orange safety vest',
                'cleaner in blue workwear with high visibility orange',
                'railway cleaner uniform'
            ],
            'color': (0, 100, 200)
        },
        'worker': {
            'name': 'Worker',
            'prompts': [
                'construction worker in grey overalls',
                'person in dark grey work uniform',
                'worker wearing hard hat and safety clothes',
                'manual laborer in industrial gear'
            ],
            'color': (128, 128, 128)
        },
        'driver': {
            'name': 'Driver',
            'prompts': [
                'train driver',
                'person sitting in train cabin',
                'man inside locomotive',
                'person operating a train'
            ],
            'color': (50, 50, 200)
        },
        'unknown': {
            'name': 'Unknown',
            'prompts': [
                'person in casual clothes',
                'random person',
                'pedestrian'
            ],
            'color': (100, 100, 100)
        }
    }

    def __init__(
        self,
        model_name: str = 'hf-hub:Marqo/marqo-fashionCLIP',
        device: str = None,
        use_fine_tuned: bool = False,
        fine_tuned_path: str = None,
        history_len: int = 15,
        vote_len: int = 9
    ):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_len = history_len
        self.vote_len = vote_len

        # Per-track structures
        # track_history_probs used for smoothing/temporal averaging if desired (kept for compatibility)
        self.track_history_probs = {}  # track_id -> deque of prob dicts (maxlen=history_len)
        # voting deque used only for initial stabilization of new tracks
        self.track_vote_buf = {}  # track_id -> deque of top-1 class names (maxlen=vote_len)
        self.confirmed_class = {}  # track_id -> confirmed class name
        self.confirmed_conf = {}  # track_id -> float confidence

        print(f"Initializing WorkerClassifier (history_len={history_len}, vote_len={vote_len})")

        # Load model
        if use_fine_tuned and fine_tuned_path and Path(fine_tuned_path).exists():
            print(f"  Loading fine-tuned: {fine_tuned_path}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=fine_tuned_path, device=self.device
            )
        else:
            print(f"  Loading pretrained: {model_name}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, device=self.device
            )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Prepare text embeddings
        self._prepare_text_embeddings()
        print("Classifier ready")

    def _prepare_text_embeddings(self):
        all_prompts = []
        self.class_indices = {}
        idx = 0
        for cls_name, data in self.WORKER_CLASSES.items():
            prompts = data['prompts']
            all_prompts.extend(prompts)
            self.class_indices[cls_name] = list(range(idx, idx + len(prompts)))
            idx += len(prompts)

        with torch.no_grad():
            tokens = self.tokenizer(all_prompts).to(self.device)
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def classify_crop(self, frame: np.ndarray, bbox: List[float]) -> Dict[str, float]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {k: 0.0 for k in self.WORKER_CLASSES}

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_features @ self.text_features.T).softmax(dim=-1)

        class_probs = {}
        for cls_name, indices in self.class_indices.items():
            class_probs[cls_name] = similarity[0, indices].mean().item()

        total = sum(class_probs.values())
        if total > 0:
            class_probs = {k: v / total for k, v in class_probs.items()}

        # Apply color boosts
        color_boosts = get_color_boosts(crop)
        boosted = {cls: class_probs.get(cls, 0.0) * color_boosts.get(cls, 1.0) for cls in class_probs}
        total = sum(boosted.values())
        return {k: v / total for k, v in boosted.items()} if total > 0 else class_probs

    def classify_batch(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        zone_manager=None,
        force_classify: bool = False
    ) -> List[Dict]:
        """
        Backwards-compatible classify_batch.

        Behavior:
        - If a track_id already has a confirmed class and force_classify==False -> skip visual re-classification.
        - For new track_ids: collect top-1 predictions into a vote buffer of length `vote_len`.
          Once buffer is full (or force_classify=True), take majority vote and confirm class for that track_id.
        - Adds the same keys to track dicts as original implementation: 'worker_class', 'worker_name',
          'worker_confidence', 'worker_color'. Also keeps optional 'zone_name'.
        """
        for track in tracks:
            if track.get('class_name') != 'person':
                continue

            track_id = track.get('track_id')
            if track_id is None:
                continue

            # zone handling (optional)
            zone = None
            if zone_manager is not None:
                zone = zone_manager.get_zone(track['bbox'])
                track['zone_name'] = zone

            # If already confirmed and not forcing — just write confirmed values
            if (track_id in self.confirmed_class) and (not force_classify):
                cls = self.confirmed_class[track_id]
                track['worker_class'] = cls
                track['worker_name'] = self.WORKER_CLASSES[cls]['name']
                track['worker_confidence'] = float(self.confirmed_conf.get(track_id, 1.0))
                track['worker_color'] = self.WORKER_CLASSES[cls]['color']
                continue

            # Otherwise produce visual probabilities now
            visual_probs = self.classify_crop(frame, track['bbox'])

            # Apply zone prior if available
            if zone is not None:
                visual_probs = apply_zone_prior(visual_probs, zone)

            # Store into history (for optional smoothing)
            if track_id not in self.track_history_probs:
                self.track_history_probs[track_id] = deque(maxlen=self.history_len)
            self.track_history_probs[track_id].append(visual_probs)

            # Top-1 prediction for voting
            top1 = max(visual_probs, key=visual_probs.get)

            if track_id not in self.track_vote_buf:
                self.track_vote_buf[track_id] = deque(maxlen=self.vote_len)
            self.track_vote_buf[track_id].append(top1)

            # If not enough votes yet and not forcing — we put provisional unknown values
            if len(self.track_vote_buf[track_id]) < self.vote_len and not force_classify:
                # provide a best-effort provisional label (most common so far) but don't confirm it
                provisional = Counter(self.track_vote_buf[track_id]).most_common(1)[0][0]
                # compute provisional confidence as average prob for that class across history
                confid = self._avg_confidence(track_id, provisional)
                track['worker_class'] = provisional
                track['worker_name'] = self.WORKER_CLASSES[provisional]['name']
                track['worker_confidence'] = float(confid)
                track['worker_color'] = self.WORKER_CLASSES[provisional]['color']
                continue

            # Once we have enough votes (or force_classify), confirm class
            votes = list(self.track_vote_buf[track_id])
            most_common, count = Counter(votes).most_common(1)[0]

            # Compute confidence as average smoothed probability over stored probs
            avg_conf = self._avg_confidence(track_id, most_common)

            # Confirm
            self.confirmed_class[track_id] = most_common
            self.confirmed_conf[track_id] = float(avg_conf)

            track['worker_class'] = most_common
            track['worker_name'] = self.WORKER_CLASSES[most_common]['name']
            track['worker_confidence'] = float(avg_conf)
            track['worker_color'] = self.WORKER_CLASSES[most_common]['color']

        return tracks

    def _avg_confidence(self, track_id: int, cls_name: str) -> float:
        """Average probability for class `cls_name` over stored history for the track."""
        if track_id not in self.track_history_probs:
            return 0.0
        probs_list = list(self.track_history_probs[track_id])
        if not probs_list:
            return 0.0
        vals = [p.get(cls_name, 0.0) for p in probs_list]
        return float(sum(vals) / len(vals))

    def cleanup_lost_tracks(self, active_track_ids: List[int]):
        to_remove = [tid for tid in list(self.track_history_probs.keys()) if tid not in active_track_ids]
        for tid in to_remove:
            self.track_history_probs.pop(tid, None)
            self.track_vote_buf.pop(tid, None)
            self.confirmed_class.pop(tid, None)
            self.confirmed_conf.pop(tid, None)

    @staticmethod
    def get_worker_info(worker_class: str) -> Dict:
        return WorkerClassifier.WORKER_CLASSES.get(
            worker_class,
            WorkerClassifier.WORKER_CLASSES['unknown']
        )
