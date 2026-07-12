"""Optional ArcFace-quality donor identity embedding.

The first version's ``donor_pool.cheap_face_embedding`` (a downsampled-grayscale intensity
vector) was confirmed too coarse in practice: 585/4000 (14.6%) of donors got excluded by the
probe-overlap guard at a 0.9 cosine-similarity threshold on the first render-sheet run --
suspiciously high for a guard meant to catch only near-duplicates, and a symptom that
lighting/pose/background were swamping actual identity in that embedding space, degrading
hard-case donor pairing into "similar lighting" matching rather than "similar person" matching.

This module provides a real face-recognition embedding (ArcFace, insightface's
``buffalo_l``/``w600k_r50``) WHEN the environment has it available offline -- checked once,
lazily, not assumed. VESSL's insightface install, already used for the SCRFD detector that
fills the regions cache (``freuid.preprocess``), bundles this recognition model in the exact
same downloaded model pack (``~/.insightface/models/buffalo_l/w600k_r50.onnx``) -- confirmed
present and loadable with `allowed_modules=["detection", "recognition"]` + CPUExecutionProvider,
no new network fetch needed.

If ArcFace genuinely isn't available in a given environment (e.g. a bare install with only the
detection model), ``best_available_embed_fn`` degrades to ``donor_pool.cheap_face_embedding`` and
says so loudly, once -- silently returning a worse embedding without comment would be a much
worse failure mode than a printed warning. Hard-case donor pairing quality is approximate in
that fallback path (see donor_pool.sample_donor's own gender/embedding caveat), and this module
makes that condition detectable (``arcface_available()``) rather than assumed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_app = None  # lazy singleton; sentinel string "unavailable" marks a failed load
_warned = False


def _load_arcface():
    """Loads with an EXPLICITLY small onnxruntime thread pool. Without this, onnxruntime
    defaults each ONNX session's intra-op thread pool to roughly the host's core count --
    harmless on a normal workstation, but on a high-core-count VESSL box (96 cores observed)
    this caused thread-scheduling overhead to dominate the actual ~10-20ms of compute for these
    tiny (112x112 / 320x320) models: measured ~1000ms/embedding with the default thread pool
    (326 OS threads spawned) vs. ~530-570ms/embedding with intra_op_num_threads=1 -- roughly a
    2x speedup, and the 1000ms figure was itself still climbing (thread creation/teardown
    overhead compounding), not a stable steady-state number. Confirmed via
    insightface.app.FaceAnalysis's own kwarg forwarding: FaceAnalysis(**kwargs) ->
    model_zoo.get_model(onnx_file, **kwargs) -> onnxruntime.InferenceSession(path, **kwargs),
    so `sess_options` passed here reaches every model's session directly."""
    global _app
    if _app is None:
        try:
            import cv2  # noqa: F401  -- import here so a missing cv2 also falls back cleanly
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            app = FaceAnalysis(
                allowed_modules=["detection", "recognition"], sess_options=sess_options,
            )
            app.prepare(ctx_id=0, det_size=(320, 320))  # crops are small; no need for 640
            _app = app
            print("[face_embedding] ArcFace (insightface buffalo_l/w600k_r50) loaded")
        except Exception as exc:
            _app = "unavailable"
            global _warned
            if not _warned:
                print(
                    f"[face_embedding] ArcFace unavailable ({exc}) -- falling back to the cheap "
                    "embedding for every donor; hard-case pairing quality is approximate, see "
                    "this module's docstring"
                )
                _warned = True
    return None if _app == "unavailable" else _app


def arcface_available() -> bool:
    return _load_arcface() is not None


def arcface_embedding(face_crop: Image.Image) -> np.ndarray | None:
    """Re-runs SCRFD detection + 5-point alignment + ArcFace embedding on ``face_crop`` (expects
    a MARGINED crop, not a tight one -- SCRFD needs some context around the face to redetect
    reliably; ``donor_pool.build_donor_record`` supplies this via ``_margined_crop``). Returns
    None if no face is (re-)detected in the crop -- callers should treat that as "reject this
    donor" rather than silently falling back to a different embedding space for just this one
    donor (mixing embedding spaces within one pool would make cosine similarity meaningless)."""
    app = _load_arcface()
    if app is None:
        return None
    import cv2

    bgr = cv2.cvtColor(np.asarray(face_crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: float(f.det_score))
    emb = best.embedding
    return emb / (np.linalg.norm(emb) + 1e-8)


def best_available_embed_fn():
    """The embedding function donor_pool should use: ArcFace if this environment has it, the
    cheap fallback (which never returns None, so it never rejects a donor) otherwise. Checked
    ONCE per process, not per call -- callers that need to know which one they got should call
    ``arcface_available()`` themselves for logging/reporting."""
    from freuid.photosub.donor_pool import cheap_face_embedding

    if arcface_available():
        return arcface_embedding
    return cheap_face_embedding
