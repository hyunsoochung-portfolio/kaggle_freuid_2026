import pathlib

p = pathlib.Path("src/freuid/augment.py")
src = p.read_text()
assert "class RecaptureAugWrapper" not in src, "already patched"

new_class = '''

class RecaptureAugWrapper(Dataset):
    """Wraps a FreuidDataset, applying print-and-recapture degradation
    (recapture_transforms) to a random fraction of ALL samples regardless of
    label (bona-fide or fraud). Unlike SynthTamperWrapper, this never changes
    the label -- it exists purely to expose the model to realistic capture-
    pipeline degradation (JPEG re-compression, downscale, blur, noise, mild
    geometry) during training, since the competition's private test set
    emphasises non-synthetic, physically-recaptured examples over clean or
    purely-digital ones.

    The base dataset must be created with transform=None so this wrapper
    owns all transform decisions.
    """

    def __init__(
        self,
        base: Dataset,
        clean_transform,
        recapture_transform,
        prob: float,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.clean_tf = clean_transform
        self.recapture_tf = recapture_transform
        self.prob = prob
        self._rng = np.random.default_rng(seed)
        self._return_face_meta = getattr(base, "_return_face_meta", False)

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        sample = self.base.samples[idx]  # type: ignore[attr-defined]
        src = sample.card_path if sample.card_path is not None else sample.path
        img_pil = Image.open(src).convert("RGB")
        img_size = img_pil.size
        label = sample.label

        if self._rng.random() < self.prob:
            img_out = self.recapture_tf(img_pil)
        else:
            img_out = self.clean_tf(img_pil) if self.clean_tf is not None else img_pil

        if self._return_face_meta:
            return img_out, label, face_meta_tensor(sample, img_size)
        return img_out, label
'''

p.write_text(src.rstrip("\n") + "\n" + new_class + "\n")
print("augment.py patched")

p2 = pathlib.Path("src/freuid/train.py")
src2 = p2.read_text()
assert "recapture_aug_prob" not in src2, "already patched"

old = '''    synth_double = bool(cfg.extra.get("synth_tamper_double", False))
    synth_prob = float(cfg.extra.get("synth_tamper_prob", 0.0))
    if synth_double:'''
new = '''    synth_double = bool(cfg.extra.get("synth_tamper_double", False))
    synth_prob = float(cfg.extra.get("synth_tamper_prob", 0.0))
    recapture_aug_prob = float(cfg.extra.get("recapture_aug_prob", 0.0))
    if synth_double:'''
assert old in src2, "anchor 1 not found"
src2 = src2.replace(old, new, 1)

old2 = '''    else:
        train_ds = FreuidDataset(
            cfg.data_dir, "train", train_tf, ids=train_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
    val_ds = FreuidDataset('''
new2 = '''    elif recapture_aug_prob > 0.0:
        from freuid.augment import RecaptureAugWrapper, recapture_transforms
        _base_train_ds = FreuidDataset(
            cfg.data_dir, "train", None, ids=train_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
        _recap_tf = recapture_transforms(size, mean, std)
        train_ds = RecaptureAugWrapper(
            _base_train_ds, clean_transform=train_tf, recapture_transform=_recap_tf,
            prob=recapture_aug_prob, seed=cfg.seed,
        )
        print(
            f"[train] recapture_aug: prob={recapture_aug_prob:.2f} "
            f"| {len(_base_train_ds)} samples, labels unchanged"
        )
    else:
        train_ds = FreuidDataset(
            cfg.data_dir, "train", train_tf, ids=train_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
    val_ds = FreuidDataset('''
assert old2 in src2, "anchor 2 not found"
src2 = src2.replace(old2, new2, 1)

p2.write_text(src2)
print("train.py patched")
