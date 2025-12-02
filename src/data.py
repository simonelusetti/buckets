from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from dora import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DATASETS_WITH_CONFIG = {
    "wikiann": "en",
    "cnn": "3.0.0",
    "cnn_highlights": "3.0.0",
}

NER_LABEL_NAMES = {
    "wikiann": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    "conll2003": ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
}


def sanitize_fragment(fragment: str) -> str:
    return fragment.replace("/", "-")


def normalize_dataset_config(name: str, dataset_config: Optional[str]) -> Optional[str]:
    default = DATASETS_WITH_CONFIG.get(name)
    if default is None:
        return None
    return dataset_config or default


def dataset_cache_filename(
    name: str,
    split: str,
    subset,
    *,
    cnn_field: Optional[str] = None,
    dataset_config: Optional[str] = None,
) -> str:
    parts = [name]
    if dataset_config:
        parts.append(sanitize_fragment(dataset_config))
    if cnn_field:
        parts.append(cnn_field)
    parts.append(split)
    if subset is not None and subset != 1.0:
        parts.append(str(subset))
    return "_".join(parts) + ".pt"


def _freeze_encoder(encoder: AutoModel) -> AutoModel:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def _dataset_cache_paths(
    name: str,
    split: str,
    subset,
    *,
    cnn_field: Optional[str],
    dataset_config: Optional[str],
) -> Tuple[Path, Optional[Path]]:
    primary = Path(
        to_absolute_path(
            f"./data/{dataset_cache_filename(name, split, subset, cnn_field=cnn_field, dataset_config=dataset_config)}"
        )
    )
    fallback = None
    if subset not in (None, 1.0):
        fallback = Path(
            to_absolute_path(
                f"./data/{dataset_cache_filename(name, split, None, cnn_field=cnn_field, dataset_config=dataset_config)}"
            )
        )
    return primary, fallback


def _apply_subset_and_shuffle(ds: Dataset, subset, *, shuffle: bool, log: bool) -> Dataset:
    if shuffle:
        ds = ds.shuffle(seed=42)
    if subset is None or subset == 1.0:
        return ds

    target_subset = int(len(ds) * subset) if subset <= 1.0 else int(subset)
    if target_subset <= 0:
        raise ValueError(f"Requested subset {subset} results in 0 examples.")
    ds = ds.select(range(target_subset))
    if log:
        logger.info("Materialised subset of %s examples.", target_subset)
    return ds


def _resolve_dataset(
    name: str,
    split: str,
    dataset_config: Optional[str],
    raw_dataset_root: Optional[str],
    cnn_field: Optional[str],
) -> Tuple[Dataset, Callable]:
    raw_split_path = None
    if raw_dataset_root is not None:
        raw_split_path = Path(raw_dataset_root) / split
        if not raw_split_path.exists():
            raise FileNotFoundError(f"Raw dataset split not found at {raw_split_path}")

    if name in {"cnn_highlights", "cnn"}:
        config_name = dataset_config or DATASETS_WITH_CONFIG["cnn"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("cnn_dailymail", config_name, split=split)
        target_field = cnn_field or "highlights"
        text_fn = lambda x: x[target_field]
    elif name == "wikiann":
        config_name = dataset_config or DATASETS_WITH_CONFIG["wikiann"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("wikiann", config_name, split=split)
        text_fn = lambda x: x["tokens"]
    elif name == "conll2003":
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("conll2003", revision="refs/convert/parquet", split=split)
        text_fn = lambda x: x["tokens"]
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return ds, text_fn


def encode_examples(
    ds: Dataset,
    tok: AutoTokenizer,
    encoder: AutoModel,
    text_fn: Callable,
    max_length: int,
) -> Dataset:
    def _tokenize_and_encode(example):
        text = text_fn(example)
        split_into_words = isinstance(text, (list, tuple))
        enc = tok(
            text,
            truncation=True,
            max_length=max_length,
            is_split_into_words=split_into_words,
        )

        device = next(encoder.parameters()).device
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"], device=device).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"], device=device).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs, output_attentions=False, return_dict=True)

        out_dict = {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
            "embeddings": out.last_hidden_state.squeeze(0).detach().cpu().to(torch.float32).numpy(),
        }

        if "ner_tags" in example and split_into_words:
            word_ids = enc.word_ids()
            ner_tags = example["ner_tags"]
            aligned = []
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(0)
                else:
                    aligned.append(ner_tags[word_id])
            out_dict["ner_tags"] = np.asarray(aligned, dtype=np.int64)
            out_dict["tokens"] = example.get("tokens", [])
        return out_dict

    return ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)


def build_dataset(
    name: str,
    split: str,
    tokenizer_name: str,
    max_length: int,
    subset=None,
    shuffle: bool = False,
    cnn_field: Optional[str] = None,
    dataset_config: Optional[str] = None,
    raw_dataset_root: Optional[str] = None,
    log: bool = True,
) -> Tuple[Dataset, AutoTokenizer]:
    dataset_config = normalize_dataset_config(name, dataset_config)
    ds, text_fn = _resolve_dataset(
        name=name,
        split=split,
        dataset_config=dataset_config,
        raw_dataset_root=raw_dataset_root,
        cnn_field=cnn_field,
    )
    ds = _apply_subset_and_shuffle(ds, subset, shuffle=shuffle, log=log)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = _freeze_encoder(AutoModel.from_pretrained(tokenizer_name))
    ds = encode_examples(ds, tok, encoder, text_fn, max_length)
    return ds, tok


def get_dataset(
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    name: str = "cnn",
    split: str = "train",
    dataset_config: Optional[str] = None,
    cnn_field: Optional[str] = None,
    subset=None,
    rebuild: bool = False,
    shuffle: bool = False,
    log: bool = True,
    max_length: int = 512,
    raw_dataset_root: Optional[str] = None,
) -> Tuple[Dataset, AutoTokenizer]:
    dataset_config = normalize_dataset_config(name, dataset_config)
    cache_path, fallback_path = _dataset_cache_paths(
        name,
        split,
        subset,
        cnn_field=cnn_field,
        dataset_config=dataset_config,
    )
    if cache_path.exists() and not rebuild:
        if log:
            logger.info("Loading cached dataset from %s", cache_path)
        ds = load_from_disk(cache_path)
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if shuffle and subset not in (None, 1.0):
            ds = _apply_subset_and_shuffle(ds, subset, shuffle=shuffle, log=log)
        return ds, tok

    if fallback_path and fallback_path.exists() and not rebuild:
        if log:
            logger.info(
                "Subset cache %s not found; using full cache %s and selecting subset in-memory.",
                cache_path,
                fallback_path,
            )
        ds = load_from_disk(fallback_path)
        ds = _apply_subset_and_shuffle(ds, subset, shuffle=shuffle, log=log)
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        return ds, tok

    if log:
        logger.info("Building dataset for %s/%s (subset=%s)", name, split, subset)
    ds, tok = build_dataset(
        name=name,
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        subset=subset,
        shuffle=shuffle,
        cnn_field=cnn_field,
        dataset_config=dataset_config,
        raw_dataset_root=raw_dataset_root,
        log=log,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(cache_path)
    return ds, tok


def collate(batch):
    def _as_tensor(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    input_ids = pad_sequence(
        tuple(_as_tensor(item["input_ids"], torch.long) for item in batch),
        batch_first=True,
        padding_value=0,
    )
    attention_masks = pad_sequence(
        tuple(_as_tensor(item["attention_mask"], torch.long) for item in batch),
        batch_first=True,
        padding_value=0,
    )

    has_ner = "ner_tags" in batch[0]
    if has_ner:
        ner_tags = pad_sequence(
            tuple(_as_tensor(item["ner_tags"], torch.long) for item in batch),
            batch_first=True,
            padding_value=-100,
        )

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }
    if has_ner:
        batch_out["ner_tags"] = ner_tags

    if "embeddings" in batch[0]:
        embeddings = pad_sequence(
            tuple(_as_tensor(item["embeddings"], torch.float) for item in batch),
            batch_first=True,
            padding_value=0.0,
        )
        batch_out["embeddings"] = embeddings
    return batch_out


def initialize_dataloaders(cfg, log: bool = True):
    train_cfg = cfg.data.train
    eval_cfg = cfg.data.eval
    dev_cfg = cfg.data.dev

    tokenizer_name = cfg.bucket_model.sbert_model

    def _build_loader(split_cfg, split_override: Optional[str] = None):
        split = split_override or split_cfg.split
        cnn_field = split_cfg.cnn_field if hasattr(split_cfg, "cnn_field") else None
        ds, _ = get_dataset(
            tokenizer_name=tokenizer_name,
            name=split_cfg.dataset,
            split=split,
            subset=split_cfg.subset,
            dataset_config=split_cfg.config,
            cnn_field=cnn_field,
            rebuild=cfg.data.rebuild_ds,
            shuffle=split_cfg.shuffle,
            log=log,
            raw_dataset_root=None,
        )
        return DataLoader(
            ds,
            batch_size=split_cfg.batch_size,
            collate_fn=collate,
            num_workers=split_cfg.num_workers,
            pin_memory=(cfg.runtime.device == "cuda"),
            persistent_workers=(split_cfg.num_workers > 0),
            shuffle=split_cfg.shuffle,
        )

    def _attach_label_names(dataloader, dataset_name: str):
        label_names = NER_LABEL_NAMES.get(dataset_name)
        if label_names:
            setattr(dataloader, "label_names", label_names)

    train_dl = _build_loader(train_cfg, "train")
    eval_dl = _build_loader(eval_cfg, eval_cfg.split)

    dev_dl = None
    if dev_cfg and dev_cfg.dataset:
        dev_dl = _build_loader(dev_cfg, dev_cfg.split)

    _attach_label_names(train_dl, train_cfg.dataset)
    _attach_label_names(eval_dl, eval_cfg.dataset)
    if dev_dl is not None:
        _attach_label_names(dev_dl, dev_cfg.dataset)

    return train_dl, eval_dl, dev_dl
