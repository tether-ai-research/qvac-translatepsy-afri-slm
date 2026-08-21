import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*persistent_workers.*")

import argparse
import ast
import json
import hashlib
import random
import logging
import functools
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH, LeanMinHash
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from huggingface_hub import hf_hub_download

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


# All local paths are configurable through CLI flags or environment variables.
# Hugging Face datasets and models are downloaded through their standard caches.
RAW_DATA_DIR = Path(os.environ.get("QVAC_RAW_DATA_DIR", "data/raw")).expanduser()
OUTPUT_DIR = Path(os.environ.get("QVAC_OUTPUT_DIR", "data/processed")).expanduser()
EVAL_DATA_DIR = Path(os.environ.get("QVAC_EVAL_DATA_DIR", "data/eval")).expanduser()

EVAL_DATASET_NAMES = (
    "eval_smol",
    "eval_flores_200",
    "eval_wmt24pp",
    "eval_bouquet",
)

EVAL_DATASET_REVISIONS = {
    "facebook/bouquet": "9a6070a9652e350dda1d353c4fd198533199a911",
    "google/smol": "43939e46fd7a708726df6ed9a23a980ab9806545",
    "facebook/flores": "71abf77d8b7beb5cfef59898d6b24d92ab7654fc",
    "google/wmt24pp": "fd7405c06494bc66a57b25f55d217a72f96e60dc",
}

# SMOL SmolSent configurations used by the evaluation suite.
SMOL_EVAL_LANGUAGES = {
    "am": "amh_Ethi",
    "ha": "hau_Latn",
    "ig": "ibo_Latn",
    "ln": "lin_Latn",
    "so": "som_Latn",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
    "zu": "zul_Latn",
    "es": "spa_Latn",
    "om": "gaz_Latn",
    "mg": "plt_Latn",
    "rw": "kin_Latn",
    "xh": "xho_Latn",
    "af": "afr_Latn",
    "wo": "wol_Latn",
    "lg": "lug_Latn",
    "pcm": "pcm_Latn",
    "ny": "nya_Latn",
    "sn": "sna_Latn",
    "tn": "tsn_Latn",
    "st": "sot_Latn",
    "ak": "aka_Latn",
    "ber": "tzm_Tfng",
    "bm": "bam_Latn",
    "nso": "nso_Latn",
    "mos": "mos_Latn",
    "aeb": "aeb_Arab",
    "apd": "apd_Arab",
    "ktu": "ktu_Latn",
}
SMOL_ID_ALIGNED_LANGUAGES = {
    "amh_Ethi",
    "hau_Latn",
    "ibo_Latn",
    "lin_Latn",
    "som_Latn",
    "swh_Latn",
    "yor_Latn",
    "zul_Latn",
    "spa_Latn",
}

# One regional WMT24++ configuration per output language. This matches the
# evaluation artifacts: fr_FR, pt_PT, and sw_KE are used rather than their
# alternate regional variants.
WMT24PP_EVAL_LANGUAGES = {
    "en-ar_EG": "arb_Arab",
    "en-ar_SA": "ars_Arab",
    "en-bg_BG": "bul_Cyrl",
    "en-bn_IN": "ben_Beng",
    "en-ca_ES": "cat_Latn",
    "en-cs_CZ": "ces_Latn",
    "en-da_DK": "dan_Latn",
    "en-de_DE": "deu_Latn",
    "en-el_GR": "ell_Grek",
    "en-es_MX": "spa_Latn",
    "en-et_EE": "est_Latn",
    "en-fa_IR": "pes_Arab",
    "en-fi_FI": "fin_Latn",
    "en-fil_PH": "tgl_Latn",
    "en-fr_FR": "fra_Latn",
    "en-gu_IN": "guj_Gujr",
    "en-he_IL": "heb_Hebr",
    "en-hi_IN": "hin_Deva",
    "en-hr_HR": "hrv_Latn",
    "en-hu_HU": "hun_Latn",
    "en-id_ID": "ind_Latn",
    "en-is_IS": "isl_Latn",
    "en-it_IT": "ita_Latn",
    "en-ja_JP": "jpn_Jpan",
    "en-kn_IN": "kan_Knda",
    "en-ko_KR": "kor_Hang",
    "en-lt_LT": "lit_Latn",
    "en-lv_LV": "lvs_Latn",
    "en-ml_IN": "mal_Mlym",
    "en-mr_IN": "mar_Deva",
    "en-nl_NL": "nld_Latn",
    "en-no_NO": "nob_Latn",
    "en-pa_IN": "pan_Guru",
    "en-pl_PL": "pol_Latn",
    "en-pt_PT": "por_Latn",
    "en-ro_RO": "ron_Latn",
    "en-ru_RU": "rus_Cyrl",
    "en-sk_SK": "slk_Latn",
    "en-sl_SI": "slv_Latn",
    "en-sr_RS": "srp_Cyrl",
    "en-sv_SE": "swe_Latn",
    "en-sw_KE": "swh_Latn",
    "en-ta_IN": "tam_Taml",
    "en-te_IN": "tel_Telu",
    "en-th_TH": "tha_Thai",
    "en-tr_TR": "tur_Latn",
    "en-uk_UA": "ukr_Cyrl",
    "en-ur_PK": "urd_Arab",
    "en-vi_VN": "vie_Latn",
    "en-zh_CN": "zho_Hans",
    "en-zh_TW": "zho_Hant",
    "en-zu_ZA": "zul_Latn",
}


def _dataset_path(root, name):
    return Path(root).expanduser().resolve() / name


def _save_dataset(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    return output_path


def _prepare_smol_eval():
    rows_by_language = {}

    for language_code, flores_code in SMOL_EVAL_LANGUAGES.items():
        filename = f"smolsent/en_{language_code}.jsonl"
        logging.info(f"Loading google/smol file {filename}")
        local_path = hf_hub_download(
            repo_id="google/smol",
            filename=filename,
            repo_type="dataset",
            revision=EVAL_DATASET_REVISIONS["google/smol"],
        )
        ds = load_dataset(
            "json",
            data_files=local_path,
            split="train",
        )

        rows = {
            example["id"]: (example["src"], example["trg"])
            for example in ds
        }
        rows_by_language[flores_code] = rows

    # The multilingual benchmark follows the 862-row Somali configuration.
    # Other languages occasionally omit a translation; preserve those cells
    # as empty strings so all language columns remain aligned.
    source_code = "som_Latn"
    eval_ids = sorted(rows_by_language[source_code])
    source_by_id = {
        row_id: rows_by_language[source_code][row_id][0]
        for row_id in eval_ids
    }
    sources = [source_by_id[row_id] for row_id in eval_ids]
    columns = {
        "id": list(range(len(eval_ids))),
        "sentence_eng_Latn": sources,
    }
    for flores_code, rows in rows_by_language.items():
        if flores_code in SMOL_ID_ALIGNED_LANGUAGES:
            for row_id in eval_ids:
                if row_id in rows and rows[row_id][0] != source_by_id[row_id]:
                    raise ValueError(
                        f"SMOL language {flores_code} has a misaligned English "
                        f"source at ID {row_id}."
                    )
            targets = [
                rows[row_id][1] if row_id in rows else ""
                for row_id in eval_ids
            ]
        else:
            targets = [
                rows[row_id][1] if row_id in rows else ""
                for row_id in range(len(eval_ids))
            ]
        columns[f"sentence_{flores_code}"] = targets

    return DatasetDict({"dev": Dataset.from_dict(columns)})


def _prepare_wmt24pp_eval():
    columns = None
    expected_segment_ids = None
    expected_sources = None

    for config, flores_code in WMT24PP_EVAL_LANGUAGES.items():
        logging.info(f"Loading google/wmt24pp configuration {config}")
        ds = load_dataset(
            "google/wmt24pp",
            config,
            split="train",
            revision=EVAL_DATASET_REVISIONS["google/wmt24pp"],
        )
        ds = ds.filter(
            lambda example: not example["is_bad_source"],
            desc=f"Filtering bad WMT24++ sources for {config}",
        )

        segment_ids = list(ds["segment_id"])
        sources = list(ds["source"])
        if columns is None:
            expected_segment_ids = segment_ids
            expected_sources = sources
            columns = {
                "segment_id": segment_ids,
                "sentence_eng_Latn": sources,
                "domain": list(ds["domain"]),
            }
        elif segment_ids != expected_segment_ids or sources != expected_sources:
            raise ValueError(
                f"WMT24++ configuration {config} is not aligned with the other "
                "evaluation configurations."
            )

        columns[f"sentence_{flores_code}"] = list(ds["target"])

    columns["id"] = list(range(len(expected_segment_ids)))
    return DatasetDict({"test": Dataset.from_dict(columns)})


def _prepare_bouquet_eval():
    language_code_renames = {
        "cmn_Hans": "zho_Hans",
        "cmn_Hant": "zho_Hant",
        "ekk_Latn": "est_Latn",
        "kor_Kore": "kor_Hang",
        "por_Latn_braz1246": "por_Latn",
    }
    long_dataset = load_dataset(
        "facebook/bouquet",
        "sentence_level",
        revision=EVAL_DATASET_REVISIONS["facebook/bouquet"],
    )
    prepared_splits = {}

    for split_name, split_dataset in long_dataset.items():
        row_order = []
        metadata_by_id = {}
        sentences_by_id = {}
        language_codes = set()

        for example in tqdm(
            split_dataset,
            desc=f"Pivoting BOUQUET {split_name}",
        ):
            uniq_id = example["uniq_id"]
            if uniq_id not in metadata_by_id:
                row_order.append(uniq_id)
                metadata_by_id[uniq_id] = {
                    "uniq_id": uniq_id,
                    "domain": example["domain"],
                    "register": example["register"],
                    "par_id": example["par_id"],
                    "tags": example["tags"],
                }
                sentences_by_id[uniq_id] = {}

            source_language = language_code_renames.get(
                example["src_lang"], example["src_lang"]
            )
            target_language = language_code_renames.get(
                example["tgt_lang"], example["tgt_lang"]
            )
            language_codes.update((source_language, target_language))
            sentences_by_id[uniq_id].setdefault(source_language, example["src_text"])
            sentences_by_id[uniq_id].setdefault(target_language, example["tgt_text"])

        columns = {
            "id": list(range(len(row_order))),
            "uniq_id": row_order,
            "domain": [metadata_by_id[row_id]["domain"] for row_id in row_order],
            "register": [metadata_by_id[row_id]["register"] for row_id in row_order],
            "par_id": [metadata_by_id[row_id]["par_id"] for row_id in row_order],
            "tags": [metadata_by_id[row_id]["tags"] for row_id in row_order],
        }
        for language_code in sorted(language_codes):
            columns[f"sentence_{language_code}"] = [
                sentences_by_id[row_id].get(language_code, "")
                for row_id in row_order
            ]

        prepared_splits[split_name] = Dataset.from_dict(columns)

    return DatasetDict(prepared_splits)


def prep_eval_sets(eval_data_dir=None):
    """Download and prepare the four public evaluation datasets."""
    eval_data_dir = Path(eval_data_dir or EVAL_DATA_DIR).expanduser().resolve()
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    builders = (
        (
            "eval_bouquet",
            _prepare_bouquet_eval,
        ),
        ("eval_smol", _prepare_smol_eval),
        (
            "eval_flores_200",
            lambda: load_dataset(
                "facebook/flores",
                "all",
                revision=EVAL_DATASET_REVISIONS["facebook/flores"],
            ),
        ),
        ("eval_wmt24pp", _prepare_wmt24pp_eval),
    )

    for name, build in builders:
        output_path = _dataset_path(eval_data_dir, name)
        if output_path.exists():
            logging.info(f"Evaluation dataset already exists, skipping: {output_path}")
            continue
        logging.info(f"Preparing {name}")
        dataset = build()
        _save_dataset(dataset, output_path)
        logging.info(f"Saved {name} to {output_path}")



def get_mt_prompt(source_lang, target_lang, source_text, target_text):
    return [
        {"role": "system", "content": f"""You are a professional {source_lang} to {target_lang} translator. 
Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, 
and cultural sensitivities. Produce only the {target_lang} translation, without any additional explanations or commentary. """},
        {"role": "user", "content": f"""Please translate the following {source_lang} text into {target_lang}: {source_text}.\n\nTranslation:"""},
        {"role": "assistant", "content": f"""{target_text}"""}
    ]


def get_instruct_prompt(batch):
    messages = []
    for message, custom_instructions in zip(batch["messages"], batch["chat_template_kwargs"]):
        system_prompt = custom_instructions["custom_instructions"] if custom_instructions["custom_instructions"] else "You are a helpful assistant."
        messages.append([{"role": "system", "content": system_prompt}] + message)
    return {"messages": messages}


@functools.lru_cache(maxsize=1)
def _load_eval_sentences(eval_data_dir):
    """Load all evaluation test sets and return a list of unique individual sentences.

    Each sentence is treated independently — no pairing needed. A training example
    is contaminated if either its source or target is approximately similar to any
    sentence in any eval set, regardless of what it was paired with.
    Cached so the datasets are only loaded once per process.
    """
    eval_paths = [_dataset_path(eval_data_dir, name) for name in EVAL_DATASET_NAMES]
    missing_paths = [path for path in eval_paths if not path.exists()]
    if missing_paths:
        expected = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Exact reproduction requires the evaluation datasets used for "
            f"decontamination. Missing:\n{expected}\n"
            "Set --eval-data-dir or QVAC_EVAL_DATA_DIR to their parent directory."
        )

    seen = set()
    sentences = []
    for path in eval_paths:
        before = len(sentences)
        ds_dict = load_from_disk(str(path))
        for split_name, split_ds in ds_dict.items():
            sentence_cols = [c for c in split_ds.column_names if c.startswith("sentence_")]
            logging.info(f"  {path}/{split_name}: {len(split_ds)} rows x {len(sentence_cols)} languages")
            for row in split_ds:
                for col in sentence_cols:
                    text = row[col]
                    if text and text not in seen:
                        seen.add(text)
                        sentences.append(text)
        logging.info(f"  {path}: {len(sentences) - before} unique sentences added (total so far: {len(sentences)})")

    logging.info(f"Loaded {len(sentences)} unique eval sentences in total for decontamination")
    return sentences


def _bpe_minhash_dedup(
    dataset,
    tokenizer,
    num_perm=256,
    lsh_threshold=0.8,
    num_proc=32,
    decontaminate=True,
    eval_data_dir=None,
):
    """Approximate dedup via BPE subword unigram MinHash with side-prefixed shingles.
    Expects 'source' and 'target' text columns alongside 'messages'.

    If decontaminate=True, builds a separate LSH index from all unique sentences in
    the eval test sets (SMOL, FLORES-200, WMT24++) and removes any training example
    whose source or target is approximately similar to any eval sentence.
    """

    def compute_sent_mh(example, tokenizer, num_perm):
        """Compute a single-sentence MinHash (no side prefix) for decontamination."""
        token_ids = tokenizer.encode(example["text"], add_special_tokens=False)
        m = MinHash(num_perm=num_perm)
        for t in set(token_ids):
            m.update(str(t).encode())
        return {"_sig": m.hashvalues.tolist()}

    def compute_pair_mh(example, tokenizer, num_perm):
        """Compute three signatures per training example:
        - _src_sig / _tgt_sig: individual sentence hashes for decontamination
        - _mh_sig: side-prefixed pair hash for training dedup
        """
        src_ids = tokenizer.encode(example["source"], add_special_tokens=False)
        tgt_ids = tokenizer.encode(example["target"], add_special_tokens=False)
        src_m = MinHash(num_perm=num_perm)
        for t in set(src_ids):
            src_m.update(str(t).encode())
        tgt_m = MinHash(num_perm=num_perm)
        for t in set(tgt_ids):
            tgt_m.update(str(t).encode())
        pair_m = MinHash(num_perm=num_perm)
        for t in set(src_ids):
            pair_m.update(f"s{t}".encode())
        for t in set(tgt_ids):
            pair_m.update(f"t{t}".encode())
        return {
            "_src_sig": src_m.hashvalues.tolist(),
            "_tgt_sig": tgt_m.hashvalues.tolist(),
            "_mh_sig": pair_m.hashvalues.tolist(),
        }

    dataset = dataset.map(
        compute_pair_mh,
        num_proc=num_proc,
        fn_kwargs={"num_perm": num_perm, "tokenizer": tokenizer},
        desc="Computing BPE MinHash signatures",
    )

    contaminated = set()
    if decontaminate:
        eval_sentences = _load_eval_sentences(
            str(Path(eval_data_dir or EVAL_DATA_DIR).expanduser().resolve())
        )
        logging.info(f"Computing MinHash signatures for {len(eval_sentences)} eval sentences...")
        eval_ds = Dataset.from_dict({"text": eval_sentences})
        eval_ds = eval_ds.map(
            compute_sent_mh,
            num_proc=num_proc,
            fn_kwargs={"num_perm": num_perm, "tokenizer": tokenizer},
            desc="Computing eval sentence MinHash signatures",
        )

        logging.info("Building eval LSH index...")
        eval_lsh = MinHashLSH(threshold=0.9, num_perm=num_perm)
        for i, sig in enumerate(tqdm(eval_ds["_sig"], desc="Inserting eval sentences into LSH")):
            eval_lsh.insert(f"eval_{i}", LeanMinHash(seed=1, hashvalues=np.array(sig, dtype=np.uint64)))
        logging.info("Eval LSH index built. Scanning training data for contamination...")

        batch_size = 10_000
        for start in tqdm(range(0, len(dataset), batch_size), desc="Decontaminating"):
            batch = dataset[start:start + batch_size]
            for i, (src_sig, tgt_sig) in enumerate(zip(batch["_src_sig"], batch["_tgt_sig"])):
                src_m = LeanMinHash(seed=1, hashvalues=np.array(src_sig, dtype=np.uint64))
                if eval_lsh.query(src_m) or eval_lsh.query(LeanMinHash(seed=1, hashvalues=np.array(tgt_sig, dtype=np.uint64))):
                    contaminated.add(start + i)

        logging.info(f"Removed {len(contaminated)} training examples as contaminated by eval sets.")

    clean_indices = [i for i in range(len(dataset)) if i not in contaminated]
    dataset = dataset.select(clean_indices)

    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    duplicates = set()
    batch_size = 10_000

    for start in tqdm(range(0, len(dataset), batch_size), desc="BPE MinHash LSH dedup"):
        sigs = dataset[start:start + batch_size]["_mh_sig"]
        for i, sig in enumerate(sigs):
            idx = start + i
            m = LeanMinHash(seed=1, hashvalues=np.array(sig, dtype=np.uint64))
            if lsh.query(m):
                duplicates.add(idx)
            else:
                lsh.insert(idx, m)

    keep = [i for i in range(len(dataset)) if i not in duplicates]
    dataset = dataset.select(keep)
    dataset = dataset.remove_columns(["_mh_sig", "_src_sig", "_tgt_sig"])
    return dataset, len(duplicates)


def _comet_worker(gpu_id, model_path, batch_size, shard, shard_indices, return_dict):
    from comet import load_from_checkpoint

    model = load_from_checkpoint(model_path)
    scores = model.predict(shard, batch_size=batch_size, gpus=1, devices=[gpu_id])["scores"]
    return_dict[gpu_id] = (shard_indices, scores)


def predict_multi_gpu(model_path, data, n_gpus=8, batch_size=256):
    """Run COMET predict across n_gpus GPUs in parallel, return scores in original order."""
    ctx = mp.get_context("spawn")

    shards, indices = [], []
    for i in range(n_gpus):
        idx = list(range(i, len(data), n_gpus))
        indices.append(idx)
        shards.append([data[j] for j in idx])

    manager = ctx.Manager()
    return_dict = manager.dict()
    processes = []
    for gpu_id in range(n_gpus):
        if not shards[gpu_id]:
            continue
        p = ctx.Process(
            target=_comet_worker,
            args=(gpu_id, model_path, batch_size, shards[gpu_id], indices[gpu_id], return_dict),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    scores = [None] * len(data)
    for gpu_id in range(n_gpus):
        if gpu_id not in return_dict:
            continue
        shard_indices, shard_scores = return_dict[gpu_id]
        for idx, score in zip(shard_indices, shard_scores):
            scores[idx] = score
    return scores


def predict_bidirectional_comet_scores(model_path, sources, targets, n_gpus=8, batch_size=256):
    """Run COMET in both source->target and target->source directions."""
    forward_data = [{"src": src, "mt": tgt} for src, tgt in zip(sources, targets)]
    reverse_data = [{"src": tgt, "mt": src} for src, tgt in zip(sources, targets)]

    forward_scores = predict_multi_gpu(model_path, forward_data, n_gpus=n_gpus, batch_size=batch_size)
    reverse_scores = predict_multi_gpu(model_path, reverse_data, n_gpus=n_gpus, batch_size=batch_size)

    return {
        "src_tgt": forward_scores,
        "tgt_src": reverse_scores,
    }


def prep_opus_100_raw(
    raw_data_dir=None,
    n_gpus=8,
    comet_batch_size=256,
):
    from comet import download_model

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    logging.info("Using COMET model: Unbabel/wmt22-cometkiwi-da (multi-GPU)")

    lang_map = {
        "zh": "Chinese", "ja": "Japanese", "it": "Italian", "ru": "Russian",
        "es": "Spanish", "tr": "Turkish", "fr": "French", "pl": "Polish",
        "en": "English", "ar": "Arabic", "uk": "Ukrainian", "pt": "Portuguese",
        "sk": "Slovak", "de": "German", "hu": "Hungarian", "el": "Greek",
        "ko": "Korean", "vi": "Vietnamese", "th": "Thai", "sv": "Swedish",
        "hr": "Croatian", "is": "Icelandic", "lt": "Lithuanian",
        "ms": "Malay", "id": "Indonesian", "si": "Sinhala", "hi": "Hindi",
        "bn": "Bengali", "ur": "Urdu", "fa": "Persian", "kk": "Kazakh",
        "ro": "Romanian", "bg": "Bulgarian", "cs": "Czech", "da": "Danish",
        "lv": "Latvian", "nl": "Dutch", "et": "Estonian", "fi": "Finnish",
    }

    other_langs = [k for k in lang_map if k != "en"]
    configs = [f"en-{lang}" for lang in other_langs]
    reverse_load = {"en-ar": "ar-en", "en-de": "de-en", "en-el": "el-en", "en-bn": "bn-en", 
                    "en-bg": "bg-en", "en-cs": "cs-en", "en-da": "da-en"}

    datasets = {}
    num_proc = 32

    for config in configs:
        key = reverse_load.get(config, config)
        logging.info(f"Loading OPUS-100 config: {config} (key={key})")
        ds = load_dataset("Helsinki-NLP/opus-100", key, split="train")

        lang1, lang2 = config.split("-")

        def extract_translations(batch, lang1, lang2):
            sources = [item[lang1] for item in batch["translation"]]
            targets = [item[lang2] for item in batch["translation"]]
            return {"source": sources, "target": targets}

        ds = ds.map(
            extract_translations,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=num_proc,
            fn_kwargs={"lang1": lang1, "lang2": lang2},
            desc=f"Extracting translations for {config}",
        )

        logging.info(f"Calculating COMET scores for {config}...")
        comet_scores = predict_bidirectional_comet_scores(
            model_path,
            ds["source"],
            ds["target"],
            n_gpus=n_gpus,
            batch_size=comet_batch_size,
        )
        ds = ds.add_column("comet_score_src_tgt", comet_scores["src_tgt"])
        ds = ds.add_column("comet_score_tgt_src", comet_scores["tgt_src"])

        datasets[config] = ds
        logging.info(f"Processed {config} with {len(ds)} samples.")

    final_ds = DatasetDict(datasets)
    output_path = _dataset_path(raw_data_dir or RAW_DATA_DIR, "raw_opus_100")
    _save_dataset(final_ds, output_path)
    logging.info(
        f"Saved raw dataset (total = {sum(len(ds) for ds in final_ds.values())}) "
        f"to {output_path}"
    )


def prep_opus_mix(tokenizer, raw_data_dir=None, output_dir=None, eval_data_dir=None):
    """Create the filtered, prompted, and decontaminated OPUS-100 mix."""
    lang_map = {
        "zh": "Chinese", "ja": "Japanese", "it": "Italian", "ru": "Russian",
        "es": "Spanish", "tr": "Turkish", "fr": "French", "pl": "Polish",
        "en": "English", "ar": "Arabic", "uk": "Ukrainian", "pt": "Portuguese",
        "sk": "Slovak", "de": "German", "hu": "Hungarian", "el": "Greek",
        "ko": "Korean", "vi": "Vietnamese", "th": "Thai", "sv": "Swedish",
        "hr": "Croatian", "is": "Icelandic", "lt": "Lithuanian",
        "ms": "Malay", "id": "Indonesian", "si": "Sinhala", "hi": "Hindi",
        "bn": "Bengali", "ur": "Urdu", "fa": "Persian", "kk": "Kazakh",
        "ro": "Romanian", "bg": "Bulgarian", "cs": "Czech", "da": "Danish",
        "lv": "Latvian", "nl": "Dutch", "et": "Estonian", "fi": "Finnish",
    }

    num_proc = 32
    all_datasets = []

    input_path = _dataset_path(raw_data_dir or RAW_DATA_DIR, "raw_opus_100")
    opus_ds = load_from_disk(str(input_path))
    logging.info(f"Loaded raw_opus_100 with configs: {list(opus_ds.keys())}")

    for config in opus_ds.keys():
        logging.info(f"Loading OPUS config: {config}")
        ds = opus_ds[config]
        
        lang1, lang2 = config.split("-")
        logging.info(f"Processing {config} with {len(ds)} samples.")

        # COMET filtering
        ds = ds.filter(
            lambda x: (x["comet_score_src_tgt"] + x["comet_score_tgt_src"]) / 2.0 >= 0.6,
            num_proc=num_proc,
            desc=f"COMET filtering for {config}",
        )
        logging.info(f"Size after COMET filtering with mean score >= 0.6 for {config}: {len(ds)}")

        # APPROXIMATE deduplication via MinHash + LSH
        num_perm = 128
        lsh_threshold = 0.8
        mh_shingle_k = 4

        def compute_minhash(example, num_perm, shingle_k):
            text = example["source"] + " ||| " + example["target"]
            m = MinHash(num_perm=num_perm)
            for i in range(len(text) - shingle_k + 1):
                m.update(text[i:i + shingle_k].encode("utf-8"))
            return {"_mh_sig": m.hashvalues.tolist()}

        ds = ds.map(
            compute_minhash,
            num_proc=num_proc,
            fn_kwargs={"num_perm": num_perm, "shingle_k": mh_shingle_k},
            desc=f"Computing MinHash signatures for {config}",
        )

        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        duplicates = set()
        mh_batch_size = 10_000

        for start in tqdm(range(0, len(ds), mh_batch_size), desc=f"LSH dedup for {config}"):
            sigs = ds[start:start + mh_batch_size]["_mh_sig"]
            for i, sig in enumerate(sigs):
                idx = start + i
                m = LeanMinHash(seed=1, hashvalues=np.array(sig, dtype=np.uint64))
                if lsh.query(m):
                    duplicates.add(idx)
                else:
                    lsh.insert(idx, m)

        keep = [i for i in range(len(ds)) if i not in duplicates]
        ds = ds.select(keep)
        ds = ds.remove_columns(["_mh_sig"])
        logging.info(f"Size after MinHash dedup for {config}: {len(ds)} (removed {len(duplicates)})")

        def process_fwd(batch):
            messages = []
            for s, t in zip(batch["source"], batch["target"]):
                messages.append(get_mt_prompt(lang_map[lang1], lang_map[lang2], s, t))
            return {
                "messages": messages,
                "source": list(batch["source"]),
                "target": list(batch["target"]),
                "src_lang": [lang1] * len(messages),
                "tgt_lang": [lang2] * len(messages),
            }

        ds = ds.map(process_fwd, batched=True, num_proc=num_proc, remove_columns=ds.column_names, desc=f"Creating prompts for {config}")
        all_datasets.append(ds)

    # Concatenate
    final_ds = concatenate_datasets(all_datasets)
    logging.info(f"Dataset size before deduplication: {len(final_ds)}")

    # Stage 1: exact deduplication
    def get_hash(example):
        return {"hash": hashlib.md5(json.dumps(example["messages"], sort_keys=True).encode("utf-8")).hexdigest()}

    final_ds = final_ds.map(get_hash, num_proc=num_proc, desc="Hashing for exact deduplication")

    unique_indices = []
    seen_hashes = set()
    hashes = final_ds["hash"]
    for idx, h in tqdm(enumerate(hashes), desc="Exact deduplicating", total=len(hashes)):
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(idx)

    final_ds = final_ds.select(unique_indices)
    final_ds = final_ds.remove_columns(["hash"])
    logging.info(f"Size after exact deduplication: {len(final_ds)}")

    # Stage 2: BPE subword unigram MinHash dedup for semantic diversity
    final_ds, n_removed = _bpe_minhash_dedup(
        final_ds,
        tokenizer,
        num_perm=256,
        lsh_threshold=0.8,
        num_proc=num_proc,
        eval_data_dir=eval_data_dir,
    )
    logging.info(f"Size after BPE MinHash dedup: {len(final_ds)} (removed {n_removed})")

    output_path = _dataset_path(output_dir or OUTPUT_DIR, "latest_opus_100_mix")
    _save_dataset(final_ds, output_path)
    logging.info(f"Saved final dataset to {output_path}")

def prep_instruct_mix(output_dir=None):

    hf_ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", 
        split=[
            "smoltalk_multilingual_8languages_lang_5_no_think",
            "smoltalk_smollm3_systemchats_30k_no_think",
            "smoltalk_smollm3_everyday_conversations_no_think",
            "smoltalk_smollm3_explore_instruct_rewriting_no_think",
            "smoltalk_smollm3_smol_rewrite_no_think",
            "smoltalk_smollm3_smol_summarize_no_think",
        ]
    )
    hf_ds = concatenate_datasets(hf_ds)
    hf_ds = hf_ds.map(get_instruct_prompt, batched=True, num_proc=32, remove_columns=hf_ds.column_names, desc=f"Creating prompts:")

    dolci_ds = load_dataset("allenai/Dolci-Instruct-SFT-No-Tools")["train"]
    dolci_ds = dolci_ds.map(
        lambda x: {"messages": [{"role": "system", "content": "You are a helpful assistant."}] + x["messages"]}, 
        batched=False, 
        num_proc=32, 
        remove_columns=dolci_ds.column_names, desc=f"Creating prompts:"    
    )

    final_ds = concatenate_datasets([hf_ds, dolci_ds])
    logging.info(f"Final dataset size before deduplication: {len(final_ds)}")

    # Deduplication
    def get_hash(example):
        return {"hash": hashlib.md5(json.dumps(example["messages"], sort_keys=True).encode("utf-8")).hexdigest()}

    final_ds = final_ds.map(get_hash, num_proc=32, desc="Hashing for deduplication")
    
    unique_indices = []
    seen_hashes = set()
    hashes = final_ds["hash"]
    for idx, h in tqdm(enumerate(hashes), desc="Deduplicating", total=len(hashes)):
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(idx)
            
    final_ds = final_ds.select(unique_indices)
    final_ds = final_ds.remove_columns(["hash"])
    logging.info(f"Final dataset size after deduplication: {len(final_ds)}")

    output_path = _dataset_path(output_dir or OUTPUT_DIR, "latest_instruct_mix")
    _save_dataset(final_ds, output_path)
    logging.info(f"Saved final dataset to {output_path}")


def prep_human_mix(tokenizer, output_dir=None, eval_data_dir=None):

    smol_lang_map = {
        "af": ("afr", "Afrikaans"),
        "am": ("amh", "Amharic"),
        "ar-MA": ("ara", "Arabic"),
        "ha": ("hau", "Hausa"),
        "ig": ("ibo", "Igbo"),
        "lg": ("lug", "Luganda"),
        "ln": ("lin", "Lingala"),
        "mg": ("mlg", "Malagasy"),
        "ny": ("nya", "Nyanja"),
        "om": ("orm", "Oromo"),
        "pcm": ("pcm", "Nigerian Pidgin"),
        "rw": ("kin", "Kinyarwanda"),
        "sn": ("sna", "Shona"),
        "so": ("som", "Somali"),
        "st": ("sot", "Southern Sotho"),
        "sw": ("swh", "Swahili"),
        "tn": ("tsn", "Tswana"),
        "wo": ("wol", "Wolof"),
        "xh": ("xho", "Xhosa"),
        "yo": ("yor", "Yoruba"),
        "zu": ("zul", "Zulu"),
    }

    num_proc = 4
    all_datasets = []

    for smol_code, (three_char, lang_name) in smol_lang_map.items():
        pair_datasets = []

        # SmolDoc: document-level translations with aligned sentence lists
        try:
            smoldoc = load_dataset("google/smol", f"smoldoc__en_{smol_code}", split="train")

            def flatten_smoldoc(batch):
                sources, targets = [], []
                for srcs, trgs in zip(batch["srcs"], batch["trgs"]):
                    sources.extend(srcs)
                    targets.extend(trgs)
                return {"source": sources, "target": targets}

            smoldoc = smoldoc.map(
                flatten_smoldoc, batched=True, num_proc=num_proc,
                remove_columns=smoldoc.column_names,
                desc=f"Flattening SmolDoc for en-{smol_code}",
            )
            pair_datasets.append(smoldoc)
            logging.info(f"SmolDoc en-{smol_code}: {len(smoldoc)} sentence pairs")
        except Exception as e:
            logging.warning(f"Could not load SmolDoc for en-{three_char}: {e}")

        if not pair_datasets:
            logging.warning(f"No SMOL data loaded for en-{smol_code}")
            continue

        combined = concatenate_datasets(pair_datasets)
        logging.info(f"Combined en-{smol_code}: {len(combined)} sentence pairs")

        # en -> xx prompts
        def make_fwd_prompts(batch, src_lang, tgt_lang):
            messages = []
            for s, t in zip(batch["source"], batch["target"]):
                messages.append(get_mt_prompt(src_lang, tgt_lang, s, t))
            return {
                "messages": messages,
                "source": list(batch["source"]),
                "target": list(batch["target"]),
                "src_lang": ["eng"] * len(messages),
                "tgt_lang": [three_char] * len(messages),
            }

        fwd = combined.map(
            make_fwd_prompts, batched=True, num_proc=num_proc,
            remove_columns=combined.column_names,
            fn_kwargs={"src_lang": "English", "tgt_lang": lang_name},
            desc=f"Creating eng->{smol_code} ({three_char}) prompts",
        )
        all_datasets.append(fwd)

        # xx -> en prompts (swap source and target text)
        def make_rev_prompts(batch, src_lang, tgt_lang):
            messages = []
            for s, t in zip(batch["target"], batch["source"]):
                messages.append(get_mt_prompt(src_lang, tgt_lang, s, t))
            return {
                "messages": messages,
                "source": list(batch["target"]),
                "target": list(batch["source"]),
                "src_lang": [three_char] * len(messages),
                "tgt_lang": ["eng"] * len(messages),
            }

        rev = combined.map(
            make_rev_prompts, batched=True, num_proc=num_proc,
            remove_columns=combined.column_names,
            fn_kwargs={"src_lang": lang_name, "tgt_lang": "English"},
            desc=f"Creating {smol_code} ({three_char})->eng prompts",
        )
        all_datasets.append(rev)

    # AfriDocMT: multi-parallel human translations (health + tech subsets)
    afridocmt_lang_map = {
        "am": ("amh", "Amharic"),
        "ha": ("hau", "Hausa"),
        "sw": ("swh", "Swahili"),
        "yo": ("yor", "Yoruba"),
        "zu": ("zul", "Zulu"),
    }

    for subset in ["health", "tech"]:
        try:
            splits = load_dataset(
                "masakhane/AfriDocMT", subset,
                split=["train", "validation", "test"]
            )
            afridocmt_ds = concatenate_datasets(splits)
            logging.info(f"AfriDocMT {subset}: {len(afridocmt_ds)} rows")

            for lang_code, (three_char, lang_name) in afridocmt_lang_map.items():
                pair_ds = afridocmt_ds.select_columns(["en", lang_code])
                pair_ds = pair_ds.rename_column("en", "source")
                pair_ds = pair_ds.rename_column(lang_code, "target")

                def make_afridocmt_fwd(batch, src_lang, tgt_lang):
                    messages = []
                    for s, t in zip(batch["source"], batch["target"]):
                        messages.append(get_mt_prompt(src_lang, tgt_lang, s, t))
                    return {
                        "messages": messages,
                        "source": list(batch["source"]),
                        "target": list(batch["target"]),
                        "src_lang": ["eng"] * len(messages),
                        "tgt_lang": [three_char] * len(messages),
                    }

                fwd = pair_ds.map(
                    make_afridocmt_fwd, batched=True, num_proc=num_proc,
                    remove_columns=pair_ds.column_names,
                    fn_kwargs={"src_lang": "English", "tgt_lang": lang_name},
                    desc=f"AfriDocMT {subset} eng->{three_char} prompts"
                )
                all_datasets.append(fwd)

                def make_afridocmt_rev(batch, src_lang, tgt_lang):
                    messages = []
                    for s, t in zip(batch["target"], batch["source"]):
                        messages.append(get_mt_prompt(src_lang, tgt_lang, s, t))
                    return {
                        "messages": messages,
                        "source": list(batch["target"]),
                        "target": list(batch["source"]),
                        "src_lang": [three_char] * len(messages),
                        "tgt_lang": ["eng"] * len(messages),
                    }

                rev = pair_ds.map(
                    make_afridocmt_rev, batched=True, num_proc=num_proc,
                    remove_columns=pair_ds.column_names,
                    fn_kwargs={"src_lang": lang_name, "tgt_lang": "English"},
                    desc=f"AfriDocMT {subset} {three_char}->eng prompts"
                )
                all_datasets.append(rev)

                logging.info(f"AfriDocMT {subset} en-{three_char}: {len(fwd)} fwd + {len(rev)} rev pairs")
        except Exception as e:
            logging.warning(f"Could not load AfriDocMT {subset}: {e}")

    final_ds = concatenate_datasets(all_datasets)
    logging.info(f"Total dataset size before filtering: {len(final_ds)}")

    # Stage 1: exact deduplication
    def get_hash(example):
        return {"hash": hashlib.md5(json.dumps(example["messages"], sort_keys=True).encode("utf-8")).hexdigest()}

    final_ds = final_ds.map(get_hash, num_proc=num_proc, desc="Hashing for exact deduplication")

    unique_indices = []
    seen_hashes = set()
    hashes = final_ds["hash"]
    for idx, h in tqdm(enumerate(hashes), desc="Exact deduplicating", total=len(hashes)):
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(idx)

    final_ds = final_ds.select(unique_indices)
    final_ds = final_ds.remove_columns(["hash"])
    logging.info(f"Size after exact deduplication: {len(final_ds)}")

    # Stage 2: BPE subword unigram MinHash dedup for semantic diversity
    final_ds, n_removed = _bpe_minhash_dedup(
        final_ds,
        tokenizer,
        num_perm=256,
        lsh_threshold=0.8,
        num_proc=num_proc,
        eval_data_dir=eval_data_dir,
    )
    logging.info(f"Size after BPE MinHash dedup: {len(final_ds)} (removed {n_removed})")

    output_path = _dataset_path(output_dir or OUTPUT_DIR, "latest_human_mix")
    _save_dataset(final_ds, output_path)
    logging.info(f"Saved final dataset to {output_path}")


def prep_afri_nllb_mix(tokenizer, output_dir=None, eval_data_dir=None):
    lang_map = {
        "eng_Latn": "English",
        "amh_Ethi": "Amharic",
        "hau_Latn": "Hausa",
        "ibo_Latn": "Igbo",
        "lin_Latn": "Lingala",
        "som_Latn": "Somali",
        "swh_Latn": "Swahili",
        "yor_Latn": "Yoruba",
        "zul_Latn": "Zulu",
        "afr_Latn": "Afrikaans",
        "wol_Latn": "Wolof",
        "arz_Arab": "Egyptian Arabic",
        "arb_Arab": "Arabic",
        "fra_Latn": "French",
        "por_Latn": "Portuguese",
        "spa_Latn": "Spanish",
    }

    num_proc = 32
    ds = load_dataset("AfriNLP/AfriNLLB-train", split="train")
    logging.info(f"Loaded AfriNLLB-train with {len(ds)} samples.")

    def build_prompts(batch):
        messages = []
        for src, tgt, src_lang, tgt_lang in zip(
            batch["source"], batch["target"], batch["src_lang"], batch["tgt_lang"]
        ):
            src_name = lang_map.get(src_lang, src_lang)
            tgt_name = lang_map.get(tgt_lang, tgt_lang)
            messages.append(get_mt_prompt(src_name, tgt_name, src, tgt))
        return {
            "messages": messages,
            "source": list(batch["source"]),
            "target": list(batch["target"]),
            "src_lang": list(batch["src_lang"]),
            "tgt_lang": list(batch["tgt_lang"]),
        }

    drop_cols = [c for c in ds.column_names if c not in ("source", "target", "src_lang", "tgt_lang")]
    ds = ds.map(
        build_prompts,
        batched=True,
        num_proc=num_proc,
        remove_columns=drop_cols,
        desc="Building AfriNLLB prompts",
    )

    def get_hash(example):
        return {"hash": hashlib.md5(json.dumps(example["messages"], sort_keys=True).encode("utf-8")).hexdigest()}

    ds = ds.map(get_hash, num_proc=num_proc, desc="Hashing for exact deduplication")

    unique_indices = []
    seen_hashes = set()
    hashes = ds["hash"]
    for idx, h in tqdm(enumerate(hashes), desc="Exact deduplicating", total=len(hashes)):
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(idx)

    ds = ds.select(unique_indices)
    ds = ds.remove_columns(["hash"])
    logging.info(f"Size after exact deduplication: {len(ds)}")

    # Stage 2: BPE subword unigram MinHash dedup + decontamination against eval test sets
    ds, n_removed = _bpe_minhash_dedup(
        ds,
        tokenizer,
        num_perm=256,
        lsh_threshold=0.8,
        num_proc=num_proc,
        eval_data_dir=eval_data_dir,
    )
    logging.info(f"Size after BPE MinHash dedup: {len(ds)} (removed {n_removed})")

    output_path = _dataset_path(output_dir or OUTPUT_DIR, "latest_afri_nllb_mix")
    _save_dataset(ds, output_path)
    logging.info(f"Saved final dataset ({len(ds)} samples) to {output_path}")


def prep_afri_instruct(output_dir=None):

    # masakhane/african-ultrachat
    target_languages = {
        "Amharic",
        "Hausa",
        "Igbo",
        "Kinyarwanda",
        "Sesotho",
        "Shona",
        "Somali",
        "Swahili",
        "Xhosa",
        "Yoruba",
        "Zulu",
    }
    num_proc = 16

    def _no_empty_user_assistant(x):
        has_assistant = False
        for msg in x["messages"]:
            if msg["role"] in ("user", "assistant"):
                if not msg.get("content", "").strip():
                    return False
                if msg["role"] == "assistant":
                    has_assistant = True
        return has_assistant

    splits = load_dataset("masakhane/african-ultrachat", split=["train", "test"])
    ultrachat_ds = concatenate_datasets(splits)
    logging.info(f"Loaded african-ultrachat with {len(ultrachat_ds)} samples.")

    ultrachat_ds = ultrachat_ds.filter(lambda x: x["language"] in target_languages, desc="Filtering by language", num_proc=num_proc)
    logging.info(f"Dataset size after language filtering: {len(ultrachat_ds)}")

    ultrachat_ds = ultrachat_ds.select_columns(["messages"])

    ultrachat_ds = ultrachat_ds.map(
        lambda x: {"messages": [{"role": "system", "content": "You are a helpful assistant."}] + x["messages"]},
        num_proc=num_proc,
        desc="Adding system prompt",
    )

    ultrachat_ds = ultrachat_ds.filter(_no_empty_user_assistant, desc="Filtering empty ultrachat responses", num_proc=num_proc)
    logging.info(f"african-ultrachat after empty response filtering: {len(ultrachat_ds)}")

    # masakhane/african-translated-alpaca (files are per-language).
    # Keep as many of the target African language codes as are available.
    target_afri_lang_codes = [
        "afr", "amh", "ibo", "kin", "lin", "lug", "mlg", "nya", "sna", "som",
        "sot", "swh", "wol", "xho", "yor", "zul",
        # Desired but currently unavailable in this dataset:
        "hau", "orm", "pcm", "tsn",
    ]
    # Code normalization between internal conventions and dataset file names.
    alpaca_code_alias = {
        "mlg": "plt",   # Malagasy
        "swa": "swh",   # Swahili alias
        "ara": "arb",   # Arabic alias (not part of current target_afri_lang_codes)
    }
    # Snapshot of available language file codes under train/*.json.
    alpaca_available_codes = {
        "afr", "amh", "arb", "eng", "ewe", "fra", "gaz", "ibo", "kin", "lin", "lug", "nya",
        "plt", "por", "sna", "som", "sot", "swh", "tir", "wol", "xho", "yor", "zul",
    }

    selected_alpaca_codes = []
    missing_alpaca_codes = []
    for code in target_afri_lang_codes:
        resolved = alpaca_code_alias.get(code, code)
        if resolved in alpaca_available_codes:
            selected_alpaca_codes.append(resolved)
        else:
            missing_alpaca_codes.append(code)
    selected_alpaca_codes = sorted(set(selected_alpaca_codes))

    logging.info(
        "african-translated-alpaca target=%d available=%d missing=%s",
        len(target_afri_lang_codes),
        len(selected_alpaca_codes),
        missing_alpaca_codes,
    )
    alpaca_lang_files = [f"train/{lang}.json" for lang in selected_alpaca_codes]
    alpaca_ds = load_dataset("masakhane/african-translated-alpaca", data_files=alpaca_lang_files, split="train")
    logging.info(f"Loaded african-translated-alpaca with {len(alpaca_ds)} samples (target languages only).")

    def build_alpaca_messages(batch):
        messages = []
        for instruction, inp, output in zip(batch["instruction"], batch["input"], batch["output"]):
            user_content = f"{instruction}\n\n{inp}" if inp and inp.strip() else instruction
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ])
        return {"messages": messages}

    alpaca_ds = alpaca_ds.map(
        build_alpaca_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=alpaca_ds.column_names,
        desc="Building alpaca chat messages",
    )

    alpaca_ds = alpaca_ds.filter(_no_empty_user_assistant, desc="Filtering empty alpaca responses", num_proc=num_proc)
    logging.info(f"african-translated-alpaca after empty response filtering: {len(alpaca_ds)}")

    # ptrdvn/kakugo-<lang>: include as many of the target African language codes as available.
    kakugo_code_alias = {
        "mlg": "plt",  # Malagasy
        "swh": "swa",  # Swahili variant used by kakugo repos
    }
    kakugo_available_codes = {
        # Snapshot of currently available kakugo repos relevant to our African target set.
        "amh", "hau", "ibo", "kin", "nya", "plt", "sna", "sot", "swa", "xho", "yor", "zul",
    }
    selected_kakugo_codes = []
    missing_kakugo_codes = []
    for code in target_afri_lang_codes:
        resolved = kakugo_code_alias.get(code, code)
        if resolved in kakugo_available_codes:
            selected_kakugo_codes.append(resolved)
        else:
            missing_kakugo_codes.append(code)
    selected_kakugo_codes = sorted(set(selected_kakugo_codes))
    logging.info(
        "kakugo target=%d available=%d missing=%s",
        len(target_afri_lang_codes),
        len(selected_kakugo_codes),
        missing_kakugo_codes,
    )

    kakugo_datasets = []
    for lang_code in selected_kakugo_codes:
        kakugo_ds = load_dataset(f"ptrdvn/kakugo-{lang_code}", split="train")
        logging.info(f"Loaded kakugo-{lang_code} with {len(kakugo_ds)} samples.")
        kakugo_ds = kakugo_ds.map(
            lambda x: {"messages": [{"role": "system", "content": x["system"]}] + x["messages"]},
            num_proc=num_proc,
            remove_columns=kakugo_ds.column_names,
            desc=f"Building kakugo-{lang_code} messages",
        )
        kakugo_ds = kakugo_ds.filter(
            _no_empty_user_assistant,
            desc=f"Filtering empty kakugo-{lang_code} responses",
            num_proc=num_proc,
        )
        logging.info(f"kakugo-{lang_code} after empty response filtering: {len(kakugo_ds)}")
        kakugo_datasets.append(kakugo_ds)

    if not kakugo_datasets:
        raise ValueError("No Kakugo datasets available for the configured target languages.")
    kakugo_ds = concatenate_datasets(kakugo_datasets)
    logging.info(f"Combined Kakugo dataset size: {len(kakugo_ds)}")

    # https://huggingface.co/datasets/CohereLabs/aya_dataset
    aya_ds = load_dataset("CohereLabs/aya_dataset", split="train")
    logging.info(f"Loaded aya_dataset with {len(aya_ds)} samples.")

    def build_aya_messages(batch):
        messages = []
        for inp, tgt in zip(batch["inputs"], batch["targets"]):
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": tgt},
            ])
        return {"messages": messages}

    aya_ds = aya_ds.map(
        build_aya_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=aya_ds.column_names,
        desc="Building aya_dataset messages",
    )

    aya_ds = aya_ds.filter(_no_empty_user_assistant, desc="Filtering empty aya responses", num_proc=num_proc)
    logging.info(f"aya_dataset after empty response filtering: {len(aya_ds)}")

    # https://huggingface.co/collections/michsethowusu/afri-code-datasets
    afri_code_lang_map = {
        "afr": "afrikaans",
        "amh": "amharic",
        "hau": "hausa",
        "ibo": "igbo",
        "kin": "kinyarwanda",
        "lin": "lingala",
        "lug": "luganda",
        "mlg": "malagasy",
        "orm": "oromo",
        "sna": "shona",
        "som": "somali",
        "sot": "sesotho",
        "swa": "swahili",
        "swh": "swahili",
        "tsn": "tswana",
        "wol": "wolof",
        "xho": "xhosa",
        "yor": "yoruba",
        "zul": "zulu",
    }
    afri_code_available_langs = {
        "acholi", "afar", "afrikaans", "alur", "amharic", "bambara", "baoule", "bemba", "dinka",
        "dombe", "dyula", "fon", "fulani", "hausa", "igbo", "kanuri", "kiga", "kikongo", "kinyarwanda", "kituba",
        "krio", "lingala", "luganda", "luo", "malagasy", "mauritian-creole", "ndebele-south", "nuer", "oromo",
        "rundi", "sango", "sepedi", "sesotho", "seychellois-creole", "shona", "somali", "susu", "swahili",
        "swati", "tamazight-tifinagh", "tigrinya", "tiv", "tshiluba", "tsonga", "tswana", "tumbuka", "venda",
        "wolof", "xhosa", "yoruba", "zulu",
    }
    afri_code_langs = []
    missing_afri_code_langs = []
    for code in target_afri_lang_codes:
        dataset_lang = afri_code_lang_map.get(code)
        if dataset_lang and dataset_lang in afri_code_available_langs:
            afri_code_langs.append(dataset_lang)
        else:
            missing_afri_code_langs.append(code)
    afri_code_langs = sorted(set(afri_code_langs))
    logging.info(
        "afri-code target=%d available=%d missing=%s",
        len(target_afri_lang_codes),
        len(afri_code_langs),
        missing_afri_code_langs,
    )

    afri_code_datasets = []
    afri_code_per_lang_sample_size = 30_000
    for lang in afri_code_langs:
        lang_ds = load_dataset(f"michsethowusu/Code-170k-{lang}", split="train")
        original_size = len(lang_ds)
        sample_size = min(afri_code_per_lang_sample_size, original_size)
        lang_ds = lang_ds.shuffle(seed=42).select(range(sample_size))
        logging.info(
            f"Code-170k-{lang}: dataset size={original_size}, sampled={sample_size} (random)"
        )
        afri_code_datasets.append(lang_ds)

    afri_code_ds = concatenate_datasets(afri_code_datasets)
    logging.info(f"Afri-code combined: {len(afri_code_ds)} samples.")

    def build_afri_code_messages(batch):
        role_map = {"human": "user", "gpt": "assistant"}
        messages = []
        for convs in batch["conversations"]:
            msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            for turn in convs:
                msgs.append({"role": role_map[turn["from"]], "content": turn["value"]})
            messages.append(msgs)
        return {"messages": messages}

    afri_code_ds = afri_code_ds.map(
        build_afri_code_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afri_code_ds.column_names,
        desc="Building afri-code messages",
    )

    afri_code_ds = afri_code_ds.filter(_no_empty_user_assistant, desc="Filtering empty afri-code responses", num_proc=num_proc)
    logging.info(f"Afri-code after empty response filtering: {len(afri_code_ds)}")

    # https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment
    afrisenti_langs = ["amh", "hau", "ibo", "kin", "orm", "pcm", "swa", "yor"]
    afrisenti_label_names = ["positive", "neutral", "negative"]
    afrisenti_datasets = []
    loaded_afrisenti_langs = []
    missing_afrisenti_langs = []
    for lang in afrisenti_langs:
        try:
            lang_ds = load_dataset(
                "shmuhammad/AfriSenti-twitter-sentiment",
                data_dir=lang,
                split="train+validation+test",
                revision="refs/convert/parquet",
            )
            logging.info(f"AfriSenti-{lang}: {len(lang_ds)} samples")
            afrisenti_datasets.append(lang_ds)
            loaded_afrisenti_langs.append(lang)
        except Exception as exc:
            logging.warning(f"Skipping AfriSenti-{lang}: {exc}")
            missing_afrisenti_langs.append(lang)

    if not afrisenti_datasets:
        raise ValueError("No AfriSenti language subsets could be loaded.")
    logging.info(
        "AfriSenti target=%d loaded=%d missing=%s",
        len(afrisenti_langs),
        len(loaded_afrisenti_langs),
        missing_afrisenti_langs,
    )

    afrisenti_ds = concatenate_datasets(afrisenti_datasets)
    logging.info(f"AfriSenti combined: {len(afrisenti_ds)} samples")

    def build_afrisenti_messages(batch):
        messages = []
        for tweet, label in zip(batch["tweet"], batch["label"]):
            label_str = afrisenti_label_names[label]
            messages.append([
                {"role": "system", "content": "You are a sentiment classifier. Analyze the sentiment of the given text and respond with exactly one word: positive, negative, or neutral."},
                {"role": "user", "content": tweet},
                {"role": "assistant", "content": label_str},
            ])
        return {"messages": messages}

    afrisenti_ds = afrisenti_ds.map(
        build_afrisenti_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afrisenti_ds.column_names,
        desc="Building AfriSenti messages",
    )

    afrisenti_ds = afrisenti_ds.filter(_no_empty_user_assistant, desc="Filtering empty AfriSenti responses", num_proc=num_proc)
    logging.info(f"AfriSenti after empty response filtering: {len(afrisenti_ds)}")

    # https://huggingface.co/datasets/masakhane/masakhanews
    masakhanews_code_alias = {
        "swh": "swa",  # Swahili alias
    }
    masakhanews_available_codes = {
        # Snapshot from dataset repo dirs (plus known converted subsets).
        "amh", "eng", "fra", "hau", "ibo", "lin", "lug", "orm",
        "pcm", "run", "sna", "som", "swa", "tir", "xho", "yor",
    }
    masakhanews_langs = []
    missing_masakhanews_langs = []
    for code in target_afri_lang_codes:
        resolved = masakhanews_code_alias.get(code, code)
        if resolved in masakhanews_available_codes:
            masakhanews_langs.append(resolved)
        else:
            missing_masakhanews_langs.append(code)
    masakhanews_langs = sorted(set(masakhanews_langs))
    masakhanews_datasets = []
    loaded_masakhanews_langs = []
    for lang in masakhanews_langs:
        try:
            lang_ds = load_dataset(
                "masakhane/masakhanews",
                data_dir=lang,
                split="train+validation+test",
                revision="refs/convert/parquet",
            )
            logging.info(f"MasakhaNEWS-{lang}: {len(lang_ds)} samples")
            masakhanews_datasets.append(lang_ds)
            loaded_masakhanews_langs.append(lang)
        except Exception as exc:
            logging.warning(f"Skipping MasakhaNEWS-{lang}: {exc}")

    if not masakhanews_datasets:
        raise ValueError("No MasakhaNEWS language subsets could be loaded.")
    logging.info(
        "MasakhaNEWS target=%d selected=%d loaded=%d missing=%s",
        len(target_afri_lang_codes),
        len(masakhanews_langs),
        len(loaded_masakhanews_langs),
        missing_masakhanews_langs,
    )

    masakhanews_ds = concatenate_datasets(masakhanews_datasets)
    logging.info(f"MasakhaNEWS combined: {len(masakhanews_ds)} samples")

    masakhanews_ds = masakhanews_ds.filter(
        lambda x: len(x["headline"].strip()) > 0 and len(x["text"].strip()) > 0,
        desc="Filtering empty MasakhaNEWS entries", num_proc=num_proc,
    )
    logging.info(f"MasakhaNEWS after filtering empty entries: {len(masakhanews_ds)}")

    def build_masakhanews_messages(batch):
        messages = []
        for text, headline in zip(batch["text"], batch["headline"]):
            messages.append([
                {"role": "system", "content": "You are a news editor. Given a news article, generate a concise and informative headline that captures the main point of the article."},
                {"role": "user", "content": text},
                {"role": "assistant", "content": headline},
            ])
        return {"messages": messages}

    masakhanews_ds = masakhanews_ds.map(
        build_masakhanews_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=masakhanews_ds.column_names,
        desc="Building MasakhaNEWS messages",
    )

    masakhanews_ds = masakhanews_ds.filter(_no_empty_user_assistant, desc="Filtering empty MasakhaNEWS responses", num_proc=num_proc)
    logging.info(f"MasakhaNEWS after empty response filtering: {len(masakhanews_ds)}")

    # https://huggingface.co/datasets/csebuetnlp/xlsum
    xlsum_code_to_dir = {
        "amh": "amharic",
        "hau": "hausa",
        "ibo": "igbo",
        "orm": "oromo",
        "som": "somali",
        "swa": "swahili",
        "swh": "swahili",
        "yor": "yoruba",
    }
    xlsum_langs = []
    missing_xlsum_langs = []
    for code in target_afri_lang_codes:
        xlsum_dir = xlsum_code_to_dir.get(code)
        if xlsum_dir:
            xlsum_langs.append(xlsum_dir)
        else:
            missing_xlsum_langs.append(code)
    xlsum_langs = sorted(set(xlsum_langs))
    xlsum_datasets = []
    loaded_xlsum_langs = []
    for lang in xlsum_langs:
        try:
            lang_ds = load_dataset(
                "csebuetnlp/xlsum",
                data_dir=lang,
                split="train+validation+test",
                revision="refs/convert/parquet",
            )
            logging.info(f"XLSum-{lang}: {len(lang_ds)} samples")
            xlsum_datasets.append(lang_ds)
            loaded_xlsum_langs.append(lang)
        except Exception as exc:
            logging.warning(f"Skipping XLSum-{lang}: {exc}")

    if not xlsum_datasets:
        raise ValueError("No XLSum language subsets could be loaded.")
    logging.info(
        "XLSum target=%d selected=%d loaded=%d missing=%s",
        len(target_afri_lang_codes),
        len(xlsum_langs),
        len(loaded_xlsum_langs),
        missing_xlsum_langs,
    )

    xlsum_ds = concatenate_datasets(xlsum_datasets)
    logging.info(f"XLSum combined: {len(xlsum_ds)} samples")

    xlsum_ds = xlsum_ds.filter(
        lambda x: len(x["text"].strip()) > 0 and len(x["summary"].strip()) > 0,
        desc="Filtering empty XLSum entries", num_proc=num_proc,
    )
    logging.info(f"XLSum after filtering empty entries: {len(xlsum_ds)}")

    def build_xlsum_messages(batch):
        messages = []
        for text, summary in zip(batch["text"], batch["summary"]):
            messages.append([
                {"role": "system", "content": "You are a summarization assistant. Read the following text and produce a clear, concise summary that captures the key information."},
                {"role": "user", "content": text},
                {"role": "assistant", "content": summary},
            ])
        return {"messages": messages}

    xlsum_ds = xlsum_ds.map(
        build_xlsum_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=xlsum_ds.column_names,
        desc="Building XLSum messages",
    )

    xlsum_ds = xlsum_ds.filter(_no_empty_user_assistant, desc="Filtering empty XLSum responses", num_proc=num_proc)
    logging.info(f"XLSum after empty response filtering: {len(xlsum_ds)}")

    # https://huggingface.co/datasets/masakhane/AfriADR
    afriadr_langs = ["ibo", "wol", "yor"]
    afriadr_datasets = []
    for lang in afriadr_langs:
        lang_ds = load_dataset(
            "masakhane/AfriADR",
            data_dir=lang,
            split="train+validation+test",
            revision="refs/convert/parquet",
        )
        logging.info(f"AfriADR-{lang}: {len(lang_ds)} samples")
        afriadr_datasets.append(lang_ds)

    afriadr_ds = concatenate_datasets(afriadr_datasets)
    logging.info(f"AfriADR combined: {len(afriadr_ds)} samples")

    afriadr_ds = afriadr_ds.filter(
        lambda x: len(x["text"].strip()) > 0 and len(x["target"].strip()) > 0,
        desc="Filtering empty AfriADR entries", num_proc=num_proc,
    )
    logging.info(f"AfriADR after filtering empty entries: {len(afriadr_ds)}")

    def build_afriadr_messages(batch):
        messages = []
        for text, target in zip(batch["text"], batch["target"]):
            messages.append([
                {"role": "system", "content": "You are a diacritics restoration assistant. Given text with missing or simplified diacritical marks, restore the correct tone marks, accents, and diacritics."},
                {"role": "user", "content": text},
                {"role": "assistant", "content": target},
            ])
        return {"messages": messages}

    afriadr_ds = afriadr_ds.map(
        build_afriadr_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afriadr_ds.column_names,
        desc="Building AfriADR messages",
    )

    afriadr_ds = afriadr_ds.filter(_no_empty_user_assistant, desc="Filtering empty AfriADR responses", num_proc=num_proc)
    logging.info(f"AfriADR after empty response filtering: {len(afriadr_ds)}")

    # https://huggingface.co/datasets/masakhane/afrixnli
    afrixnli_langs = ["amh", "hau", "ibo", "kin", "lin", "lug", "orm", "sna", "swa", "wol", "xho", "yor", "zul"]
    afrixnli_label_names = ["entailment", "neutral", "contradiction"]
    afrixnli_datasets = []
    for lang in afrixnli_langs:
        lang_ds = load_dataset(
            "masakhane/afrixnli",
            data_dir=lang,
            split="validation+test",
            revision="refs/convert/parquet",
        )
        logging.info(f"AfriXNLI-{lang}: {len(lang_ds)} samples")
        afrixnli_datasets.append(lang_ds)

    afrixnli_ds = concatenate_datasets(afrixnli_datasets)
    logging.info(f"AfriXNLI combined: {len(afrixnli_ds)} samples")

    afrixnli_ds = afrixnli_ds.filter(
        lambda x: len(x["premise"].strip()) > 0 and len(x["hypothesis"].strip()) > 0,
        desc="Filtering empty AfriXNLI entries", num_proc=num_proc,
    )
    logging.info(f"AfriXNLI after filtering empty entries: {len(afrixnli_ds)}")

    def build_afrixnli_messages(batch):
        messages = []
        for premise, hypothesis, label in zip(batch["premise"], batch["hypothesis"], batch["label"]):
            label_str = afrixnli_label_names[label]
            messages.append([
                {"role": "system", "content": "You are a natural language inference assistant. Given a premise and a hypothesis, determine the relationship between them. Respond with exactly one word: entailment, neutral, or contradiction."},
                {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}"},
                {"role": "assistant", "content": label_str},
            ])
        return {"messages": messages}

    afrixnli_ds = afrixnli_ds.map(
        build_afrixnli_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afrixnli_ds.column_names,
        desc="Building AfriXNLI messages",
    )

    afrixnli_ds = afrixnli_ds.filter(_no_empty_user_assistant, desc="Filtering empty AfriXNLI responses", num_proc=num_proc)
    logging.info(f"AfriXNLI after empty response filtering: {len(afrixnli_ds)}")

    # https://huggingface.co/datasets/masakhane/afriqa
    afriqa_lang_map = {
        "hau": "Hausa",
        "ibo": "Igbo",
        "kin": "Kinyarwanda",
        "swa": "Swahili",
        "wol": "Wolof",
        "yor": "Yoruba",
        "zul": "Zulu",
    }
    afriqa_datasets = []
    for lang in afriqa_lang_map:
        lang_ds = load_dataset(
            "masakhane/afriqa",
            data_dir=lang,
            split="train+validation+test",
            revision="refs/convert/parquet",
        )
        logging.info(f"AfriQA-{lang}: {len(lang_ds)} samples")
        afriqa_datasets.append(lang_ds)

    afriqa_ds = concatenate_datasets(afriqa_datasets)
    logging.info(f"AfriQA combined: {len(afriqa_ds)} samples")

    def parse_answer_field(value):
        if isinstance(value, list):
            return value
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            return []

    def filter_afriqa_single_answer(x):
        answers = parse_answer_field(x["answers"])
        translated = parse_answer_field(x["translated_answer"])
        if len(answers) != 1 or len(translated) != 1:
            return False
        return bool(
            answers[0].strip() and translated[0].strip()
            and x["question"].strip() and x["translated_question"].strip()
        )

    afriqa_ds = afriqa_ds.filter(filter_afriqa_single_answer, desc="Filtering AfriQA to single-answer entries", num_proc=num_proc)
    logging.info(f"AfriQA after filtering to single-answer entries: {len(afriqa_ds)}")

    def extract_single_answers(x):
        return {
            "answer": parse_answer_field(x["answers"])[0].strip(),
            "translated_answer_text": parse_answer_field(x["translated_answer"])[0].strip(),
        }

    afriqa_ds = afriqa_ds.map(extract_single_answers, num_proc=num_proc, desc="Extracting AfriQA single answers")

    def build_afriqa_mt_messages(batch):
        messages = []
        for question, translated_question, lang in zip(
            batch["question"], batch["translated_question"], batch["lang"],
        ):
            lang_name = afriqa_lang_map[lang]
            messages.append(get_mt_prompt(lang_name, "English", question, translated_question))
            messages.append(get_mt_prompt("English", lang_name, translated_question, question))
        return {"messages": messages}

    afriqa_mt_ds = afriqa_ds.map(
        build_afriqa_mt_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afriqa_ds.column_names,
        desc="Building AfriQA question translation messages",
    )
    logging.info(f"AfriQA MT examples: {len(afriqa_mt_ds)}")

    def build_afriqa_qa_messages(batch):
        messages = []
        for question, answer, translated_question, translated_answer in zip(
            batch["question"], batch["answer"],
            batch["translated_question"], batch["translated_answer_text"],
        ):
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": translated_question},
                {"role": "assistant", "content": translated_answer},
            ])
        return {"messages": messages}

    afriqa_qa_ds = afriqa_ds.map(
        build_afriqa_qa_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afriqa_ds.column_names,
        desc="Building AfriQA regular QA messages",
    )
    logging.info(f"AfriQA QA examples: {len(afriqa_qa_ds)}")

    def build_afriqa_xlqa_messages(batch):
        messages = []
        for question, answer, translated_question, translated_answer, lang in zip(
            batch["question"], batch["answer"],
            batch["translated_question"], batch["translated_answer_text"],
            batch["lang"],
        ):
            lang_name = afriqa_lang_map[lang]
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{question}\n\nAnswer the question in English."},
                {"role": "assistant", "content": translated_answer},
            ])
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{translated_question}\n\nAnswer the question in {lang_name}."},
                {"role": "assistant", "content": answer},
            ])
        return {"messages": messages}

    afriqa_xlqa_ds = afriqa_ds.map(
        build_afriqa_xlqa_messages,
        batched=True,
        num_proc=num_proc,
        remove_columns=afriqa_ds.column_names,
        desc="Building AfriQA cross-lingual QA messages",
    )
    logging.info(f"AfriQA cross-lingual QA examples: {len(afriqa_xlqa_ds)}")

    afriqa_ds = concatenate_datasets([afriqa_mt_ds, afriqa_qa_ds, afriqa_xlqa_ds])
    logging.info(f"AfriQA total training examples: {len(afriqa_ds)}")

    afriqa_ds = afriqa_ds.filter(_no_empty_user_assistant, desc="Filtering empty AfriQA responses", num_proc=num_proc)
    logging.info(f"AfriQA after empty response filtering: {len(afriqa_ds)}")

    final_ds = concatenate_datasets([ultrachat_ds, alpaca_ds, kakugo_ds, aya_ds, afri_code_ds, afrisenti_ds, masakhanews_ds, xlsum_ds, afriadr_ds, afrixnli_ds, afriqa_ds])
    logging.info(f"Combined afri_chat dataset size: {len(final_ds)}")

    output_path = _dataset_path(output_dir or OUTPUT_DIR, "latest_afri_instruct_mix")
    _save_dataset(final_ds, output_path)
    logging.info(f"Saved final dataset ({len(final_ds)} samples) to {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Recreate the QVAC TranslatePsy training dataset mixes."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=(
            "all",
            "eval",
            "opus-raw",
            "opus",
            "instruct",
            "afri-instruct",
            "human",
            "afri-nllb",
        ),
        help="Dataset stages to build. 'opus' requires 'opus-raw' to have run first.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory for intermediate datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for final processed datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=Path,
        default=EVAL_DATA_DIR,
        help="Parent directory containing the four decontamination datasets.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-2B",
        help="Tokenizer used for BPE MinHash deduplication (default: %(default)s).",
    )
    parser.add_argument(
        "--comet-gpus",
        type=int,
        default=8,
        help="GPUs used by the OPUS raw COMET stage (default: %(default)s).",
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=256,
        help="Per-process COMET prediction batch size (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    args = _parse_args()

    random.seed(42)
    np.random.seed(42)
    requested = set(args.datasets)
    if "all" in requested:
        requested = {
            "eval",
            "opus-raw",
            "opus",
            "instruct",
            "afri-instruct",
            "human",
            "afri-nllb",
        }

    if "eval" in requested:
        prep_eval_sets(eval_data_dir=args.eval_data_dir)

    tokenizer = None
    if requested & {"opus", "human", "afri-nllb"}:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Keep OPUS stages ordered because the final mix consumes the raw artifact.
    if "opus-raw" in requested:
        prep_opus_100_raw(
            raw_data_dir=args.raw_data_dir,
            n_gpus=args.comet_gpus,
            comet_batch_size=args.comet_batch_size,
        )
    if "opus" in requested:
        prep_opus_mix(
            tokenizer=tokenizer,
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            eval_data_dir=args.eval_data_dir,
        )
    if "instruct" in requested:
        prep_instruct_mix(output_dir=args.output_dir)
    if "afri-instruct" in requested:
        prep_afri_instruct(output_dir=args.output_dir)
    if "human" in requested:
        prep_human_mix(
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            eval_data_dir=args.eval_data_dir,
        )
    if "afri-nllb" in requested:
        prep_afri_nllb_mix(
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            eval_data_dir=args.eval_data_dir,
        )


if __name__ == "__main__":
    main()
