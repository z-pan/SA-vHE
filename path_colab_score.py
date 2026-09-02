#!/usr/bin/env python3
"""Score tiles against the structure list. Runs on Colab; backend is swappable.

Two backends, meant to be used in that order:

  conch     CLIP-style image-text similarity. One forward pass per tile scores every
            structure at once, so a whole slide is minutes, and the output is a number
            per structure per tile -- rankable, reproducible, and able to go into
            supplementary material as it stands.
  medgemma  A chat VLM answering the same list as a 0-4 checklist. Much slower, so run
            it only on the tiles CONCH already shortlisted, to get wording for a figure
            caption and a second opinion on the top candidates.

Both write the same CSV shape, so path_report.py does not care which produced it.

Colab notes
-----------
MedGemma 1.5 4B (google/medgemma-1.5-4b-it, the January 2026 release that added
whole-slide histopathology) is about 8 GB in bf16 and fits a T4 without quantisation. CONCH and
TITAN need an access agreement with mahmoodlab; MedGemma needs the Health AI Developer
Foundations licence accepted on Hugging Face. Log in with huggingface_hub.login()
before either.

    !python path_colab_score.py --tiles /content/tiles_dir --backend conch --out conch.csv
    # caption bake-off, PROBES in path_structures.py; free, it shares the image pass
    !python path_colab_score.py --tiles /content/tiles_dir --backend conch --probes --out conch_probes.csv
    !python path_colab_score.py --tiles /content/tiles_dir --backend medgemma \\
        --shortlist conch.csv --top 60 --out medgemma.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_structures import (KEYS, PROBE_KEYS, PROBES, STRUCTURES,  # noqa: E402
                             VLM_INSTRUCTION)



def tile_path(tiles_dir, tid):
    """Tiles are PNG or JPEG depending on how they were cut; the id does not say
    which, and a set may legitimately mix the two."""
    for ext in ('.png', '.jpg'):
        p = os.path.join(tiles_dir, 'tiles', tid + ext)
        if os.path.exists(p):
            return p
    raise SystemExit(f'no tile image for {tid} in {tiles_dir}/tiles')

def load_index(d):
    return list(csv.DictReader(open(os.path.join(d, 'index.csv'), encoding='utf-8')))


def shortlist_ids(path, top, per_structure=True):
    """Top tiles from a previous scoring run.

    Taken per structure rather than by overall score, so a rare finding -- psammoma
    bodies, say -- still reaches the shortlist instead of being buried under whichever
    structure happens to score high everywhere.
    """
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    if not rows:
        raise SystemExit(f'{path} has no rows -- did the earlier run get cut short?')
    if not per_structure:
        rows.sort(key=lambda r: -max(float(r[k]) for k in KEYS if k in r))
        return [r['tile_id'] for r in rows[:top]]
    picked, per = [], max(1, top // max(len(KEYS), 1))
    for k in KEYS:
        if k not in rows[0]:
            continue
        for r in sorted(rows, key=lambda r: -float(r[k]))[:per]:
            if r['tile_id'] not in picked:
                picked.append(r['tile_id'])
    # Integer division plus the dedup leaves the quota short -- 18 structures at top=60
    # fills only 48. Spend the remainder on the best tiles overall rather than dropping
    # them, since the shortlist size is the VLM budget.
    if len(picked) < top:
        seen = set(picked)
        for r in sorted(rows, key=lambda r: -max(float(r[k]) for k in KEYS if k in r)):
            if r['tile_id'] not in seen:
                picked.append(r['tile_id']); seen.add(r['tile_id'])
            if len(picked) >= top:
                break
    return picked[:top]


def run_conch(tiles_dir, ids, model_name, batch=32, token=None, probes=False):
    """With probes=True the candidate captions in PROBES are scored as extra columns.

    They ride along on the same image pass, which is the entire cost of the run, so a
    caption bake-off is effectively free -- and, more to the point, the candidates and
    the incumbent then share one set of image embeddings, which is the only way the
    columns are comparable at all. path_report.py and path_locate.py select columns by
    name, so the extra ones are invisible to them unless asked for with --probes.
    """
    import torch
    from PIL import Image
    from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

    # MahmoodLab/conch is gated; without a token the download 401s well into the run.
    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16', model_name,
        **({'hf_auth_token': token} if token else {}))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval()
    tok = get_tokenizer()
    # transformers 5.x dropped batch_encode_plus, which CONCH's own tokenize()
    # still calls. __call__ takes the identical keywords, so a thin shim keeps
    # CONCH working without pinning transformers back and breaking MedGemma.
    if not hasattr(tok, 'batch_encode_plus'):
        tok.batch_encode_plus = lambda texts, **kw: tok(texts, **kw)
    entries = STRUCTURES + (PROBES if probes else [])
    keys = [s['key'] for s in entries]
    prompts = [s['prompt'] for s in entries]
    with torch.inference_mode():
        tfeat = model.encode_text(tokenize(texts=prompts, tokenizer=tok).to(dev))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    out = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        ims = torch.stack([preprocess(Image.open(
            tile_path(tiles_dir, t)).convert('RGB')) for t in chunk]).to(dev)
        with torch.inference_mode():
            f = model.encode_image(ims, proj_contrast=True, normalize=True)
            sim = (f @ tfeat.T).float().cpu().numpy()
        for t, row in zip(chunk, sim):
            out.append(dict(tile_id=t, **{k: round(float(v), 4) for k, v in zip(keys, row)}))
        print(f'  {min(i+batch, len(ids))}/{len(ids)}', flush=True)
    return out


def run_medgemma(tiles_dir, ids, model_name, max_new_tokens=256, out_path=None):
    """One tile at a time, flushed after each.

    A few seconds per tile means a shortlist is tens of minutes, which is long enough
    that a Colab disconnect is a real risk; writing as we go means a rerun resumes
    instead of starting over.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    done = {}
    if out_path and os.path.exists(out_path):
        done = {r['tile_id']: r for r in csv.DictReader(open(out_path, encoding='utf-8'))}
        ids = [t for t in ids if t not in done]
        print(f'resuming: {len(done)} already scored, {len(ids)} left', flush=True)
        if not ids:
            return list(done.values())

    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map='auto').eval()

    fh = w = None
    if out_path:
        fresh = not done
        fh = open(out_path, 'a', newline='', encoding='utf-8')
        w = csv.DictWriter(fh, fieldnames=['tile_id'] + KEYS + ['n_parsed', 'raw'])
        if fresh:
            w.writeheader(); fh.flush()

    out = list(done.values())
    for n, t in enumerate(ids, 1):
        img = Image.open(tile_path(tiles_dir, t)).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                             {'type': 'text', 'text': VLM_INSTRUCTION}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = proc.decode(gen[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        scores = {k: 0.0 for k in KEYS}
        hit = set()
        for line in txt.splitlines():
            if ':' not in line:
                continue
            k, _, v = line.partition(':')
            k = k.strip().lstrip('0123456789.) ').strip('-*|_ ').lower()
            if k in scores:
                try:
                    scores[k] = float(v.strip().split()[0])
                    hit.add(k)
                except (ValueError, IndexError):
                    pass
        # n_parsed makes a silent all-zero parse visible in the CSV instead of
        # looking like a tile where the model genuinely saw nothing.
        row = dict(tile_id=t, **scores, n_parsed=len(hit), raw=txt.replace('\n', ' | ')[:500])
        out.append(row)
        if w:
            w.writerow(row); fh.flush()
        if n % 10 == 0 or n == len(ids):
            print(f'  {n}/{len(ids)}', flush=True)
    if fh:
        fh.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles', required=True, help='Directory made by path_tiles.py.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--backend', choices=['conch', 'medgemma'], default='conch')
    ap.add_argument('--model', default=None)
    ap.add_argument('--shortlist', default=None, help='Earlier CSV to take the top tiles from.')
    ap.add_argument('--top', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--probes', action='store_true',
                    help='conch only: also score the candidate captions in PROBES, as '
                         'extra columns alongside the eighteen real ones.')
    ap.add_argument('--hf_token', default=os.environ.get('HF_TOKEN'),
                    help='Needed for the gated repos; defaults to $HF_TOKEN, and '
                         'huggingface_hub.login() also works.')
    args = ap.parse_args()

    idx = load_index(args.tiles)
    ids = [r['tile_id'] for r in idx]
    if args.shortlist:
        keep = set(shortlist_ids(args.shortlist, args.top))
        ids = [t for t in ids if t in keep]
        print(f'shortlist: {len(ids)} tiles', flush=True)
    if args.limit:
        ids = ids[:args.limit]
    n_cols = len(KEYS) + (len(PROBE_KEYS) if args.probes and args.backend == 'conch' else 0)
    print(f'{args.backend}: scoring {len(ids)} tiles against {n_cols} structures'
          + (f' (incl. {len(PROBE_KEYS)} probes)' if n_cols > len(KEYS) else ''),
          flush=True)

    if args.backend == 'conch':
        rows = run_conch(args.tiles, ids, args.model or 'hf_hub:MahmoodLab/conch',
                         token=args.hf_token, probes=args.probes)
        fields = list(rows[0].keys())
        with open(args.out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    else:
        # writes as it goes, so --out is both the output and the resume point
        rows = run_medgemma(args.tiles, ids, args.model or 'google/medgemma-1.5-4b-it',
                            out_path=args.out)
    print(f'-> {args.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
