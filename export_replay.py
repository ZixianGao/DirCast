#!/usr/bin/env python3
"""Export the data of the interactive replay on the project page (no API calls).

For two test clips: DirCast's recorded run (seed 0, final decoding, eval/tune_e30) with per-second on-air camera,
cut gate and speak probability with their rate-controlled thresholds (recomputed and checked against every decision),
generated words, the human director's camera and the casters' words, game events, and frames of the ten hero feeds.
  cd /mnt/workspace/zixian/game_pd && DIRCAST_INDEX=data/index_polished python3 demo/site/export_replay.py
"""
import json
import sys
from pathlib import Path

import av
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
from make_demo_video import load_clip, RUN  # noqa: E402  (same data path and threshold check as the demo video)
from director.data import ClipDataset  # noqa: E402
from director.director_eval import load_rows  # noqa: E402

OUT = Path(__file__).resolve().parent / "static/replay"
CLIPS = {97: "m1", 447: "m2"}


def frames_at(mp4, times, size):
    out, want = {}, sorted(times)
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        c.seek(int(max(0, want[0] - 1) / st.time_base), stream=st)
        j = 0
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base)
            while j < len(want) and t >= want[j]:
                out[want[j]] = fr.to_image().resize(size, Image.LANCZOS); j += 1
            if j == len(want):
                break
    return out


def main():
    rows = load_rows("test")
    preds = {json.loads(l)["idx"]: json.loads(l) for l in open(RUN / "director_free.jsonl")}
    ds = ClipDataset("test", cam_mask_key="strict_mask", kill_feats=True)
    index = []
    for i, key in CLIPS.items():
        C = load_clip(i, rows, ds, preds); P = C["P"]; T = len(P["pred_cam"])
        d = OUT / key; d.mkdir(parents=True, exist_ok=True)
        times = [C["w0"] + s + 0.5 for s in range(T)]
        big_need = {k: set() for k in range(10)}
        for s in range(T):
            big_need[P["pred_cam"][s]].add(s); big_need[P["cam"][s]].add(s)
        for k in range(10):
            fr = frames_at(C["vdir"] / f"hero_{k}.mp4", times, (640, 360))
            for s in range(T):
                im = fr[times[s]]
                im.resize((192, 108), Image.LANCZOS).save(d / f"t{s:02d}_{k}.jpg", quality=80)
                if s in big_need[k]:
                    im.save(d / f"b{s:02d}_{k}.jpg", quality=82)
        sec = []
        for s in range(T):
            sec.append(dict(dc=P["pred_cam"][s], hu=P["cam"][s], live=bool(P["live"][s]),
                            g=round(P["p_cut"][s], 3), tc=round(C["tauc"][s], 3),
                            p=round(P["p_speak"][s], 3), ts=round(C["taus"][s], 3),
                            w=(P["pred_text"].get(str(s)) or "").strip(), hw=(P["text"][s] or "").strip(),
                            ev=C["kills"].get(s, "")))
        meta = dict(key=key, teams=C["teams"].replace("  ", " "), league=C["league"], clock0=C["clock0"],
                    heroes=C["heroes"], seconds=sec)
        (d / "data.json").write_text(json.dumps(meta, ensure_ascii=False))
        index.append(key)
        print(key, T, "seconds")


if __name__ == "__main__":
    main()
