#!/usr/bin/env python3
"""
Generate one embedder YAML per HF/finewiki language config.

Reads the `en.yaml` template, parses the 325 (code, row_count) pairs from
the HF dataset viewer HTML, and writes one YAML per language with only
`source.config` and `storage.s3_prefix` swapped.

Usage:
  python scripts/generate_finewiki_configs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONFIG_DIR = Path("configs/embedder/finewiki_gte_multilingual")
TEMPLATE_NAME = "en.yaml"
MANIFEST_NAME = "_manifest.json"

# Paste directly from the HF dataset viewer optgroup. One entry per language.
# Format: <option value="<code>">(code) (N rows|Nk rows|NM rows)</option>
HTML = """<option selected="" value="en">en (6.61M rows)</option><option value="ab">ab (6.06k rows)</option><option value="ace">ace (13.1k rows)</option><option value="ady">ady (710 rows)</option><option value="af">af (125k rows)</option><option value="als">als (29.1k rows)</option><option value="alt">alt (1.09k rows)</option><option value="am">am (13.8k rows)</option><option value="ami">ami (1.84k rows)</option><option value="an">an (47.4k rows)</option><option value="ang">ang (4.85k rows)</option><option value="anp">anp (3.09k rows)</option><option value="ar">ar (1.23M rows)</option><option value="arc">arc (1.55k rows)</option><option value="ary">ary (8.32k rows)</option><option value="arz">arz (1.64M rows)</option><option value="as">as (19.5k rows)</option><option value="ast">ast (135k rows)</option><option value="atj">atj (2.11k rows)</option><option value="av">av (2.84k rows)</option><option value="avk">avk (27k rows)</option><option value="awa">awa (3.77k rows)</option><option value="ay">ay (5.38k rows)</option><option value="az">az (201k rows)</option><option value="azb">azb (224k rows)</option><option value="ba">ba (62.1k rows)</option><option value="ban">ban (31.6k rows)</option><option value="bar">bar (23.5k rows)</option><option value="bat_smg">bat_smg (17.1k rows)</option><option value="bbc">bbc (1.12k rows)</option><option value="bcl">bcl (20.9k rows)</option><option value="be">be (235k rows)</option><option value="bg">bg (285k rows)</option><option value="bh">bh (8.66k rows)</option><option value="bi">bi (1.61k rows)</option><option value="bjn">bjn (11.5k rows)</option><option value="blk">blk (3.24k rows)</option><option value="bm">bm (1.3k rows)</option><option value="bn">bn (187k rows)</option><option value="bo">bo (14.5k rows)</option><option value="bpy">bpy (24.7k rows)</option><option value="br">br (82.2k rows)</option><option value="bs">bs (80.4k rows)</option><option value="bug">bug (16k rows)</option><option value="bxr">bxr (2.46k rows)</option><option value="ca">ca (962k rows)</option><option value="cbk_zam">cbk_zam (3.33k rows)</option><option value="cdo">cdo (13.1k rows)</option><option value="ce">ce (519k rows)</option><option value="ceb">ceb (5.65M rows)</option><option value="ch">ch (604 rows)</option><option value="chr">chr (604 rows)</option><option value="chy">chy (758 rows)</option><option value="ckb">ckb (75.3k rows)</option><option value="co">co (8.56k rows)</option><option value="cr">cr (11 rows)</option><option value="crh">crh (28.6k rows)</option><option value="cs">cs (535k rows)</option><option value="csb">csb (5k rows)</option><option value="cu">cu (1.2k rows)</option><option value="cv">cv (54.5k rows)</option><option value="cy">cy (296k rows)</option><option value="da">da (292k rows)</option><option value="dag">dag (14.7k rows)</option><option value="de">de (2.71M rows)</option><option value="dga">dga (3.28k rows)</option><option value="din">din (501 rows)</option><option value="diq">diq (41.6k rows)</option><option value="dsb">dsb (3.39k rows)</option><option value="dty">dty (3.87k rows)</option><option value="dv">dv (4.57k rows)</option><option value="dz">dz (1.06k rows)</option><option value="ee">ee (1.45k rows)</option><option value="el">el (243k rows)</option><option value="eml">eml (13.5k rows)</option><option value="eo">eo (363k rows)</option><option value="es">es (1.95M rows)</option><option value="et">et (247k rows)</option><option value="eu">eu (454k rows)</option><option value="ext">ext (4.15k rows)</option><option value="fa">fa (1.02M rows)</option><option value="fat">fat (1.9k rows)</option><option value="ff">ff (20.9k rows)</option><option value="fi">fi (573k rows)</option><option value="fiu_vro">fiu_vro (6.22k rows)</option><option value="fj">fj (1.57k rows)</option><option value="fo">fo (12.3k rows)</option><option value="fon">fon (2.84k rows)</option><option value="fr">fr (2.57M rows)</option><option value="frp">frp (5.76k rows)</option><option value="frr">frr (18.7k rows)</option><option value="fur">fur (4.91k rows)</option><option value="fy">fy (52.7k rows)</option><option value="ga">ga (59.5k rows)</option><option value="gag">gag (2.85k rows)</option><option value="gan">gan (5.14k rows)</option><option value="gcr">gcr (2.4k rows)</option><option value="gd">gd (15.4k rows)</option><option value="gl">gl (214k rows)</option><option value="glk">glk (47.6k rows)</option><option value="gn">gn (5.92k rows)</option><option value="gom">gom (4.21k rows)</option><option value="gor">gor (15.4k rows)</option><option value="got">got (1.06k rows)</option><option value="gpe">gpe (4k rows)</option><option value="gu">gu (30.6k rows)</option><option value="guc">guc (939 rows)</option><option value="gur">gur (1.61k rows)</option><option value="guw">guw (1.85k rows)</option><option value="gv">gv (6.79k rows)</option><option value="ha">ha (70.7k rows)</option><option value="hak">hak (9.78k rows)</option><option value="haw">haw (3.02k rows)</option><option value="he">he (372k rows)</option><option value="hi">hi (168k rows)</option><option value="hif">hif (12k rows)</option><option value="hr">hr (196k rows)</option><option value="hsb">hsb (13.6k rows)</option><option value="ht">ht (70k rows)</option><option value="hu">hu (515k rows)</option><option value="hy">hy (310k rows)</option><option value="hyw">hyw (12.9k rows)</option><option value="ia">ia (29.3k rows)</option><option value="id">id (723k rows)</option><option value="ie">ie (13.3k rows)</option><option value="ig">ig (50.3k rows)</option><option value="ik">ik (896 rows)</option><option value="ilo">ilo (15k rows)</option><option value="inh">inh (2.29k rows)</option><option value="io">io (55.1k rows)</option><option value="is">is (58.7k rows)</option><option value="it">it (1.8M rows)</option><option value="iu">iu (462 rows)</option><option value="ja">ja (1.35M rows)</option><option value="jam">jam (1.8k rows)</option><option value="jbo">jbo (1.41k rows)</option><option value="jv">jv (73k rows)</option><option value="ka">ka (180k rows)</option><option value="kaa">kaa (11.2k rows)</option><option value="kab">kab (6.82k rows)</option><option value="kbd">kbd (1.6k rows)</option><option value="kbp">kbp (1.97k rows)</option><option value="kcg">kcg (1.74k rows)</option><option value="kg">kg (1.73k rows)</option><option value="ki">ki (2.12k rows)</option><option value="kk">kk (217k rows)</option><option value="kl">kl (301 rows)</option><option value="km">km (12.8k rows)</option><option value="kn">kn (34.5k rows)</option><option value="ko">ko (582k rows)</option><option value="koi">koi (3.14k rows)</option><option value="krc">krc (1.99k rows)</option><option value="ks">ks (7.78k rows)</option><option value="ksh">ksh (3k rows)</option><option value="ku">ku (89.4k rows)</option><option value="kv">kv (5.35k rows)</option><option value="kw">kw (7.03k rows)</option><option value="ky">ky (76.4k rows)</option><option value="la">la (136k rows)</option><option value="lad">lad (3.76k rows)</option><option value="lb">lb (61.8k rows)</option><option value="lbe">lbe (1.1k rows)</option><option value="lez">lez (3.95k rows)</option><option value="lfn">lfn (4.88k rows)</option><option value="lg">lg (5.25k rows)</option><option value="li">li (14.5k rows)</option><option value="lij">lij (11.5k rows)</option><option value="lld">lld (181k rows)</option><option value="lmo">lmo (78k rows)</option><option value="ln">ln (5.04k rows)</option><option value="lo">lo (5.11k rows)</option><option value="lt">lt (201k rows)</option><option value="ltg">ltg (1.09k rows)</option><option value="lv">lv (127k rows)</option><option value="mad">mad (2.26k rows)</option><option value="mai">mai (15.1k rows)</option><option value="map_bms">map_bms (14k rows)</option><option value="mdf">mdf (4.06k rows)</option><option value="mg">mg (99.7k rows)</option><option value="mhr">mhr (10.7k rows)</option><option value="mi">mi (7.96k rows)</option><option value="min">min (229k rows)</option><option value="mk">mk (153k rows)</option><option value="ml">ml (86.2k rows)</option><option value="mn">mn (28.1k rows)</option><option value="mni">mni (11k rows)</option><option value="mnw">mnw (3.35k rows)</option><option value="mr">mr (98.4k rows)</option><option value="mrj">mrj (9.92k rows)</option><option value="ms">ms (430k rows)</option><option value="mt">mt (7.55k rows)</option><option value="mwl">mwl (4.54k rows)</option><option value="my">my (111k rows)</option><option value="myv">myv (6.99k rows)</option><option value="mzn">mzn (63.4k rows)</option><option value="nah">nah (4.71k rows)</option><option value="nap">nap (14.8k rows)</option><option value="nds">nds (33.1k rows)</option><option value="nds_nl">nds_nl (6.72k rows)</option><option value="ne">ne (29.7k rows)</option><option value="new">new (72.2k rows)</option><option value="nia">nia (1.72k rows)</option><option value="nl">nl (2.07M rows)</option><option value="nn">nn (171k rows)</option><option value="no">no (621k rows)</option><option value="nov">nov (1.49k rows)</option><option value="nqo">nqo (1.64k rows)</option><option value="nrm">nrm (4.68k rows)</option><option value="nso">nso (8.88k rows)</option><option value="nv">nv (22.5k rows)</option><option value="ny">ny (1.18k rows)</option><option value="oc">oc (87.4k rows)</option><option value="olo">olo (4.92k rows)</option><option value="om">om (2.41k rows)</option><option value="or">or (19.5k rows)</option><option value="os">os (18.7k rows)</option><option value="pa">pa (58.5k rows)</option><option value="pag">pag (2.71k rows)</option><option value="pam">pam (10.1k rows)</option><option value="pap">pap (4.79k rows)</option><option value="pcd">pcd (5.56k rows)</option><option value="pcm">pcm (1.76k rows)</option><option value="pdc">pdc (2.23k rows)</option><option value="pfl">pfl (2.85k rows)</option><option value="pi">pi (2.67k rows)</option><option value="pih">pih (6 rows)</option><option value="pl">pl (1.54M rows)</option><option value="pms">pms (69k rows)</option><option value="pnb">pnb (68.5k rows)</option><option value="pnt">pnt (532 rows)</option><option value="ps">ps (20.7k rows)</option><option value="pt">pt (1.14M rows)</option><option value="pwn">pwn (455 rows)</option><option value="qu">qu (23.7k rows)</option><option value="rm">rm (3.79k rows)</option><option value="rmy">rmy (851 rows)</option><option value="rn">rn (962 rows)</option><option value="ro">ro (493k rows)</option><option value="roa_rup">roa_rup (1.49k rows)</option><option value="roa_tara">roa_tara (9.38k rows)</option><option value="ru">ru (1.82M rows)</option><option value="rue">rue (8.05k rows)</option><option value="rw">rw (10.1k rows)</option><option value="sa">sa (12.3k rows)</option><option value="sah">sah (17.2k rows)</option><option value="sat">sat (14.2k rows)</option><option value="sc">sc (7.84k rows)</option><option value="scn">scn (24.7k rows)</option><option value="sco">sco (32.3k rows)</option><option value="sd">sd (17.7k rows)</option><option value="se">se (6.7k rows)</option><option value="sg">sg (581 rows)</option><option value="sh">sh (430k rows)</option><option value="shi">shi (10.9k rows)</option><option value="shn">shn (11k rows)</option><option value="si">si (26.6k rows)</option><option value="simple">simple (265k rows)</option><option value="sk">sk (220k rows)</option><option value="skr">skr (24.1k rows)</option><option value="sl">sl (176k rows)</option><option value="sm">sm (1.23k rows)</option><option value="smn">smn (5.47k rows)</option><option value="sn">sn (10.9k rows)</option><option value="so">so (9.55k rows)</option><option value="sq">sq (113k rows)</option><option value="sr">sr (664k rows)</option><option value="srn">srn (1.2k rows)</option><option value="ss">ss (1.35k rows)</option><option value="st">st (1.83k rows)</option><option value="stq">stq (4.09k rows)</option><option value="su">su (61.9k rows)</option><option value="sv">sv (2.47M rows)</option><option value="sw">sw (98.9k rows)</option><option value="szl">szl (58.8k rows)</option><option value="szy">szy (4.99k rows)</option><option value="ta">ta (177k rows)</option><option value="tay">tay (2.85k rows)</option><option value="tcy">tcy (3.12k rows)</option><option value="te">te (112k rows)</option><option value="tet">tet (1.48k rows)</option><option value="tg">tg (110k rows)</option><option value="th">th (164k rows)</option><option value="ti">ti (472 rows)</option><option value="tk">tk (7.77k rows)</option><option value="tl">tl (46k rows)</option><option value="tly">tly (9.55k rows)</option><option value="tn">tn (3.06k rows)</option><option value="to">to (1.9k rows)</option><option value="tpi">tpi (1.43k rows)</option><option value="tr">tr (630k rows)</option><option value="trv">trv (1.98k rows)</option><option value="ts">ts (1.05k rows)</option><option value="tt">tt (433k rows)</option><option value="tum">tum (18.8k rows)</option><option value="tw">tw (5.4k rows)</option><option value="ty">ty (1.38k rows)</option><option value="tyv">tyv (3.63k rows)</option><option value="udm">udm (5.07k rows)</option><option value="ug">ug (8.86k rows)</option><option value="uk">uk (1.24M rows)</option><option value="ur">ur (226k rows)</option><option value="uz">uz (294k rows)</option><option value="ve">ve (944 rows)</option><option value="vec">vec (69.5k rows)</option><option value="vep">vep (6.89k rows)</option><option value="vi">vi (1.28M rows)</option><option value="vls">vls (7.92k rows)</option><option value="vo">vo (41.6k rows)</option><option value="wa">wa (11.7k rows)</option><option value="war">war (1.22M rows)</option><option value="wo">wo (1.79k rows)</option><option value="wuu">wuu (45.2k rows)</option><option value="xal">xal (1.31k rows)</option><option value="xh">xh (2.68k rows)</option><option value="xmf">xmf (21.8k rows)</option><option value="yi">yi (15.4k rows)</option><option value="yo">yo (36.1k rows)</option><option value="za">za (3.07k rows)</option><option value="zea">zea (6.98k rows)</option><option value="zgh">zgh (11.7k rows)</option><option value="zh">zh (1.3M rows)</option><option value="zh_classical">zh_classical (13.1k rows)</option><option value="zh_min_nan">zh_min_nan (427k rows)</option><option value="zh_yue">zh_yue (106k rows)</option><option value="zu">zu (12.2k rows)</option>"""

ROW_COUNT_RE = re.compile(r"\(([\d.]+)([kM]?) rows?\)")


def parse_rows(label: str) -> int:
    m = ROW_COUNT_RE.search(label)
    if not m:
        return 0
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "k":
        n *= 1_000
    elif unit == "M":
        n *= 1_000_000
    return int(n)


def parse_langs() -> list[tuple[str, int]]:
    out = []
    for m in re.finditer(r'value="([^"]+)">([^<]+)</option>', HTML):
        code = m.group(1)
        rows = parse_rows(m.group(2))
        out.append((code, rows))
    return out


def main():
    langs = parse_langs()
    print(f"Parsed {len(langs)} languages from HTML")

    template_path = CONFIG_DIR / TEMPLATE_NAME
    if not template_path.exists():
        print(f"Template not found: {template_path}", file=sys.stderr)
        sys.exit(1)
    template = template_path.read_text()

    # extract the storage prefix stem once (everything before the trailing language)
    # so we can rewrite it per language. E.g. "finewiki/embed-multilingual-e5-large/en" -> "finewiki/embed-multilingual-e5-large"
    s3_prefix_stem_re = re.compile(r"(s3_prefix:\s*)(.+?)/[a-zA-Z0-9_]+(\s*)$", re.M)
    path_filter_re = re.compile(r'(path_filter:\s*"data/)[^/]+(/\*")')

    written = 0
    skipped = 0
    for code, rows in langs:
        out_path = CONFIG_DIR / f"{code}.yaml"
        if out_path.exists() and code == "en":
            skipped += 1
            continue  # keep the user's original en.yaml untouched

        body = template
        body = path_filter_re.sub(rf'\g<1>{code}wiki\g<2>', body)
        body = s3_prefix_stem_re.sub(rf'\g<1>\g<2>/{code}\g<3>', body)

        out_path.write_text(body)
        written += 1

    # write a small manifest of (code, rows) so we can sort runs by size later
    import json
    manifest = {
        "dataset": "HuggingFaceFW/finewiki",
        "total_languages": len(langs),
        "total_rows": sum(r for _, r in langs),
        "languages": [{"code": c, "rows": r} for c, r in sorted(langs, key=lambda x: -x[1])],
    }
    (CONFIG_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {written} configs, skipped {skipped} (en untouched)")
    print(f"Manifest: {CONFIG_DIR / MANIFEST_NAME}")
    print(f"Total rows across all langs: {manifest['total_rows']:,}")


if __name__ == "__main__":
    main()