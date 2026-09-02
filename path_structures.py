#!/usr/bin/env python3
"""The structure list scored on every tile, shared by the CONCH and VLM backends.

Fixed on purpose. Free-form description of hundreds of fields cannot be compared, ranked
or audited; a fixed list turns the run into a table, and the table is what selects the
regions and can go into supplementary material.

Chosen for ovarian carcinoma, and weighted toward what TPAF has a physical reason to
show differently: autofluorescence at 740-860 nm comes from NAD(P)H, FAD, collagen and
elastin, so ECM architecture is where the two modalities should genuinely diverge --
H&E stains collagen pink but conveys little about fibre organisation. Those entries
are marked ecm=True so they can be pulled out separately.

CONCH-style models are prompted with a caption, so each entry carries one; a chat VLM
gets the same list as a checklist.
"""

STRUCTURES = [
    # architecture
    dict(key='papillary', ecm=False, zh='乳头状结构',
         prompt='an H&E image of papillary architecture with fibrovascular cores'),
    dict(key='solid', ecm=False, zh='实性生长',
         prompt='an H&E image of solid sheets of carcinoma cells'),
    dict(key='glandular', ecm=False, zh='腺样/筛状结构',
         prompt='an H&E image of glandular and cribriform architecture'),
    # Was "...slit-like spaces in high grade serous carcinoma". Naming the disease put
    # a term in the caption that is true of every tile on an HGSOC slide, so the score
    # sat high everywhere (raw mean 0.412, the highest of all 18) and stood out nowhere
    # (top z 1.88, the lowest). A prompt should describe morphology only -- see the
    # note under KEYS.
    dict(key='slit_like', ecm=False, zh='裂隙样腔隙',
         prompt='an H&E image with irregular slit-like spaces between tumour cell nests'),
    # cytology
    dict(key='high_atypia', ecm=False, zh='显著核异型',
         prompt='an H&E image of marked nuclear atypia and pleomorphism'),
    dict(key='mitoses', ecm=False, zh='核分裂象',
         prompt='an H&E image with frequent mitotic figures'),
    # Was "...clear cell carcinoma with clear cytoplasm" -- naming the subtype, mean 0.314.
    dict(key='clear_cell', ecm=False, zh='透明细胞',
         prompt='an H&E image of tumour cells with abundant clear cytoplasm'),
    dict(key='mucinous', ecm=False, zh='黏液性上皮',
         prompt='an H&E image of mucinous epithelium with intracytoplasmic mucin'),
    # stroma / ECM -- where TPAF is expected to add information
    dict(key='desmoplasia', ecm=True, zh='促纤维增生间质',
         prompt='an H&E image of desmoplastic stroma'),
    dict(key='dense_collagen', ecm=True, zh='致密胶原纤维',
         prompt='an H&E image of dense collagenous fibrous stroma'),
    # Was "fibrous septa separating tumour nests", and that column was a tumour-nest
    # detector, not a septa detector. Measured on all six survey slides (4665 tiles),
    # its mean correlation with the other seventeen columns was +0.50 glandular,
    # +0.42 slit_like, +0.41 high_atypia, +0.41 solid, +0.35 necrosis -- against
    # -0.36 dense_collagen, -0.52 ovarian_stroma and +0.04 desmoplasia. Same sign on
    # every sample: the caption correlated with everything epithelial and against
    # everything fibrous, i.e. the intended meaning inverted. Its top tiles are solid
    # sheets of carcinoma with no fibrous tissue in them, and its top z was the lowest
    # of the four ECM entries on all six samples (1.92-3.03).
    #
    # Scoring "an H&E image of tumour nests" as its own column put a number on why:
    # the two agree at r=+0.66, which is the old caption's single closest neighbour
    # out of all twenty-one other columns and sits at the 97.6th percentile of the 918
    # pairwise correlations among the eighteen real columns (median +0.19). The word
    # "septa" contributed nothing that survived pooling.
    #
    # This is a different failure from the slit_like one below. There a disease name
    # true of every tile added a constant offset. Here a relational caption "A
    # separating B" was captured by B: the text encoder pools the whole caption, and
    # "tumour nests" carries the stronger visual prior, so it decides the direction.
    # Hence: never name the thing a structure sits between.
    #
    # Three replacements were scored side by side against that control on identical
    # image embeddings. What they show is that avoiding the epithelial noun is not by
    # itself enough -- the caption has to name the tissue too:
    #
    #   "thin fibrous bands dividing the tissue into lobules"
    #       epi +0.35, dense_collagen +0.41, top z 3.26. Nearest neighbours necrosis
    #       +0.56, solid +0.54, tumour nests +0.53, adipose +0.45. Dropping "tumour
    #       nests" moved it sideways rather than across: with only geometry to go on
    #       ("dividing", "lobules") it drifted to anything partitioned, and its top
    #       tiles mix real collagen with haemorrhage and necrotic debris.
    #   "long thin strands of fibrous tissue traversing the section"  (probe)
    #       dense_collagen +0.74 and 4.0/10 shared top tiles with it -- close enough
    #       to be a second name for a column that already exists.
    #   "elongated collagenous bands running between cellular areas"  <- promoted
    #       epi -0.16, top z 3.81, and not one epithelial term in its four nearest
    #       neighbours (dense_collagen +0.61, ovarian_stroma +0.34, clear_cell +0.28,
    #       vessels +0.19). Distinct from dense_collagen despite that +0.61: only
    #       2.5/10 of their top tiles coincide, so the two rank different regions.
    #
    # So the working rule is narrower than "describe shape". Name the material
    # (collagenous) to land in the right region of the space, and let the geometry
    # (elongated, bands, between) do the separating within it. Geometry alone has
    # nothing to anchor to.
    #
    # Still needs per-sample vetting: on 240828_pt1, four of the top eight are sheets
    # of extravasated red cells rather than collagen -- the same haemorrhage that
    # already misleads dense_collagen on that one slide. A better caption did not fix
    # a sample-specific artefact, and was never going to.
    dict(key='fibrous_septa', ecm=True, zh='纤维间隔',
         prompt='an H&E image of elongated collagenous bands running between '
                'cellular areas'),
    dict(key='ovarian_stroma', ecm=True, zh='卵巢间质',
         prompt='an H&E image of ovarian stroma with spindle cells'),
    # other findings
    dict(key='psammoma', ecm=False, zh='砂粒体',
         prompt='an H&E image containing psammoma bodies'),
    dict(key='necrosis', ecm=False, zh='坏死',
         prompt='an H&E image of tumour necrosis'),
    dict(key='tils', ecm=False, zh='淋巴细胞浸润',
         prompt='an H&E image with dense lymphocytic infiltrate'),
    dict(key='vessels', ecm=True, zh='血管',
         prompt='an H&E image containing blood vessels'),
    dict(key='adipose', ecm=False, zh='脂肪组织',
         prompt='an H&E image of adipose tissue'),
    # Was "...benign ovarian surface epithelium" -- naming the organ, mean 0.315.
    dict(key='normal_epithelium', ecm=False, zh='良性上皮',
         prompt='an H&E image of a single layer of bland cuboidal surface epithelium'),
]

KEYS = [s['key'] for s in STRUCTURES]
ECM_KEYS = [s['key'] for s in STRUCTURES if s['ecm']]

# Prompt design: describe morphology, never the diagnosis or the organ.
#
# Measured on 2502 tiles of one HGSOC slide, the three highest mean similarities were
# slit_like (+0.412), normal_epithelium (+0.315) and clear_cell (+0.314) -- and those
# were exactly the three captions naming a disease or an organ ("in high grade serous
# carcinoma", "benign ovarian surface epithelium", "clear cell carcinoma"). Such a term
# is true of every tile on the slide, so it contributes a constant offset and no
# discrimination; slit_like ended up with the highest mean and the lowest top z of all
# eighteen. The purely morphological captions (dense_collagen "dense collagenous
# fibrous stroma", desmoplasia "desmoplastic stroma") had low means, the widest spread,
# and the top tiles that held up on inspection.
#
# Second rule, from fibrous_septa: never name the thing a structure sits between.
# A relational caption "A separating B" is pooled into one text embedding, and the
# noun with the stronger visual prior decides where the column points -- "fibrous
# septa separating tumour nests" came out correlating +0.50 with glandular and
# -0.36 with dense_collagen, i.e. pointing at B. This is not the constant-offset
# failure above; the meaning inverts rather than flattens, so a high top z would
# not have caught it. What catches it is the correlation of the column with the
# other seventeen: a structure whose column tracks its own opposites is broken
# however clean its ranking looks.
#
# Third rule, from what the fibrous_septa replacements did: name the material, and
# let geometry separate within it. Dropping "tumour nests" and describing only shape
# ("thin fibrous bands dividing the tissue into lobules") did not land on stroma -- it
# landed on necrosis +0.56, solid +0.54 and adipose +0.45, because shape words have
# nothing to anchor to and drift to anything partitioned. The caption that worked names
# the tissue first and uses geometry only to distinguish it from its neighbours
# ("elongated collagenous bands running between cellular areas"). Rules one and two say
# what to leave out; this one says what cannot be left out.
#
# And read a column by its nearest neighbours, not by one number against a fixed cutoff.
# Every caption correlates positively with every other -- the median of the 918 pairs
# among these eighteen is +0.19 -- so a bare threshold means nothing, while the rank of
# a column's neighbours is interpretable on its own. Where two candidates cannot be
# separated that way (dense_collagen +0.61 vs +0.74), compare the top tiles they
# actually rank: 2.5/10 shared is a distinct structure, 4.0/10 is a second name for one.
#
# Changing one prompt only changes that structure's column: the image embeddings are
# unchanged, so a rerun leaves the other seventeen bit-identical and stays comparable.


# Candidate captions under test, scored as extra columns and kept out of KEYS so the
# eighteen real columns and everything downstream of them are untouched.
#
# A prompt change only moves that one column: the image embeddings do not depend on the
# text, so several candidates scored in the same run are directly comparable, and the
# marginal cost is one text embedding each -- the image pass, which is the whole run
# time, is shared. Scoring the retired caption alongside the new one is therefore free,
# and it is the only way to tell an improvement from a difference.
#
# The two kept here are permanent, not leftovers from the September bake-off:
#
#   probe_septa_old   the caption fibrous_septa carried until 2026-08-20. Because the
#                     text is fixed, its column must come back bit-identical to the
#                     archived run -- which makes it a regression check on the whole
#                     pipeline, not just on this entry. It caught nothing this time
#                     (max|diff| 0.00000 on all six samples) and that is the point:
#                     without it a candidate could look better only because the tiles,
#                     preprocessing or weights had moved underneath.
#   probe_nests_only  the mechanism. "an H&E image of tumour nests" is what the old
#                     caption turned out to mean; keeping it scored means any future
#                     septa wording can be checked for the same capture directly
#                     rather than argued about.
#
# How to read a bake-off, in order. Steps 1-3 are calibrated against the run itself,
# because an absolute threshold on a CONCH correlation is meaningless -- every caption
# correlates positively with every other one, and the median of the 918 pairs among the
# eighteen real columns is +0.19, not 0.
#
#   1. Diagnosis. Rank the retired caption's correlations across all other columns. If
#      probe_nests_only comes first, and high in the pairwise distribution, the capture
#      story holds. Rank is the honest test; a cutoff picked in advance is not.
#   2. The defect. A candidate has to drop the epithelial correlations (glandular,
#      solid, high_atypia, slit_like, necrosis) that the retired caption carried.
#   3. Redundancy. Not by correlation with dense_collagen -- +0.61 was fine and +0.74
#      was not, and no threshold separates those. Compare top-tile sets instead: two
#      captions that rank the same regions are one column under two names whatever
#      their correlation says. 2.5/10 shared is a distinct structure; 4.0/10 is not.
#   4. Top z above the retired caption's.
#   5. Then look at the tiles. Steps 1-4 only say the statistic is clean; they cannot
#      say the tiles contain the structure. Nothing is settled without that, and on the
#      2026-08-20 run they were also the only step that showed the winner still picks
#      up haemorrhage on 240828_pt1.
# The morphology that makes a tile a picture of high grade serous carcinoma, plus the
# ECM axis the thesis turns on. Used to qualify additional figure candidates: a region
# is worth adding only if it is a strong example of something on this list.
#
# Left out deliberately. mucinous, clear_cell and normal_epithelium describe other
# tumour types or benign tissue -- a tile whose only high score is one of those is
# either a different diagnosis or not tumour, and neither belongs in an HGSOC figure.
# adipose is orientation, not pathology.
HGSOC_KEYS = ['papillary', 'solid', 'slit_like', 'high_atypia', 'mitoses', 'psammoma',
              'necrosis', 'desmoplasia', 'dense_collagen', 'fibrous_septa',
              'ovarian_stroma', 'tils', 'vessels']


PROBES = [
    dict(key='probe_septa_old', ecm=True, zh='[探针]旧 septa caption',
         prompt='an H&E image of fibrous septa separating tumour nests'),
    dict(key='probe_nests_only', ecm=False, zh='[探针]仅瘤巢',
         prompt='an H&E image of tumour nests'),
]

PROBE_KEYS = [s['key'] for s in PROBES]
ALL_ENTRIES = STRUCTURES + PROBES


def _plain(prompt):
    return prompt.replace('an H&E image of ', '').replace('an H&E image ', '')


# The first version told the model to output a line "key: score". MedGemma 1.5 read that
# as the literal template and returned 40+ lines of "key: 2" -- it was reading the images
# correctly (different tiles drew clearly different grades) but never named what it was
# grading, so every score parsed back to zero and the run still looked successful.
#
# The 18 lines are therefore written out in full now, with only the digit left to fill
# in. That removes the ambiguity, and because the list is finite and explicit it also
# stops the model running on past the end of it.
VLM_INSTRUCTION = '\n'.join([
    'You are grading one H&E histopathology tile from an ovarian specimen.',
    '',
    'Score each feature 0-4: 0 absent, 1 minimal, 2 moderate, 3 marked, '
    '4 the dominant feature of the tile.',
    'Judge only what is visible in this tile. Do not diagnose the patient.',
    '',
    'What each name means:',
] + [f'  {s["key"]} = {_plain(s["prompt"])}' for s in STRUCTURES] + [
    '',
    f'Reply with EXACTLY the following {len(STRUCTURES)} lines, replacing each ? with '
    'one digit 0-4.',
    'Output nothing else: no prose, no extra lines, no repetition.',
] + [f'{s["key"]}: ?' for s in STRUCTURES])


if __name__ == '__main__':
    print(f'{len(STRUCTURES)} structures, {len(ECM_KEYS)} of them ECM-related\n')
    for s in STRUCTURES:
        print(f'  {s["key"]:<20}{"[ECM]" if s["ecm"] else "     "}  {s["zh"]}')
    print(f'\n--- VLM_INSTRUCTION ({len(VLM_INSTRUCTION)} chars) ---')
    print(VLM_INSTRUCTION)
