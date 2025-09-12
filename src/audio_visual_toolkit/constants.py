from typing import Literal, TypeAlias, get_args


# MPEG4 Visemes (14+1 IDs)
# https://visagetechnologies.com/uploads/2012/08/MPEG-4FBAOverview.pdf
# https://developers.meta.com/horizon/documentation/native/audio-ovrlipsync-viseme-reference
# https://beta.developers.meta.com/horizon/reference/unity/v78/class_o_v_r_face_expressions
Mpeg4Visemes = Literal[
    "sil",
    "PP",
    "FF",
    "TH",
    "DD",
    "kk",
    "CH",
    "SS",
    "nn",
    "RR",
    "aa",
    "E",
    "I",
    "O",
    "U",
]
MPEG4_VISEMES = list(get_args(Mpeg4Visemes))

# https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms717289%28v%3Dvs.85%29
# https://learn.microsoft.com/azure/ai-services/speech-service/how-to-speech-synthesis-viseme
# https://chatgpt.com/share/68c37a00-fb28-8010-8a3f-ceef2e318689
SpVisemeNickNames = Literal[
    "sil",
    "AX",
    "AA",
    "AO",
    "E",
    "ER",
    "I",
    "U",
    "OW",
    "AW",
    "OY",
    "AY",
    "H",
    "R",
    "L",
    "SZ",
    "SHCH",
    "TH",
    "FV",
    "DTN",
    "KGN",
    "PBM",
]

# OpenJTalkの日本語音素一覧
# ずんずんPJマルチモーダルデータベースのlabelファイルがOpenJTalk準拠と思われるため、そのようにした
# https://github.com/r9y9/open_jtalk/blob/462fc38e/src/jpcommon/jpcommon_rule_utf_8.h
PhonemesJaJpOpenJtalk: TypeAlias = Literal[
    "N",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gw",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "kw",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
]
PHONEMES_JA_JP_OPEN_JTALK = list(get_args(PhonemesJaJpOpenJtalk))
