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
MsVisemeNicknames = Literal[
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
    "KGNG",
    "PBM",
]
MS_VISEME_NICKNAMES = list(get_args(MsVisemeNicknames))

# TODO: 完成後、日本語音素 → MEPG4 Visemeの変換ドキュメントとも照らし合わせて検証。
# https://docs.aws.amazon.com/ja_jp/polly/latest/dg/ph-table-japanese.html
MS_VISEMES_TO_MPEG4_VISEMES_DICT: dict = {}


# OpenJTalkの日本語音素一覧
# ずんずんPJマルチモーダルデータベースのlabelファイルがOpenJTalk準拠と思われるため、そのようにした
# https://github.com/r9y9/open_jtalk/blob/462fc38e/src/jpcommon/jpcommon_rule_utf_8.h
# 便宜上 `sil` と `pau` を含める（休止記号）。
PhonemesJaJpOpenJtalk: TypeAlias = Literal[
    "sil",
    "pau",
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

# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-ssml-phonetic-sets#ja-jp を参考に、一部修正
# "r" は [r],[l],[ɾ] のバリエーションがある自由異音とされるが、ここでは [l] として捉えて "nn" にマッピングする
# "N" の異音化([m],[n],[ŋ])は考慮していない
PHONEMES_JA_JP_TO_MS_VISEMES_DICT: dict[
    PhonemesJaJpOpenJtalk, tuple[MsVisemeNicknames, ...]
] = {
    "sil": ("sil",),
    "pau": ("sil",),
    "N": ("DTN",),
    "a": ("AA",),
    "b": ("PBM",),
    "by": ("PBM", "I"),
    "ch": ("SHCH",),
    "cl": ("sil",),
    "d": ("DTN",),
    "dy": ("DTN", "I"),
    "e": ("E",),
    "f": ("FV",),
    "g": ("KGNG",),
    "gw": (),
    "gy": ("KGNG", "I"),
    "h": ("H",),
    "hy": ("H", "I"),
    "i": ("I",),
    "j": ("SHCH",),
    "k": ("KGNG",),
    "kw": (),
    "ky": ("KGNG", "I"),
    "m": ("PBM",),
    "my": ("PBM", "I"),
    "n": ("DTN",),
    "ny": ("DTN", "I"),
    "o": ("OW",),
    "p": ("PBM",),
    "py": ("PBM", "I"),
    "r": ("L",),
    "ry": ("L", "I"),
    "s": ("SZ",),
    "sh": ("SHCH",),
    "t": ("DTN",),
    "ts": ("DTN", "SZ"),
    "ty": ("DTN", "I"),
    "u": ("U",),
    "v": ("FV",),
    "w": ("U",),
    "y": ("I",),
    "z": ("SZ",),
}
