# MPEG4 Visemes (14+1 IDs)
# https://visagetechnologies.com/uploads/2012/08/MPEG-4FBAOverview.pdf
# https://developers.meta.com/horizon/documentation/native/audio-ovrlipsync-viseme-reference
MPEG4_VISEMES = [
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

# Azure Visemes (22+1 IDs)は未対応。
# https://learn.microsoft.com/azure/ai-services/speech-service/how-to-speech-synthesis-viseme

# OpenJTalkの日本語音素一覧
# ずんずんPJマルチモーダルデータベースのlabelファイルがOpenJTalk準拠と思われるため、そのようにした
# https://github.com/r9y9/open_jtalk/blob/462fc38e/src/jpcommon/jpcommon_rule_utf_8.h
PHONEMES_JA_JP_OPEN_JTALK = [
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
