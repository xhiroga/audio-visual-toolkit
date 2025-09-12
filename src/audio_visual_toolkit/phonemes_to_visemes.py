from argparse import ArgumentParser
from typing import List, Tuple

from audio_visual_toolkit.constants import Mpeg4Visemes, PhonemesJaJpOpenJtalk


Triple = Tuple[str, str, str]


def allophonize(phonemes: List[Triple]) -> List[Triple]:
    """日本語の簡易的な異音化のみを適用します。

    - 撥音 "N" は同化規則のみ適用:
      - 次音が両唇音（p/b/m、py/by/my を含む）の前では "m"
      - それ以外・語末では "n"
    """
    if not phonemes:
        return []

    out: List[Triple] = []
    n = len(phonemes)
    for i, (start, end, ph) in enumerate(phonemes):
        if ph == "N":
            # Lookahead to decide assimilation
            next_ph = phonemes[i + 1][2] if i + 1 < n else None
            if next_ph and next_ph[0] in {"p", "b", "m"}:  # includes py/by/my
                ph = "m"
            else:
                ph = "n"

        out.append((start, end, ph))

    return out


VisemeLabel = tuple[int, int, Mpeg4Visemes]


def phonemes_ja_jp_to_mpeg4_visemes(phonemes: list[Triple]) -> List[VisemeLabel]:
    """
    想定している日本語音素はOpenJTalk準拠。
    https://github.com/r9y9/open_jtalk/blob/462fc38e/src/jpcommon/jpcommon_rule_utf_8.h

    音素(日本語音素またはIPA音素)とVisemeの対応関係は次のドキュメントを参考にした。
    - https://docs.aws.amazon.com/ja_jp/polly/latest/dg/ph-table-japanese.html


    ポーズ記号は単純に除外（`_y`→`y`, `a_`→`a`）。また`_dev`は削除する。
    """

    PHONEMES_JA_JP_TO_MPEG4_VISEMES_DICT: dict[
        PhonemesJaJpOpenJtalk, Mpeg4Visemes | None
    ] = {
        "N": None,
        "a": "aa",
        "b": "PP",
        "by": None,
        "ch": "CH",
        "cl": "sil",
        "d": "DD",
        "dy": None,
        "e": "E",
        "f": "FF",
        "g": "kk",
        "gw": None,
        "gy": None,
        "h": None,  # TODO
        "hy": None,
        "i": "I",
        "j": "CH",
        "k": "kk",
        "kw": None,
        "ky": None,
        "m": "PP",
        "my": None,
        "n": "nn",
        "ny": None,
        "o": "O",
        "p": "PP",
        "py": None,
        "r": "nn",  # 日本語音素の "r" は [r]、[l]、[ɾ] のバリエーションがある自由異音とされるが、ここでは [l] として捉えて "nn" にマッピングする。
        "ry": None,
        "s": "SS",
        "sh": "CH",
        "t": "DD",
        "ts": "CH",
        "ty": None,
        "u": "U",
        "v": "FF",
        "w": None,
        "y": None,
        "z": "SS",
    }

    return []


def main():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label-file", type=str, help="Path to label file")
    parser.add_argument("--label-dir", type=str, help="Path to label file")
    parser.add_argument(
        "--language", type=str, help="Only ja-jp is supported", default="ja-jp"
    )

    # TODO: Either a label file or a directory is required.
    args = parser.parse_args()

    main()
