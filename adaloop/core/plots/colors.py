"""Fixed colour per method, so every figure uses the same colour for the same method.

Okabe-Ito palette (colour-blind safe). Grey for noise_oracle is outside Okabe-Ito.
"""

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

METHOD_COLORS: dict[str, str] = {
    "controller": OKABE_ITO["vermillion"],
    "heuristic": OKABE_ITO["orange"],
    "best_fixed": OKABE_ITO["blue"],
    "early_exit_only": OKABE_ITO["sky_blue"],
    "tokens_only": OKABE_ITO["bluish_green"],
    "upfront_15way": OKABE_ITO["reddish_purple"],
    "oracle": OKABE_ITO["black"],
    "noise_oracle": "#999999",
    "base": OKABE_ITO["blue"],
    "lora": OKABE_ITO["bluish_green"],
}
