import re

import numpy as np


def sub_punctuation(text: str) -> str:
    """
    do some dirty work for the string
    1. delete the punctuation
    2. sub the number into <NUM>

    :param text:
    :return:
    """
    punc = re.compile(
        r',|/|：|;|:|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|、|‘|’|【|】|·|！|”|“| |…|（|）|」|「|《|》|。|，|\.|。|;|；|\+',
        re.S)
    res = punc.sub(" ", text)
    
    # num = re.compile("\d+")
    # res = num.sub("<NUM>", res)
    
    return res


def compute_KL_division(distribution1, distribution2, eps=1e-6):
    """
    compute KL(P(x) || Q(x))
    :param distribution1: P(x) [batch_size, dimension]
    :param distribution2: Q(x) [batch_size, dimension]
    :return:
    """
    assert distribution1.shape == distribution2.shape, "the size must be equal"
    if len(distribution1.shape) == 1:
        distribution1 = distribution1.reshape(1, -1)
        distribution2 = distribution2.reshape(1, -1)
    distribution1 += eps
    distribution2 += eps
    distribution1 = distribution1 / np.sum(distribution1, keepdims=True, axis=-1)
    distribution2 = distribution2 / np.sum(distribution2, keepdims=True, axis=-1)
    
    KL_Div = np.sum(distribution1 * np.log2(distribution1 / distribution2), axis=-1)
    
    return KL_Div


def compute_JS_Div(P, Q, eps=1e-6):
    """

    :param P:
    :param Q:
    :param eps:
    :return:
    """
    part1 = compute_KL_division(P, (P + Q) / 2, eps=eps)
    part2 = compute_KL_division(Q, (P + Q) / 2, eps=eps)
    JS_Div = 1 / 2 * (part1 + part2)
    return JS_Div
