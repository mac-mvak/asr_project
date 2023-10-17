from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    last_char : int


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        ans = ''
        last_ind = 0
        for ind in inds:
            if ind != 0 and ind != last_ind:
                ans += self.ind2char[ind]
            last_ind = ind
        return ans
    
    def extend_and_merge(self, hypos, frame, beam_size):
        new_hypos = defaultdict(float)
        probs, args = torch.sort(frame, descending=True)
        for hypo, prob in hypos:
            for j in range(min(probs.shape[0], beam_size)):
                if args[j] == hypo.last_char or args[j] == 0:
                    new_pref = hypo.text
                else:
                    new_pref = hypo.text + self.ind2char[args[j].item()]
                last_char = args[j].item()
                new_prob = prob * probs[j].item()
                new_hypos[Hypothesis(new_pref, last_char)] += new_prob
        return list(new_hypos.items())

    def truncate(self, hypos, beam_size):
        return sorted(hypos, key=lambda x: x[1], reverse=True)[:beam_size]



    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [(Hypothesis('', 0), 1.)]
        for frame in probs[:probs_length, :]:
            hypos = self.extend_and_merge(hypos, frame, beam_size)
            hypos = self.truncate(hypos, beam_size)
        return hypos



