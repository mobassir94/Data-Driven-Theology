# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 09:22:11 2022

@author: MOBASSIR
"""

import site
import shutil

from sacremoses import MosesPunctNormalizer, MosesTokenizer
from sacremoses.util import xml_unescape
from subword_nmt.apply_bpe import BPE as subword_nmt_bpe, read_vocabulary
from transliterate import translit

from io import TextIOBase, StringIO
import re

from typing import Dict, Any, Union, List, Optional
from io import TextIOBase, BufferedIOBase
import os

import numpy as np

from collections import namedtuple

import re
import numpy as np

import torch
import torch.nn as nn


loc = site.getsitepackages() 

root = site.getusersitepackages()
data_path = loc[0]+'/laserembeddings/'
if not os.path.exists(data_path):
    os.mkdir(data_path)
    os.mkdir(data_path+'data')
    
data_path = loc[0]+'/laserembeddings/data'

shutil.copy('./data_driven_theology/93langs.fcodes', data_path)
shutil.copy('./data_driven_theology/93langs.fvocab', data_path)
shutil.copy('./data_driven_theology/bilstm.93langs.2018-12-26.pt', data_path)


# from https://github.com/yannvgn/laserembeddings/tree/master/laserembeddings

__all__ = ['adapt_bpe_codes', 'sre_performance_patch']


def adapt_bpe_codes(bpe_codes_f: TextIOBase) -> TextIOBase:
    """
    Converts fastBPE codes to subword_nmt BPE codes.
    Args:
        bpe_codes_f (TextIOBase): the text-mode file-like object of fastBPE codes
    Returns:
        TextIOBase: subword_nmt-compatible BPE codes as a text-mode file-like object
    """
    return StringIO(
        re.sub(r'^([^ ]+) ([^ ]+) ([^ ]+)$',
               r'\1 \2',
               bpe_codes_f.read(),
               flags=re.MULTILINE))


class sre_performance_patch:
    """
    Patch fixing https://bugs.python.org/issue37723 for Python 3.7 (<= 3.7.4)
    and Python 3.8 (<= 3.8.0 beta 3)
    """

    def __init__(self):
        self.sre_parse = None
        self.original_sre_parse_uniq = None

    def __enter__(self):
        #pylint: disable=import-outside-toplevel
        import sys

        if self.original_sre_parse_uniq is None and (
                0x03070000 <= sys.hexversion <= 0x030704f0
                or 0x03080000 <= sys.hexversion <= 0x030800b3):
            try:
                import sre_parse
                self.sre_parse = sre_parse
                #pylint: disable=protected-access
                self.original_sre_parse_uniq = sre_parse._uniq
                sre_parse._uniq = lambda x: list(dict.fromkeys(x))
            except (ImportError, AttributeError):
                self.sre_parse = None
                self.original_sre_parse_uniq = None

    def __exit__(self, type_, value, traceback):
        if self.sre_parse and self.original_sre_parse_uniq:
            #pylint: disable=protected-access
            self.sre_parse._uniq = self.original_sre_parse_uniq
            self.original_sre_parse_uniq = None
# Extras
try:
    import jieba
    jieba.setLogLevel(60)
except ImportError:
    jieba = None

try:
    import MeCab
    import ipadic
except ImportError:
    MeCab = None

__all__ = ['Tokenizer', 'BPE']

###############################################################################
#
# Tokenizer
#
###############################################################################


class Tokenizer:
    """
    Tokenizer.
    Args:
        lang (str): the language code (ISO 639-1) of the texts to tokenize
        lower_case (bool, optional): if True, the texts are lower-cased before being tokenized.
            Defaults to True.
        romanize (bool or None, optional): if True, the texts are romanized.
            Defaults to None (romanization enabled based on input language).
        descape (bool, optional): if True, the XML-escaped symbols get de-escaped.
            Default to False.
    """

    def __init__(self,
                 lang: str = 'en',
                 lower_case: bool = True,
                 romanize: Optional[bool] = None,
                 descape: bool = False):
        assert lower_case, 'lower case is needed by all the models'

        if lang in ('cmn', 'wuu', 'yue'):
            lang = 'zh'
        if lang == 'jpn':
            lang = 'ja'

        if lang == 'zh' and jieba is None:
            raise ModuleNotFoundError(
                '''No module named 'jieba'. Install laserembeddings with 'zh' extra to fix that: "pip install laserembeddings[zh]"'''
            )
        if lang == 'ja' and MeCab is None:
            raise ModuleNotFoundError(
                '''No module named 'MeCab'. Install laserembeddings with 'ja' extra to fix that: "pip install laserembeddings[ja]"'''
            )

        self.lang = lang
        self.lower_case = lower_case
        self.romanize = romanize if romanize is not None else lang == 'el'
        self.descape = descape

        self.normalizer = MosesPunctNormalizer(lang=lang)
        self.tokenizer = MosesTokenizer(lang=lang)
        self.mecab_tokenizer = MeCab.Tagger(
            f"{ipadic.MECAB_ARGS} -Owakati -b 50000") if lang == 'ja' else None

    def tokenize(self, text: str) -> str:
        """Tokenizes a text and returns the tokens as a string"""

        # REM_NON_PRINT_CHAR
        # not implemented

        # NORM_PUNC
        text = self.normalizer.normalize(text)

        # DESCAPE
        if self.descape:
            text = xml_unescape(text)

        # MOSES_TOKENIZER
        # see: https://github.com/facebookresearch/LASER/issues/55#issuecomment-480881573
        text = self.tokenizer.tokenize(text,
                                       return_str=True,
                                       escape=False,
                                       aggressive_dash_splits=False)

        # jieba
        if self.lang == 'zh':
            text = ' '.join(jieba.cut(text.rstrip('\r\n')))

        # MECAB
        if self.lang == 'ja':
            text = self.mecab_tokenizer.parse(text).rstrip('\r\n')

        # ROMAN_LC
        if self.romanize:
            text = translit(text, self.lang, reversed=True)

        if self.lower_case:
            text = text.lower()

        return text


###############################################################################
#
# Apply BPE
#
###############################################################################


class BPE:
    """
    BPE encoder.
    Args:
        bpe_codes (str or TextIOBase): the path to LASER's BPE codes (``93langs.fcodes``),
            or a text-mode file object.
        bpe_codes (str or TextIOBase): the path to LASER's BPE vocabulary (``93langs.fvocab``),
            or a text-mode file object.
    """

    def __init__(self, bpe_codes: Union[str, TextIOBase],
                 bpe_vocab: Union[str, TextIOBase]):

        f_bpe_codes = None
        f_bpe_vocab = None

        try:
            if isinstance(bpe_codes, str):
                f_bpe_codes = open(bpe_codes, 'r', encoding='utf-8')  # pylint: disable=consider-using-with
            if isinstance(bpe_vocab, str):
                f_bpe_vocab = open(bpe_vocab, 'r', encoding='utf-8')  # pylint: disable=consider-using-with

            self.bpe = subword_nmt_bpe(codes=adapt_bpe_codes(f_bpe_codes
                                                             or bpe_codes),
                                       vocab=read_vocabulary(f_bpe_vocab
                                                             or bpe_vocab,
                                                             threshold=None))
            self.bpe.version = (0, 2)

        finally:
            if f_bpe_codes:
                f_bpe_codes.close()
            if f_bpe_vocab:
                f_bpe_vocab.close()

    def encode_tokens(self, sentence_tokens: str) -> str:
        """Returns the BPE-encoded sentence from a tokenized sentence"""
        return self.bpe.process_line(sentence_tokens)


# The code contained in this file was copied/pasted from LASER's source code (source/embed.py)
# and nearly kept untouched besides:
# - code formatting
# - buffered_arange: fix to avoid unnecessary warning on PyTorch >= 1.4.0

# pylint: disable=redefined-builtin, consider-using-enumerate, arguments-differ, fixme, abstract-method, consider-using-from-import


__all__ = ['SentenceEncoder', 'Encoder']

SPACE_NORMALIZER = re.compile(r'\s+')
Batch = namedtuple('Batch', 'srcs tokens lengths')


def buffered_arange(max):
    if not hasattr(buffered_arange,
                   'buf') or max > buffered_arange.buf.numel():
        buffered_arange.buf = torch.LongTensor()
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


# TODO Do proper padding from the beginning
def convert_padding_direction(src_tokens,
                              padding_idx,
                              right_to_left=False,
                              left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


class SentenceEncoder:
    def __init__(self,
                 model_path,
                 max_sentences=None,
                 max_tokens=None,
                 cpu=False,
                 fp16=False,
                 sort_kind='quicksort'):
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        self.encoder = Encoder(**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']
        if fp16:
            self.encoder.half()
        if self.use_cuda:
            self.encoder.cuda()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
        self.encoder.eval()
        embeddings = self.encoder(tokens, lengths)['sentemb']
        return embeddings.detach().cpu().numpy()

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        ids = torch.LongTensor(ntokens + 1)
        for i, token in enumerate(tokens):
            ids[i] = self.dictionary.get(token, self.unk_index)
        ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)  # pylint: disable=invalid-unary-operand-type

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]),
                                      self.pad_index)
            for i in range(len(tokens)):
                toks[i, -tokens[i].shape[0]:] = tokens[i]
            return Batch(srcs=None,
                         tokens=toks,
                         lengths=torch.LongTensor(lengths)), indices

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and ((self.max_tokens is not None
                                    and ntokens + lengths[i] > self.max_tokens)
                                   or (self.max_sentences is not None
                                       and nsentences == self.max_sentences)):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class Encoder(nn.Module):
    def __init__(self,
                 num_embeddings,
                 padding_idx,
                 embed_dim=320,
                 hidden_size=512,
                 num_layers=1,
                 bidirectional=False,
                 left_pad=True,
                 padding_value=0.):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(num_embeddings,
                                         embed_dim,
                                         padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.cpu())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens,
                      final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                        1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ],
                                 dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            'sentemb':
            sentemb,
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask':
            encoder_padding_mask if encoder_padding_mask.any() else None
        }
__all__ = ['BPESentenceEmbedding']


class BPESentenceEmbedding:
    """
    LASER embeddings computation from BPE-encoded sentences.
    Args:
        encoder (str or BufferedIOBase): the path to LASER's encoder PyTorch model,
            or a binary-mode file object.
        max_sentences (int, optional): see ``.encoder.SentenceEncoder``.
        max_tokens (int, optional): see ``.encoder.SentenceEncoder``.
        stable (bool, optional): if True, mergesort sorting algorithm will be used,
            otherwise quicksort will be used. Defaults to False. See ``.encoder.SentenceEncoder``.
        cpu (bool, optional): if True, forces the use of the CPU even a GPU is available. Defaults to False.
    """

    def __init__(self,
                 encoder: Union[str, BufferedIOBase],
                 max_sentences: Optional[int] = None,
                 max_tokens: Optional[int] = 12000,
                 stable: bool = False,
                 cpu: bool = False):

        self.encoder = SentenceEncoder(
            encoder,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind='mergesort' if stable else 'quicksort',
            cpu=cpu)

    def embed_bpe_sentences(self, bpe_sentences: List[str]) -> np.ndarray:
        """
        Computes the LASER embeddings of BPE-encoded sentences
        Args:
            bpe_sentences (List[str]): The list of BPE-encoded sentences
        Returns:
            np.ndarray: A N * 1024 NumPy array containing the embeddings, N being the number of sentences provided.
        """
        return self.encoder.encode_sentences(bpe_sentences)
    
__all__ = ['Laser']


class Laser:
    """
    End-to-end LASER embedding.
    The pipeline is: ``Tokenizer.tokenize`` -> ``BPE.encode_tokens`` -> ``BPESentenceEmbedding.embed_bpe_sentences``
    Args:
        bpe_codes (str or TextIOBase, optional): the path to LASER's BPE codes (``93langs.fcodes``),
            or a text-mode file object. If omitted, ``Laser.DEFAULT_BPE_CODES_FILE`` is used.
        bpe_codes (str or TextIOBase, optional): the path to LASER's BPE vocabulary (``93langs.fvocab``),
            or a text-mode file object. If omitted, ``Laser.DEFAULT_BPE_VOCAB_FILE`` is used.
        encoder (str or BufferedIOBase, optional): the path to LASER's encoder PyToch model (``bilstm.93langs.2018-12-26.pt``),
            or a binary-mode file object. If omitted, ``Laser.DEFAULT_ENCODER_FILE`` is used.
        tokenizer_options (Dict[str, Any], optional): additional arguments to pass to the tokenizer.
            See ``.preprocessing.Tokenizer``.
        embedding_options (Dict[str, Any], optional): additional arguments to pass to the embedding layer.
            See ``.embedding.BPESentenceEmbedding``.
    
    Class attributes:
        DATA_DIR (str): the path to the directory of default LASER files.
        DEFAULT_BPE_CODES_FILE: the path to default BPE codes file.
        DEFAULT_BPE_VOCAB_FILE: the path to default BPE vocabulary file.
        DEFAULT_ENCODER_FILE: the path to default LASER encoder PyTorch model file.
    """

    DATA_DIR = data_path #os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    DEFAULT_BPE_CODES_FILE = os.path.join(DATA_DIR, '93langs.fcodes')
    DEFAULT_BPE_VOCAB_FILE = os.path.join(DATA_DIR, '93langs.fvocab')
    DEFAULT_ENCODER_FILE = os.path.join(DATA_DIR,
                                        'bilstm.93langs.2018-12-26.pt')

    def __init__(self,
                 bpe_codes: Optional[Union[str, TextIOBase]] = None,
                 bpe_vocab: Optional[Union[str, TextIOBase]] = None,
                 encoder: Optional[Union[str, BufferedIOBase]] = None,
                 tokenizer_options: Optional[Dict[str, Any]] = None,
                 embedding_options: Optional[Dict[str, Any]] = None):

        if tokenizer_options is None:
            tokenizer_options = {}
        if embedding_options is None:
            embedding_options = {}

        if bpe_codes is None:
            if not os.path.isfile(self.DEFAULT_BPE_CODES_FILE):
                raise FileNotFoundError(
                    '93langs.fcodes is missing, run "python -m laserembeddings download-models" to fix that'
                )
            bpe_codes = self.DEFAULT_BPE_CODES_FILE
        if bpe_vocab is None:
            if not os.path.isfile(self.DEFAULT_BPE_VOCAB_FILE):
                raise FileNotFoundError(
                    '93langs.fvocab is missing, run "python -m laserembeddings download-models" to fix that'
                )
            bpe_vocab = self.DEFAULT_BPE_VOCAB_FILE
        if encoder is None:
            if not os.path.isfile(self.DEFAULT_ENCODER_FILE):
                raise FileNotFoundError(
                    'bilstm.93langs.2018-12-26.pt is missing, run "python -m laserembeddings download-models" to fix that'
                )
            encoder = self.DEFAULT_ENCODER_FILE

        self.tokenizer_options = tokenizer_options
        self.tokenizers: Dict[str, Tokenizer] = {}

        self.bpe = BPE(bpe_codes, bpe_vocab)
        self.bpeSentenceEmbedding = BPESentenceEmbedding(
            encoder, **embedding_options)

    def _get_tokenizer(self, lang: str) -> Tokenizer:
        """Returns the Tokenizer instance for the specified language. The returned tokenizers are cached."""

        if lang not in self.tokenizers:
            self.tokenizers[lang] = Tokenizer(lang, **self.tokenizer_options)

        return self.tokenizers[lang]

    def embed_sentences(self, sentences: Union[List[str], str],
                        lang: Union[str, List[str]]) -> np.ndarray:
        """
        Computes the LASER embeddings of provided sentences using the tokenizer for the specified language.
        Args:
            sentences (str or List[str]): the sentences to compute the embeddings from.
            lang (str or List[str]): the language code(s) (ISO 639-1) used to tokenize the sentences
                (either as a string - same code for every sentence - or as a list of strings - one code per sentence).
        Returns:
            np.ndarray: A N * 1024 NumPy array containing the embeddings, N being the number of sentences provided.
        """
        sentences = [sentences] if isinstance(sentences, str) else sentences
        lang = [lang] * len(sentences) if isinstance(lang, str) else lang

        if len(sentences) != len(lang):
            raise ValueError(
                'lang: invalid length: the number of language codes does not match the number of sentences'
            )

        with sre_performance_patch():  # see https://bugs.python.org/issue37723
            sentence_tokens = [
                self._get_tokenizer(sentence_lang).tokenize(sentence)
                for sentence, sentence_lang in zip(sentences, lang)
            ]
            bpe_encoded = [
                self.bpe.encode_tokens(tokens) for tokens in sentence_tokens
            ]

            return self.bpeSentenceEmbedding.embed_bpe_sentences(bpe_encoded)
            

