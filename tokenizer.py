import sentencepiece as spm
import requests

class TokenizerWrapper:
    """
    A wrapper script for tokenizing part which includes data downloading, merging, and adding bos and eos tags.
    """

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Special tokens
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

        # Token IDs
        self.bos_id = self.sp.piece_to_id(self.bos_token)
        self.eos_id = self.sp.piece_to_id(self.eos_token)
        self.pad_id = self.sp.piece_to_id(self.pad_token)

    def encode(self, text, add_bos=False, add_eos=False):
        """
        Encodes the given string into token IDs, optionally adding <bos> and <eos>.

        
        """
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids += [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        to do the opposite, decode the given ids
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.decode(ids)

    def get_pad_id(self): return self.pad_id
    def get_eos_id(self): return self.eos_id
    def get_bos_id(self): return self.bos_id


def download_and_merge_text_files(api_url, output_file):
    """
    download all the .txt files to generate the corpus.txt. 
    """
    response = requests.get(api_url)
    files = response.json()

    with open(output_file, "w", encoding="utf-8") as out_f:
        for file in files:
            if file["name"].endswith(".txt"):
                text = requests.get(file["download_url"]).text
                out_f.write(text + "\n")


def train_tokenizer(corpus_path, prefix, vocab_size=10000):
    """
    trianing the corpus.txt on the BPE tokenizer.
    """
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=3,    # pad
        unk_id=0,    # unk
        bos_id=1,    # bos
        eos_id=2,    # eos
        user_defined_symbols=["<bos>", "<eos>", "<pad>"]
    )


def download_file_from_url(file_url: str, output_filename: str):
    """
    similar download function to download train and test jsonl
    """
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {file_url}")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f" Downloaded {output_filename}")
