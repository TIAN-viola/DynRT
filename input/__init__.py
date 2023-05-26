from input.loader_text import loader_text
from input.loader_label import loader_label
from input.requires import get_tokenizer_roberta,get_tokenizer_bert
from input.loader_img import loader_img
_loaders=[
    loader_text(),
    loader_label(),
    loader_img()
]
_requires={
    "tokenizer_roberta":get_tokenizer_roberta,
    "tokenizer_bert":get_tokenizer_bert,
}

_loadermap={}

for loader in _loaders:
    _loadermap[loader.name]=loader

