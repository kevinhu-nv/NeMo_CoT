import sys
import re
import json
from torchmetrics.text import WordErrorRate
from whisper_normalizer.english import EnglishTextNormalizer
texts=[]
pred_texts=[]
norm=EnglishTextNormalizer()
wer = WordErrorRate()
for l in open(sys.argv[1]).readlines():
    js=json.loads(l)
    #text=re.sub(r"[\(\);.,?\-!\"':]" ,'', js['text'].lower())
    text=js['reference']
    #text=norm(text)
    texts.append(text)
    #pred_text=re.sub(r"[\(\);.,?\-!\"':]" ,'', js['pred_text'].lower())
    pred_text=js['prediction']
    #pred_text=norm(pred_text)
    pred_texts.append(pred_text)
    #print(text, pred_text, wer(text,pred_text))
    #print(text, pred_text)
wer = WordErrorRate()
print(wer(pred_texts,texts), len(texts))
