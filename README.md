# LegalEval: Court Judgment Prediction

This project aims to build an automated Court Judgement Predictor that hat could assist a judge in predicting the outcome of a petition as `accepted` or `rejected` which in turn may play a vital role in alleviating the stress on the Indian Judicial system that is already suffering from a high backlog of case.

---

## **Challenges**
This problem is different from common ML-classification problems because of several hardles that are intrinsic to legal domain:

* Legal document are generally very long, much longer than that of the state-of-the-art models allows
* Vocabulary specified in the legal domain is geography dependent.

We have tried various approaches to solve this problem.

---

## **Dataset and Experiment**
* Experiments are conducted using the `Indian Legal Documents Corpus (ILDC) Single` dataset. The dataset is taken from of the ACL-IJCNLP 2021 paper "ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation"(ref: https://aclanthology.org/2021.acl-long.313/). As part of their policy, we can not upload the dataset.

* The dataset need to be placed in size `ILDC/ILDC_single`, which contains `Dataset_Generation` notebook. Running this notebook in presence of dataset will create train, dev, test splits of the dataset.

* In root directory ipynb file [base_models_truncate_first](<url>) contains training code for base models, where last 500 tokens are feed as input to the model. Then same way ipynb [base_models_truncate_last](<url>) truncates from the last.

* The next approch sliding window is present in the file [swbert.py](<url>). The sliding window model can be run with
```shell
python swbert.py
```
---

## **Results**

Each model is trained for 3-5 epochs using selective text analysis approach to prevent overfitting and to achieve optimal model performance.

BERT models accepts only 512 tokens at a time as input.Therefore, we have used only `first 500 and last 500 tokens` from each document as input. The validation and test results of the trained models in these cases are listed in Table 1 and Table 2 respectively.

### **Table1**

|Model    | Val Acc| Test Acc| Test Loss|
|:--------|:-------|:--------|:---------|
|RoBERTa-base|0.6464 | 0.5630 |0.7286|
|BERT-base |0.6275 |0.5280 |0.8052|
|LegalBERT-base|0.6561| 0.4997 |0.7542|
|Legal-RoBERTa-base|0.6339 |0.5511 |0.7065|

### **Table2**

|Model    | Val Acc| Test Acc| Test Loss|
|:--------|:-------|:--------|:---------|
|RoBERTa-base|0.6098 |0.5412| 0.8624|
|BERT-base |0.5777 |0.5208| 0.9969|
|LegalBERT-base|0.5682 |0.5616 |0.8761|
|Legal-RoBERTa-base|0.5995| 0.5412| 0.9848|

In a large document all of the tokens carry some meanings, so considering only first or last few tokens cannot give us the whole essence of the document. This issue may account for such poor performance of all these models. We have used a `sliding window protocol of 512 tokens each and with a stride of 64` to take the whole document into account. Furthermore, we have taken an average of all results to ensure that the predictions are not biased towards any window. In this experiment we have got test accuracy of 0.5731 and test loss of 1.2539.