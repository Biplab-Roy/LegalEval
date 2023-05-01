### Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# train_dataset = pd.read_csv('./train_dataset.csv')
# dev_dataset = pd.read_csv('./dev_dataset.csv')
# test_dataset = pd.read_csv('./test_dataset.csv')

#train_dataset = train_dataset.iloc[:200]
#dev_dataset = dev_dataset.iloc[:200]
#test_dataset = test_dataset.iloc[:200]


train_dataset = pd.read_csv('/scratch/22cs60r72/NLP/ILDC/train_dataset.csv')
dev_dataset = pd.read_csv('/scratch/22cs60r72/NLP/ILDC/dev_dataset.csv')
test_dataset = pd.read_csv('/scratch/22cs60r72/NLP/ILDC/test_dataset.csv')

print(f'Train Dataset: {train_dataset.shape}')
print(f'Dev Dataset: {dev_dataset.shape}')
print(f'Test Dataset: {test_dataset.shape}')

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
from torch import nn
from transformers.file_utils import ModelOutput

@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)
   
   
class HierarchicalBert(nn.Module):

    def __init__(self, encoder, max_segments=16, max_segment_length=32):
        super(HierarchicalBert, self).__init__()
        # assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1, encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1, encoder.config.hidden_size))
        # Init segment-wise transformer-based encoder
        # encoder = AutoModel.from_pretrained('roberta-base')
                           
        bert = AutoModel.from_pretrained('bert-base-uncased')
        self.seg_encoder = nn.Transformer(d_model=bert.config.hidden_size,
                                          nhead=bert.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=bert.config.intermediate_size,
                                          activation=bert.config.hidden_act,
                                          dropout=bert.config.hidden_dropout_prob,
                                          layer_norm_eps=bert.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder
        
        ##self.seg_encoder=bert

       
        self.fc = nn.Linear(768,2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
#        if token_type_ids is not None:
#            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
 #       else:

        # Encode segments with BERT --> (256, 128, 768)
        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        # print(encoder_outputs.shape)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)
        # Collect document representation
        outputs, _ = torch.max(seg_encoder_outputs, 1)
        outputs_cls = self.fc(outputs)
        # print("------>", labels.shape, outputs_cls.shape)
       
        output_prob = nn.Softmax(dim=1)(outputs_cls)
        loss = self.loss_fn(output_prob, labels)
        return SimpleOutput(last_hidden_state=outputs_cls, hidden_states=outputs, loss = loss)

       
def process(text):
    sentances = [' '.join(sentance.strip().split(' ')[:16]) for sentance in text.split('.')]
    if(len(sentances) < 32):
        sentances.extend(['']*(32-len(sentances)))
    return sentances[:32]
   

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')                              
    bert = AutoModel.from_pretrained('distilbert-base-uncased')
    model = HierarchicalBert(encoder=bert, max_segments=32, max_segment_length=16)

    ##################################################################################################################
    #                                                   Training                                                     #
    ##################################################################################################################
    from transformers import TrainingArguments, Trainer
    import evaluate
    import datasets
   
    train_data = datasets.Dataset.from_pandas(train_dataset)
    dev_data = datasets.Dataset.from_pandas(dev_dataset)
    test_data = datasets.Dataset.from_pandas(test_dataset)
   
    def tokenization(batched_text):
        fake_inputs = {'input_ids': [], 'attention_mask': []}
        for inp in batched_text['text']:
            # Tokenize segment
            temp_inputs = tokenizer(process(inp), padding='max_length', truncation = True, max_length=16)
    #         temp_inputs = tokenizer(['dog ' * 126] * 64)
            fake_inputs['input_ids'].append(temp_inputs['input_ids'])
            fake_inputs['attention_mask'].append(temp_inputs['attention_mask'])
        fake_inputs['input_ids'] = torch.as_tensor(fake_inputs['input_ids'])
        fake_inputs['attention_mask'] = torch.as_tensor(fake_inputs['attention_mask'])
        return fake_inputs
   
   
    train_data = train_data.map(tokenization, batched = True, batch_size = 8)
    dev_data = dev_data.map(tokenization, batched = True, batch_size = 8)
    test_data = test_data.map(tokenization, batched = True, batch_size = 8)
   
   
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs = 10)
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        (logits,_), labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
   
    trainer = Trainer(  
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        compute_metrics=compute_metrics,
        
    )
   
    trainer.train()
   
    trainer.save_model('model_distill')
    ###################################################################################################################
   