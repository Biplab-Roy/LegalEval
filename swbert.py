import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sklearn
from sklearn.metrics import accuracy_score

train_dataset = pd.read_csv('./ILDC/ILDC_Single/train_dataset.csv')
test_dataset = pd.read_csv('./ILDC/ILDC_Single/test_dataset.csv')
print(f'Train Dataset: {train_dataset.shape}')
print(f'Test Dataset: {test_dataset.shape}')

model_args = ClassificationArgs()
model_args.num_train_epochs = 3
model_args.learning_rate = 1e-5
model_args.sliding_window = True
model_args.stride = 0.8
model_args.overwrite_output_dir = True
model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args = model_args)

model.train_model(train_dataset)
  
result, model_outputs, wrong_predictions = model.eval_model(test_dataset, acc = accuracy_score)

print(result)
