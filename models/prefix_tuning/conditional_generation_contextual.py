# # Conditional Generation with Prefix Tuning.
# ref: https://github.com/thunlp/OpenPrompt/blob/main/tutorial/2.1_conditional_generation.py

import argparse
import torch
from openprompt.data_utils import InputExample

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='gpt2')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='ai-forever/mGPT')
args = parser.parse_args()
print(args)

dataset = dict()
raw_dataset = dict()
raw_dataset['train'] = []
raw_dataset['validation'] = []
raw_dataset['test'] = []
maxtrnsize = 10000

train_data_path    = "/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtrain_data2.txt"
val_data_path      = "/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xval_data2.txt"
test_data_path     = "/kuacc/users/merdogan18/hpc_run/TrMor2023/data/xtest_data2.txt"

def read_data(data_path, dataset):
    with open(data_path, 'r') as reader:
        ignore_start=["<DOC", "</DOC"]
        replace_token="^"
        special_unk="?"
        sentence_start="<S"
        sentence_end="</S"
        part_seperator="\t"
        tag_seperator="+"
        unk_token="?"
        
        data_all, sentence = [], []
        unknown_tag_flag = False
        for i,line in enumerate(reader):
            if any(line.startswith(i) for i in ignore_start):
                continue
            elif line.startswith(sentence_end):
                if(unknown_tag_flag):
                    unknown_tag_flag = False
                else:
                    data_all.append(sentence)
                continue
            elif line.startswith(sentence_start):
                sentence = []
                continue
            try:
                splits     = line.split()
                source     = splits[0]
                lemma      = splits[1].split('+')[0]
                lemma_char = [j for j in lemma]
                tags       = splits[1].split(tag_seperator)
                tags       = lemma_char + tags[1:]
            except:
                print("I am having some problems parsing the line : ", line)
            if unk_token in tags:
                tags = [special_unk]
                unknown_tag_flag = True
            if not line.startswith(sentence_start) and not line.startswith(sentence_end):
                sentence.append([[j for j in source], tags])

        ix = 0
        for sentence in data_all:
            input = ""
            output = ""
            for word in sentence:
                data = {}
                lemma = "".join(word[0])
                index = 0
                for i in range(len(word[1])):
                    index+=1
                    if(len(word[1][i])>1):
                        break
                root = "".join(word[1][:index-1])
                tags = "+".join(word[1][index-1:])
                input += lemma.lower() + " "
                output += root.lower() + "+" + tags + "~"

            data['text_a'] = input.strip()
            data['tgt_text'] = output.strip()
            data["guid"] = ix
            data["text_b"] = ""
            data["meta"] = {}
            data["label"] = None
            dataset.append(data)
            ix += 1

read_data(train_data_path, raw_dataset['train'])
read_data(val_data_path, raw_dataset['validation'])
read_data(test_data_path, raw_dataset['test'])


for split in ['train', 'validation','test']:
    dataset[split] = []
    for data in raw_dataset[split][:]:
        input_example = InputExample(text_a = data['text_a'], text_b = data['text_b'], tgt_text =data['tgt_text'], label=None, guid=data['guid'])
        dataset[split].append(input_example)
print(dataset['train'][0])

#--------------------------------------------------------------------------------------------------------------

# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
#mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)
print(mytemplate.wrap_one_example(dataset['train'][1]))
print(mytemplate.wrap_one_example(dataset['train'][2]))
print(mytemplate.wrap_one_example(dataset['train'][3]))
print(mytemplate.wrap_one_example(dataset['train'][4]))
print(mytemplate.wrap_one_example(dataset['train'][5]))
print(mytemplate.wrap_one_example(dataset['train'][6]))

'''from openprompt.plms import T5TokenizerWrapper
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=64, decoder_max_length=32, tokenizer=tokenizer,truncate_method="head")
# You can see what a tokenized example looks like by
tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))'''

# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=1024, decoder_max_length=1024,
    batch_size=2,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=1024, decoder_max_length=1024,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=1024, decoder_max_length=1024,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()


from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*10
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()
    correct = 0
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        if output_sentence == inputs['tgt_text']:
            correct+=1
    return generated_sentence, correct, groundtruth_sentence


generation_arguments = {
    "max_length": 1024,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

run = "xlast2"
ff = open(run+"context_logger_run.txt", 'w' ,encoding = "utf-8")

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
for epoch in range(10):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        del inputs
        if global_step % 500 == 0:
            ff.write(f"Epoch {epoch}, global_step {global_step} average loss: {(tot_loss-log_loss)/500} lr: {scheduler.get_last_lr()[0]} \n")
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

    generated_sentence, correct, target = evaluate(prompt_model, validation_dataloader)
    with open(run+"context_out_validation"+str(epoch)+"_run"+".txt",'w',encoding = "utf-8") as f:
        for i,sent in enumerate(generated_sentence):
            f.write(target[i] + " => "+sent+"\n")
        f.write('acc: %.3f' % (correct/len(generated_sentence)))
        print('acc: %.3f' % (correct/len(generated_sentence)))

ff.close()

generated_sentence, correct, target = evaluate(prompt_model, test_dataloader)
with open(run+"context_out_test"+str(epoch)+"_run"+".txt",'w',encoding = "utf-8") as f:
    for i,sent in enumerate(generated_sentence):
        f.write(target[i] + " => "+sent+"\n")
    f.write('acc: %.3f' % (correct/len(generated_sentence)))
    print('acc: %.3f' % (correct/len(generated_sentence)))