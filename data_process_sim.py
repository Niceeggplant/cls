# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, "r", encoding="utf-8") as fin:
        for line in fin:
            vocab[line.strip()] = i
            i += 1
    return vocab

def convert_example(example, tokenizer, max_length, is_test=False):
    sents = example["input"]
    tokenized_input = tokenizer(sents, return_length=True, is_split_into_words="token", max_length=max_length)

    if is_test:
        return tokenized_input["input_ids"], tokenized_input["token_type_ids"], tokenized_input["seq_len"]

    tag = example["tag"]
    tokenized_input["tags"] = [tag]
    return (
        tokenized_input["input_ids"],
        tokenized_input["token_type_ids"],
        tokenized_input["seq_len"],
        tokenized_input["tags"],
    )

def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

def read_custom_data(filename):
    """Reads data"""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            example = transfer_str_to_example(line.strip())
            if not example:
                continue
            yield example

def transfer_str_to_example(sample):
    items = sample.split("\t")
    if len(items) != 2:
        return
    
    sents = [items[0]]
    tag = items[1]

    res = {
        "input": sents,
        "tag": tag
    }

    return res

if __name__ == '__main__':
    import sys
    from paddlenlp.transformers import ErnieCtmTokenizer
    tokenizer = ErnieCtmTokenizer.from_pretrained("wordtag")

    test_data = sys.argv[1]
    for example in read_custom_data(test_data):
        print(convert_example(example, tokenizer, 128))
