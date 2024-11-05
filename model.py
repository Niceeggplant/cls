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
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.transformers.model_outputs import ModelOutput, TokenClassifierOutput
from paddlenlp.utils.tools import compare_version

from paddlenlp.transformers.ernie_ctm.configuration import (
    ERNIE_CTM_PRETRAINED_INIT_CONFIGURATION,
    ERNIE_CTM_PRETRAINED_RESOURCE_FILES_MAP,
    ErnieCtmConfig,
)

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder

# from .. import PretrainedModel, register_base_model
from paddlenlp.transformers import PretrainedModel, register_base_model

__all__ = [
    "ErnieCtmPretrainedModel",
    "ErnieCtmModel",
    "ErnieCtmWordtagModel",
    "ErnieCtmNptagModel",
    "ErnieCtmForTokenClassification",
]


@dataclass
class ErnieCtmModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`paddle.Tensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        content_output
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    """

    last_hidden_state: paddle.Tensor = None
    pooler_output: paddle.Tensor = None
    content_output: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


class ErnieCtmEmbeddings(Layer):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: ErnieCtmConfig):
        super().__init__() # 调用父类初始方法
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id) # 初始化词嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)  # 初始化位置嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)    # 初始化token类型嵌入层
        self.layer_norm = nn.LayerNorm(config.embedding_size, epsilon=config.layer_norm_eps)    # 初始化层归一化层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)    # 初始化丢弃层,防止过拟合，提高模型泛化能力
        self.cls_num = config.cls_num # 获取配置中分类数量给类
        """
    对输入的token ids进行前向计算，得到其对应的embedding表示。

    Args:
        input_ids (Tensor): 形状为[batch_size, seq_length]的二维张量，表示输入的token ids。
        token_type_ids (Tensor, optional): 形状为[batch_size, seq_length]的二维张量，表示输入的token的类型id。
            默认为None，表示没有类型信息。
        position_ids (Tensor, optional): 形状为[batch_size, seq_length]的二维张量，表示输入的token的位置id。
            默认为None，表示使用默认的位置编码。
        inputs_embeds (Tensor, optional): 形状为[batch_size, seq_length, embed_size]的三维张量，表示输入的token的embedding表示。
            默认为None，表示使用input_ids通过word_embeddings计算得到的embedding表示。

    Returns:
        Tensor: 形状为[batch_size, seq_length, embed_size]的三维张量，表示输入的token经过前向计算后得到的embedding表示。

    """
    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:

            content_len = input_ids.shape[1] - self.cls_num
            position_ids = paddle.concat(
                [
                    paddle.zeros(shape=[self.cls_num], dtype="int64"),
                    paddle.linspace(1, content_len, content_len, dtype="int64"),
                ]
            )
            position_ids.stop_gradient = True

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings # 将输入的token ids通过词嵌入层、位置嵌入层和token类型嵌入层计算得到embedding表示

        embeddings = self.layer_norm(embeddings) # 对embedding表示进行层归一化
        embeddings = self.dropout(embeddings)

        return embeddings

# 初始化池化层(降低维度/特征提取),提出固定大小Layer是飞桨PaddlePaddle框架中用于构建神经网络层的基础类
class ErnieCtmPooler(Layer):
    """ 
        输入序列的隐藏状态进行池化
        方法：
        取每个序列的第一个token的隐藏状态作为整个序列的表示。这个表示通过一个全连接层和双曲正切激活函数进行进一步的处理
    
    """

    def __init__(self, hidden_size):
        super().__init__() # 创建实例就会被调用
        self.dense = nn.Linear(hidden_size, hidden_size) # 输出特征函数的维度
        self.activation = nn.Tanh() # 激活，捕捉复杂关系

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] # 取每个序列的第一个token的隐藏状态作为整个序列的表示
        pooled_output = self.dense(first_token_tensor) # 将取出来的状态通过dense进行线性变化
        pooled_output = self.activation(pooled_output) # 激活函数进行进一步的处理
        return pooled_output # 池化和变化后的序列结果


class ErnieCtmPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ErnieCtm models. It provides ErnieCtm related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading
     and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json" # 模型配置文件的名字
    config_class = ErnieCtmConfig # 模型配置类 
    resource_files_names = {"model_state": "model_state.pdparams"} # 文件名

    base_model_prefix = "ernie_ctm" # 模型名称
 
    pretrained_init_configuration = ERNIE_CTM_PRETRAINED_INIT_CONFIGURATION # 预训练模型初始化配置
    pretrained_resource_files_map = ERNIE_CTM_PRETRAINED_RESOURCE_FILES_MAP # 预训练模型资源文件映射

    # 该函数针对不同类型的层进行了不同的处理。对于线性层和嵌入层，它使用正态分布来初始化权重；对于层归一化层，它设置了一个小的正值来避免计算中的问题。
    def _init_weights(self, layer): # 初始化权重 
        # Initialize weights
        if isinstance(layer, (nn.Linear, nn.Embedding)): # 如果层是线性层或嵌入层； isinstance 判断一个对象是否是某个类的实例
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor): # 如果层的权重是张量
                layer.weight.set_value(      # 如果是，使用set_value方法重置参数
                    paddle.tensor.normal(
                        mean=0.0,  # 均值设为0 
                        std=self.initializer_range    # 标准差设为初始化范围
                        if hasattr(self, "initializer_range")  # 如果有初始化的范围
                        else self.ernie_ctm.config["initializer_range"], 
                        shape=layer.weight.shape, # 权重的形状
                    )
                )
        elif isinstance(layer, nn.LayerNorm): # 如果层是层归一化层
            layer._epsilon = 1e-12  # 设置层归一化层的epsilon值，


@register_base_model
class ErnieCtmModel(ErnieCtmPretrainedModel):
    """
    The bare ErnieCtm Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieCtmModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids`
            passed when calling `ErnieCtmModel`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer.
            Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the encoder layers and the pooler layer.
            Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported
            length of an input sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`.
            Defaults to `16`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        use_content_summary (`bool`, optional):
            Whether or not to add content summary tokens.
            Defaults to `True`.
        content_summary_index (int, optional):
            The number of the content summary tokens. Only valid when use_content_summary is True.
            Defaults to `1`.
        cls_num (int, optional):
            The number of the CLS tokens. Only valid when use_content_summary is True.
            Defaults to `2`.
    """

    def __init__(self, config: ErnieCtmConfig):
        super(ErnieCtmModel, self).__init__(config)

        # 保存参数
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.content_summary_index = config.content_summary_index
        self.initializer_range = config.initializer_range
        self.embeddings = ErnieCtmEmbeddings(config)
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)

        # 定义构建编码器层的函数
        def construct_encoder_layer():
            # 构建编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation="gelu",
                attn_dropout=config.attention_probs_dropout_prob,
                act_dropout=0,
            )
            encoder_layer.activation = nn.GELU(approximate=True)
            return encoder_layer

        self.encoder = nn.TransformerEncoder(construct_encoder_layer(), config.num_hidden_layers) # 构建编码器
        self.pooler = ErnieCtmPooler(config.hidden_size) # 创建池化层，调用ErnieCtmPooler类

        self.use_content_summary = config.use_content_summary # 是否使用内容摘要
        self.content_summary_index = config.content_summary_index # 内容摘要索引
        if config.use_content_summary is True: 
            self.feature_fuse = nn.Linear(config.hidden_size * 2, config.intermediate_size) # 定义特征提取层，输入维度为隐藏层维度*2，输出维度为中间层维度
            self.feature_output = nn.Linear(config.intermediate_size, config.hidden_size) # 定义特征输出层，输入维度为中间层维度，输出维度为隐藏层维度

    def get_input_embeddings(self): # 获取输入嵌入层
        return self.embeddings.word_embeddings  

    def set_input_embeddings(self, value): # 设置输入嵌入层
        self.embeddings.word_embeddings = value 
    # 下面解释有说是关于句子索引，标记的设定
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        content_clone=False,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        r"""
        The ErnieCtmModel forward method, overrides the __call__() special method.

        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                ``[0, max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to
                `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be
                [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                We use whole-word-mask in ERNIE, so the whole word will have the same value.
                For example, "使用" as a word, "使" and "用" will have the same value.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            inputs_embeds (Tensor, optional):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation
                of shape `(batch_size, sequence_length, hidden_size)`. This is useful if you want more control over
                how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
                Default to None.
            content_clone (bool, optional):
                Whether the `content_output` is clone from `sequence_output`. If set to `True`, the content_output is
                clone from sequence_output, which may cause the classification task impact on the sequence labeling
                task.
                Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`. (currently not supported)
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.

        Returns:
            tuple: Returns tuple (``sequence_output``, ``pooled_output``, ``content_output``).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of output at the last layer of the model. Its data type should be float32 and
                has a shape of [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `content_output` (Tensor):
                The output of content summary token (`[CLS1]` in sequence). Its data type should be float32 and
                has a shape of [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieModel, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieModel.from_pretrained('ernie-1.0')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output, content_output = model(**inputs)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # 判断是否返回字典
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions 

        # check the variable of `input_ids` and `inputs_embeds` 检查有效性，输入序列的id和嵌入向量
        if input_ids is None and inputs_embeds is None: 
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if attention_mask is None: # 如果attention_mask为空，那就是去填充，这样权重为0可忽略
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )
        # For 2D attention_mask from tokenizer ，2维度的？
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True # 停止梯度传播
        # 生成嵌入向量
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        embedding_output = self.embedding_hidden_mapping_in(embedding_output) # 将嵌入向量映射到隐藏层维度

        hidden_states = embedding_output   # 线性变化了然后作为隐藏层的初始状态
        # 调用编码器，将以上的数据作为参数传入
        encoder_output = self.encoder(
            hidden_states,
            src_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # when `output_attentions` and `output_hidden_states` are False, it wll return tensor object. 
        encoder_output = (encoder_output,) if paddle.is_tensor(encoder_output) else encoder_output

        sequence_output = encoder_output[0] # 获取编码器的输出

        pooled_output = self.pooler(sequence_output) # 获取池化层的输出
        content_output = sequence_output[:, self.content_summary_index] if self.use_content_summary else None   # 获取内容摘要的输出

        if self.use_content_summary is True: # 处理内容摘要的操作
            if content_clone is True: # 克隆
                sequence_output = paddle.concat( # 匹配形状，进行拼接？
                    (
                        sequence_output,
                        sequence_output[:, self.content_summary_index] 
                        .clone()
                        .unsqueeze([1]) # 改变形状的函数
                        .expand_as(sequence_output),
                    ),
                    2,
                )
            else: # 扩展相同形状，然后拼接
                content_output = paddle.expand( 
                    content_output.unsqueeze([1]),
                    shape=(sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2]),
                )

                sequence_output = paddle.concat((sequence_output, content_output), 2) 

            sequence_output = self.feature_fuse(sequence_output) # 看懂这 特征融合

            sequence_output = self.feature_output(sequence_output)

        if not return_dict: # 返回字典
            return (
                sequence_output, 
                pooled_output,
                content_output,
            ) + encoder_output[1:]

        return ErnieCtmModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            content_output=content_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


class ErnieCtmWordtagModel(ErnieCtmPretrainedModel):
    """
    ErnieCtmWordtag Model with a token classification head on top (a crf layer on top of the hidden-states output) .
    e.g. for Named-Entity-Recognition (NER) tasks.

    Args:
        ernie_ctm (:clss:`ErnieCtmModel`):
            An instance of :class:`ErnieCtmModel`.
        num_tag (int):
            The number of different tags.
        crf_lr (float):
            The learning rate of the crf. Defaults to `100`.
    """

    def __init__(self, config: ErnieCtmConfig):
        super(ErnieCtmWordtagModel, self).__init__(config)
        self.num_tag = config.num_labels
        self.ernie_ctm = ErnieCtmModel(config)
        self.tag_classifier = nn.Linear(config.hidden_size, self.num_tag)
        self.crf = LinearChainCrf(self.num_tag, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        lengths=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        tag_labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieCtmModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieCtmModel`.
            inputs_embeds (Tensor, optional):
                See :class:`ErnieCtmModel`.
            lengths (Tensor, optional):
                The input length. Its dtype is int64 and has a shape of `[batch_size]`.
                Defaults to `None`.
            tag_labels (Tensor, optional):
                The input predicted tensor.
                Its dtype is float32 and has a shape of `[batch_size, sequence_length, num_tags]`.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`. (currently not supported)
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.


        Returns:
            tuple: Returns tuple (`seq_logits`, `cls_logits`).

            With the fields:

            - `seq_logits` (Tensor):
                A tensor of next sentence prediction logits.
                Its data type should be float32 and its shape is [batch_size, sequence_length, num_tag].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer

                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')
                model = ErnieCtmWordtagModel.from_pretrained('ernie-ctm', num_tag=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        # author want to keep the name of `tab_labels`, so add this code to keep style consistent with paddlenlp.
        tag_labels = kwargs.get("labels", tag_labels)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.ernie_ctm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        # [1, 12, 768]
        sequence_output = outputs[0]
        
        # Tensor(shape=[1, 12, 369]
        seq_logits = self.tag_classifier(sequence_output)
        loss = None

        if lengths is None:
            lengths = paddle.sum(input_ids != self.config.pad_token_id, axis=-1)

        if tag_labels is not None:
            crf_loss = self.crf_loss(seq_logits, lengths, tag_labels)
            seq_loss = F.cross_entropy(seq_logits.reshape((-1, self.num_tag)), tag_labels.reshape((-1,)))
            loss = crf_loss + seq_loss
            output = (loss, seq_logits)
        else:
            _, seq_logits = self.viterbi_decoder(seq_logits, lengths)
            output = (seq_logits,)

        if not return_dict:
            return output + outputs[1:]

        return TokenClassifierOutput(
            loss=loss, logits=seq_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class ErnieCtmMLMHead(Layer):
    def __init__(self, config: ErnieCtmConfig):
        super(ErnieCtmMLMHead, self).__init__()
        self.layer_norm = nn.LayerNorm(config.embedding_size)

        self.bias = self.create_parameter(
            [config.vocab_size], is_bias=True, default_initializer=nn.initializer.Constant(value=0.0)
        )
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = nn.GELU(approximate=True)
        # Link bias
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class ErnieCtmNptagModel(ErnieCtmPretrainedModel):
    r"""
    ErnieCtmNptag Model with a `masked language modeling` head on top.

    Args:
        ernie_ctm (:clss:`ErnieCtmModel`):
            An instance of :class:`ErnieCtmModel`.
    """

    def __init__(self, config: ErnieCtmConfig):
        super(ErnieCtmNptagModel, self).__init__(config)

        self.ernie_ctm = ErnieCtmModel(config)
        self.predictions = ErnieCtmMLMHead(config)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieCtmModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieCtmModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            inputs_embeds (Tensor, optional):
                See :class:`ErnieCtmModel`.
            output_hidden_states (bool, optional):
                See :class:`ErnieCtmModel`.
            output_attentions (bool, optional):
                See :class:`ErnieCtmModel`.
            return_dict (bool, optional):
                See :class:`ErnieCtmModel`.

        Returns:
            tuple: Returns tensor `logits`, the scores of masked token prediction.
            Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieCtmNptagModel, ErnieCtmTokenizer

                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')
                model = ErnieCtmNptagModel.from_pretrained('ernie-ctm')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 45, 23000]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.ernie_ctm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  
        logits = self.predictions(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape([-1, self.config.vocab_size]), labels.reshape([-1]))

        if not return_dict:
            outputs = (logits,) + outputs[2:]
            return (loss,) + outputs if loss is not None else outputs

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class ErnieCtmForTokenClassification(ErnieCtmPretrainedModel):
    r"""
    ERNIECtm Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        ernie (`ErnieModel`):
            An instance of `ErnieModel`.
        num_tag (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ERNIE.
            If None, use the same value as `hidden_dropout_prob`
            of `ErnieCtmModel` instance `ernie`. Defaults to `None`.
    """

    def __init__(self, config: ErnieCtmConfig):
        super(ErnieCtmForTokenClassification, self).__init__(config)
        self.num_tag = config.num_labels # 设置分类标签数量
        print(self.num_tag)
        self.ernie_ctm = ErnieCtmModel(config)  # 实例化ErnieCtmModel
        self.dropout = nn.Dropout( # 创建dropout层，如果配置中指定了classifier_dropout则使用它，否则使用hidden_dropout_prob
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建线性层，输入维度为hidden_size，维度为num_tag
        self.classifier = nn.Linear(config.hidden_size, self.num_tag)  

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ):
        """
        Args:
            input_ids (Tensor):
                See :class:`ErnieCtmModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieCtmModel`.
            inputs_embeds (Tensor, optional):
                See :class:`ErnieCtmModel`.
            labels (Tensor, optional): labels for model to compute the loss

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[sequence_length, num_tag]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieCtmForTokenClassification, ErnieCtmTokenizer

                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')
                model = ErnieCtmForTokenClassification.from_pretrained('ernie-ctm')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output = self.ernie_ctm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        pooled_output = output[1] 
        pooled_output = self.dropout(pooled_output) 
        # [batch_size, num_tags]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None: # 如果标签数据存在，计算损失函数
            loss_fct = nn.CrossEntropyLoss()  # 忽略无效标签
            loss = loss_fct(logits, labels) # 计算损失函数
        if not return_dict:
            output = (logits,) + output[2:] # 将模型输出和损失函数组合成元组
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)
        
        return TokenClassifierOutput( # 返回TokenClassifierOutput对象，包含模型输出和损失函数
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )