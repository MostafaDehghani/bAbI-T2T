"""Transformer-based models for bAbi tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.models.transformer import features_to_nonpadding
from tensor2tensor.utils import beam_search
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import babi_qa
from tensor2tensor.models.research import universal_transformer_util

import tensorflow as tf

FLAGS = tf.flags.FLAGS
PAD = text_encoder.PAD_ID

# ============================================================================
# Transformer-base models
# ============================================================================
@registry.register_model
class BabiTransformer(transformer.Transformer):

  @property
  def has_input(self):
      return True

  def estimator_spec_predict(self, features):
    """Construct EstimatorSpec for PREDICT mode."""
    decode_hparams = self._decode_hparams
    infer_out = self.infer(features, beam_size=decode_hparams.beam_size,
      top_beams=(
        decode_hparams.beam_size if decode_hparams.return_beams else 1),
      alpha=decode_hparams.alpha, decode_length=decode_hparams.extra_length)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
    else:
      outputs = infer_out
      scores = None

    # assert (
    #     common_layers.shape_list(features[babi_qa.FeatureNames.STORY])[0] ==
    #     common_layers.shape_list(features[babi_qa.FeatureNames.QUESTION])[0])

    batch_size = common_layers.shape_list(
      features[babi_qa.FeatureNames.STORY])[0]
    batched_problem_choice = (features["problem_choice"] * tf.ones(
      (batch_size,), dtype=tf.int32))
    predictions = {
      "outputs": outputs,
      "scores": scores,
      babi_qa.FeatureNames.STORY: features.get(babi_qa.FeatureNames.STORY),
      babi_qa.FeatureNames.QUESTION:features.get(babi_qa.FeatureNames.QUESTION),
      "targets": features.get("infer_targets"),
      "problem_choice": batched_problem_choice,
    }
    t2t_model._del_dict_nones(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
      predictions=predictions,
      export_outputs={"output": tf.estimator.export.PredictOutput(export_out)})


  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
      features = {}

    # todo(dehghani): remove dim-expansion and check
    story_old = None
    if len(features[babi_qa.FeatureNames.STORY].shape) < 4:
      story_old = features[babi_qa.FeatureNames.STORY]
      features[babi_qa.FeatureNames.STORY] = tf.expand_dims(
        features[babi_qa.FeatureNames.STORY], 2)

    question_old = None
    if len(features[babi_qa.FeatureNames.QUESTION].shape) < 4:
      question_old = features[babi_qa.FeatureNames.QUESTION]
      features[babi_qa.FeatureNames.QUESTION] = tf.expand_dims(
        features[babi_qa.FeatureNames.QUESTION], 2)

    targets_old = features.get("targets", None)
    target_modality = self._problem_hparams.target_modality

    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not tf.contrib.eager.in_eager_mode():
        recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      samples, logits, losses = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      if target_modality.top_is_pointwise:
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:, common_layers.shape_list(recent_output)[1], :,
                     :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)
      if not tf.contrib.eager.in_eager_mode():
        samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      logits = tf.concat([recent_logits, logits[:, -1:]], 1)
      loss = sum([l for l in losses.values() if l is not None])
      return samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = common_layers.shape_list(initial_output)[0]
    else:
      # assert (
      #     common_layers.shape_list(features[babi_qa.FeatureNames.STORY])[0] ==
      #     common_layers.shape_list(features[babi_qa.FeatureNames.QUESTION])[0])

      batch_size = common_layers.shape_list(
        features[babi_qa.FeatureNames.STORY])[0]

      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = (common_layers.shape_list(
        features[babi_qa.FeatureNames.STORY])[1] +
        common_layers.shape_list(features[babi_qa.FeatureNames.QUESTION])[1] +
                       decode_length)
    # Initial values of result, logits and loss.
    result = initial_output
    # tensor of shape [batch_size, time, 1, 1, vocab_size]
    logits = tf.zeros((batch_size, 0, 1, 1, target_modality.top_dimensionality))
    if not tf.contrib.eager.in_eager_mode():
      logits.set_shape([None, None, None, None, None])
    loss = 0.0

    def while_exit_cond(result, logits,
                        loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      length = common_layers.shape_list(result)[1]

      not_overflow = length < decode_length

      if self._problem_hparams.stop_at_eos:
        def fn_not_eos():
          return tf.not_equal(  # Check if the last predicted element is a EOS
            tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID)

        not_eos = tf.cond(
          # We only check for early stoping if there is at least 1 element (
          # otherwise not_eos will crash)
          tf.not_equal(length, 0), fn_not_eos, lambda: True, )

        return tf.cond(tf.equal(batch_size, 1),
          # If batch_size == 1, we check EOS for early stoping
          lambda: tf.logical_and(not_overflow, not_eos),
          # Else, just wait for max length
          lambda: not_overflow)
      return not_overflow

    result, logits, loss = tf.while_loop(while_exit_cond, infer_step,
      [result, logits, loss],
      shape_invariants=[tf.TensorShape([None, None, None, None]),
        tf.TensorShape([None, None, None, None, None]), tf.TensorShape([]), ],
      back_prop=False, parallel_iterations=1)
    if story_old is not None:  # Restore to not confuse Estimator.
      features[babi_qa.FeatureNames.STORY] = story_old
    if question_old is not None:  # Restore to not confuse Estimator.
      features[babi_qa.FeatureNames.QUESTION] = question_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old
    losses = {"training": loss}
    if "partial_targets" in features:
      partial_target_length = \
      common_layers.shape_list(features["partial_targets"])[1]
      result = tf.slice(result, [0, partial_target_length, 0, 0],
                        [-1, -1, -1, -1])
    return {"outputs": result, "scores": None, "logits": logits,
      "losses": losses, }

  def _beam_decode_slow(self, features, decode_length, beam_size, top_beams,
                        alpha):
    """Slow version of Beam search decoding.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """

    # assert (common_layers.shape_list(features[babi_qa.FeatureNames.STORY])[0]
    #   == common_layers.shape_list(features[babi_qa.FeatureNames.QUESTION])[0])

    batch_size = common_layers.shape_list(
      features[babi_qa.FeatureNames.STORY])[0]

    def symbols_to_logits_fn(ids):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])
      if "partial_targets" in features:
        pt = features["partial_targets"]
        pt_length = common_layers.shape_list(pt)[1]
        pt = tf.tile(pt, [1, beam_size])
        pt = tf.reshape(pt, [batch_size * beam_size, pt_length, 1, 1])
        ids = tf.concat([pt, ids], axis=1)

      features["targets"] = ids
      self._coverage = None
      logits, _ = self(features)  # pylint: disable=not-callable
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      if self._problem_hparams:
        modality = self._problem_hparams.target_modality
        if modality.top_is_pointwise:
          return tf.squeeze(logits, axis=[1, 2, 3])
      # -1 due to the pad above.
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    if self.has_input:
      story_old = features[babi_qa.FeatureNames.STORY]
      question_old = features[babi_qa.FeatureNames.QUESTION]

      features[babi_qa.FeatureNames.STORY] = tf.expand_dims(
          features[babi_qa.FeatureNames.STORY], 1)

      features[babi_qa.FeatureNames.QUESTION] = tf.expand_dims(
          features[babi_qa.FeatureNames.QUESTION], 1)

      if len(features[babi_qa.FeatureNames.STORY].shape) < 5:
        features[babi_qa.FeatureNames.STORY] = tf.expand_dims(
          features[babi_qa.FeatureNames.STORY], 4)

      if len(features[babi_qa.FeatureNames.QUESTION].shape) < 5:
        features[babi_qa.FeatureNames.QUESTION] = tf.expand_dims(
          features[babi_qa.FeatureNames.QUESTION], 4)

      # Expand the inputs in to the beam size.
      features[babi_qa.FeatureNames.STORY] = tf.tile(
          features[babi_qa.FeatureNames.STORY], [1, beam_size, 1, 1, 1])

      features[babi_qa.FeatureNames.QUESTION] = tf.tile(
        features[babi_qa.FeatureNames.QUESTION], [1, beam_size, 1, 1, 1])


      s = common_layers.shape_list(features[babi_qa.FeatureNames.STORY])
      features[babi_qa.FeatureNames.STORY] = tf.reshape(
        features[babi_qa.FeatureNames.STORY],[s[0] * s[1], s[2], s[3], s[4]])

      s = common_layers.shape_list(features[babi_qa.FeatureNames.QUESTION])
      features[babi_qa.FeatureNames.QUESTION] = tf.reshape(
        features[babi_qa.FeatureNames.QUESTION],[s[0] * s[1], s[2], s[3], s[4]])

    target_modality = self._problem_hparams.target_modality
    vocab_size = target_modality.top_dimensionality
    # Setting decode length to input length + decode_length
    decode_length = tf.constant(decode_length)
    if "partial_targets" not in features:
      decode_length += common_layers.shape_list(
        features[babi_qa.FeatureNames.STORY])[1] + common_layers.shape_list(
        features[babi_qa.FeatureNames.QUESTION])[1]

    ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        stop_early=(top_beams == 1))

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator!
    if self.has_input:
      features[babi_qa.FeatureNames.STORY] = story_old
      features[babi_qa.FeatureNames.QUESTION] = question_old

    # Return `top_beams` decodings (also remove initial id from the beam search)
    if top_beams == 1:
      samples = ids[:, 0, 1:]
    else:
      samples = ids[:, :top_beams, 1]

    return {"outputs": samples, "scores": scores}

  def _fast_decode(self, features, decode_length, beam_size=1, top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha,
      stronger
        the preference for slonger translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.target_modality

    story = features[babi_qa.FeatureNames.STORY]
    question = features[babi_qa.FeatureNames.QUESTION]

    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = (common_layers.shape_list(story)[1] +
                       common_layers.shape_list(question)[1] + decode_length)

    story = tf.expand_dims(story, axis=1)
    question = tf.expand_dims(question, axis=1)

    if len(story.shape) < 5:
      story = tf.expand_dims(story, axis=4)

    if len(question.shape) < 5:
      question = tf.expand_dims(question, axis=4)

    s = common_layers.shape_list(story)
    batch_size = s[0]
    story = tf.reshape(story, [s[0] * s[1], s[2], s[3], s[4]])

    s = common_layers.shape_list(question)
    batch_size = s[0]

    question = tf.reshape(question, [s[0] * s[1], s[2], s[3], s[4]])

    # _shard_features called to ensure that the variable names match
    story = self._shard_features({babi_qa.FeatureNames.STORY: story}
                                 )[babi_qa.FeatureNames.STORY]

    question = self._shard_features({babi_qa.FeatureNames.QUESTION: question}
                                    )[ babi_qa.FeatureNames.QUESTION]

    story_modality = self._problem_hparams.input_modality[
                  babi_qa.FeatureNames.STORY]
    question_modality = self._problem_hparams.input_modality[
                  babi_qa.FeatureNames.QUESTION]


    with tf.variable_scope(story_modality.name):
      story = story_modality.bottom_sharded(story, dp)


    with tf.variable_scope(question_modality.name,
                    reuse=(story_modality.name == question_modality.name)):
      question = question_modality.bottom_sharded(question, dp)

    with tf.variable_scope("body"):
      if target_modality.is_class_modality:
        encoder_output = dp(self.encode, story, question,
                              features["target_space_id"], hparams)
      else:
        encoder_output, encoder_decoder_attention_bias = dp(self.encode, story,
          question, features["target_space_id"],hparams,features=features)
        encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

      encoder_output = encoder_output[0]


    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(decode_length + 1,
        hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the
      decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      targets = tf.cond(tf.equal(i, 0), lambda: tf.zeros_like(targets),
        lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
        decode_length)


    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(self.decode, targets, cache.get("encoder_output"),
          cache.get("encoder_decoder_attention_bias"), bias, hparams, cache,
          nonpadding=features_to_nonpadding(features, "targets")
                          )

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      return ret, cache

    def labels_to_logits_fn(unused_ids, unused_i, cache):
      """Go from labels to logits"""
      with tf.variable_scope("body"):
        body_outputs = dp(tf.expand_dims, cache.get("encoder_output"), 2)

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      return ret, cache

    if target_modality.is_class_modality:
      ret = transformer.fast_decode(encoder_output=encoder_output,
        encoder_decoder_attention_bias=None,
        symbols_to_logits_fn=labels_to_logits_fn, hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality, beam_size=beam_size,
        top_beams=top_beams, alpha=alpha, batch_size=batch_size)

    else:
      ret = transformer.fast_decode(encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn, hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality, beam_size=beam_size,
        top_beams=top_beams, alpha=alpha, batch_size=batch_size)

    return ret

  def model_fn(self, features):
    transformed_features = self.bottom(features)

    with tf.variable_scope("body"):
      t2t_model.log_info("Building model body")
      body_out = self.body(transformed_features, features)
    output, losses = self._normalize_body_output(body_out)

    if "training" in losses:
      t2t_model.log_info("Skipping T2TModel top and loss because training loss "
               "returned from body")
      logits = output

    else:
      logits = self.top(output, features)
      losses["training"] = self.loss(logits, features)
    return logits, losses


  def inputs_encoding(self, input, original_input,
                      initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1,
    Equation 1.
    This module is also described in [End-To-End Memory Networks](
    https://arxiv.org/abs/1502.01852)
    as Position Encoding (PE). The mask allows the ordering of words in a
    sentence to affect the encoding.    """
    with tf.variable_scope(scope, 'encode_input', initializer=initializer):

      _, _, max_sentence_length, embedding_size = input.get_shape().as_list()


      pad_mask = tf.to_float(tf.not_equal(original_input,
                             tf.constant(PAD, dtype=tf.int32)))
      input_masked = input * pad_mask
      positional_mask = tf.get_variable(name='positional_mask',
        shape=[max_sentence_length, embedding_size])
      # batch_size * len * emb_size
      encoded_input = tf.reduce_sum(tf.multiply(input_masked, positional_mask)
                                    , axis=2)

      return encoded_input


  def encode(self, stories, questions, target_space, hparams,
             unused_features=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      unused_features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """

    inputs = tf.concat([stories, questions], axis=1)
    # inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, encoder_self_attention_bias, _) = (
      transformer.transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer.transformer_encoder(encoder_input,
      encoder_self_attention_bias, hparams,
      # nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=self.attention_weights)

    return encoder_output

  def body(self, features, original_features):

    stories = features.get(babi_qa.FeatureNames.STORY)
    questions = features.get(babi_qa.FeatureNames.QUESTION)
    target_space = features["target_space_id"]

    with tf.variable_scope('input'):
      # [batch_size, story_len, embed_sz]
      encoded_story = self.inputs_encoding(input=stories,
                                           original_input=original_features.get(
                                             babi_qa.FeatureNames.STORY),
                                           initializer=tf.constant_initializer(
                                             1.0), scope='story_encoding')

      # [batch_size, 1, embed_sz]
      encoded_question = self.inputs_encoding(input=questions,
                                              original_input=original_features.get(
                                                babi_qa.FeatureNames.QUESTION),
                                              initializer=tf.constant_initializer(
                                                1.0), scope='question_encoding')

    hparams = self._hparams
    encoder_output = self.encode(encoded_story, encoded_question, target_space,
                                 hparams)

    encoder_output = tf.expand_dims(encoder_output, 2)

    return encoder_output



# ============================================================================
# R-Transformer
# ============================================================================
@registry.register_model
class BabiUniversalTransformer(BabiTransformer):


  def encode(self, stories, questions, target_space, hparams,
             features=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      unused_features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """

    inputs = tf.concat([stories, questions], axis=1)
    # inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, _ = (
      transformer.transformer_prepare_encoder(inputs, target_space, hparams,
        features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output,
        extra_output) = universal_transformer_util.universal_transformer_encoder(
            encoder_input, self_attention_bias, hparams,
            nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights)

    return encoder_output, _, extra_output


  def body(self, features, original_features):

    stories = features.get(babi_qa.FeatureNames.STORY)
    questions = features.get(babi_qa.FeatureNames.QUESTION)
    target_space = features["target_space_id"]

    with tf.variable_scope('input'):
      # [batch_size, story_len, embed_sz]
      encoded_story = self.inputs_encoding(input=stories,
                                           original_input=original_features.get(
                                             babi_qa.FeatureNames.STORY),
                                           initializer=tf.constant_initializer(
                                             1.0), scope='story_encoding')

      # [batch_size, 1, embed_sz]
      encoded_question = self.inputs_encoding(input=questions,
                                              original_input=original_features.get(
                                                babi_qa.FeatureNames.QUESTION),
                                              initializer=tf.constant_initializer(
                                                1.0), scope='question_encoding')

    hparams = self._hparams
    (encoder_output, _, enc_extra_output) = self.encode(encoded_story,
                                                    encoded_question,
                                                    target_space, hparams)

    encoder_output = tf.expand_dims(encoder_output, 2)

    if (hparams.recurrence_type == 'act'
        and hparams.act_loss_weight != 0):
      (enc_ponder_times, enc_remainders) = enc_extra_output

      enc_ponder_times = tf.identity(enc_ponder_times, name="enc_ponder_times")
      enc_remainders = tf.identity(enc_remainders, name="enc_remainders")


      tf.logging.info("final ponder time tensor is this one Mostafa: :P")
      tf.logging.info(enc_ponder_times)

      tf.logging.info("final remainders tensor is this one Mostafa: :P")
      tf.logging.info(enc_remainders)

      enc_act_loss = (hparams.act_loss_weight * tf.reduce_mean(
      enc_ponder_times + enc_remainders))
      return encoder_output, {"act_loss": enc_act_loss}

    return encoder_output


  def _greedy_infer(self, features, decode_length, use_tpu):
    """Fast version of greedy decoding."""
    return self._slow_greedy_infer(features, decode_length)

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
    """Beam search decoding."""
    return self._beam_decode_slow(features, decode_length, beam_size,
                                  top_beams, alpha)

# ============================================================================
# decoder for bAbi dataset
# ============================================================================
def decode_from_babi_dataset(estimator,
                        problem_names,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        dataset_split=None):

  """Perform decoding from dataset."""
  tf.logging.info("Performing local inference from dataset for %s.",
                  str(problem_names))
  # We assume that worker_id corresponds to shard number.
  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  # If decode_hp.batch_size is specified, use a fixed batch size
  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      "shard": shard,
      "dataset_split": dataset_split,
  }

  for problem_idx, problem_name in enumerate(problem_names):
    # Build the inference input function
    problem = hparams.problem_instances[problem_idx]
    infer_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

    # Get the predictions as an iterable
    predictions = estimator.predict(infer_input_fn)

    # Prepare output file writers if decode_to_file passed
    if decode_to_file:
      if decode_hp.shards > 1:
        decode_filename = decode_to_file + ("%.2d" % decode_hp.shard_id)
      else:
        decode_filename = decode_to_file
      output_filepath = _decode_filename(decode_filename, problem_name,
                                         decode_hp)
      parts = output_filepath.split(".")

      parts[-1] = "decoded"
      decoded_filepath = ".".join(parts)


      dir = os.path.dirname(decoded_filepath)
      if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)

      decoded_file = tf.gfile.Open(decoded_filepath, "w")


    problem_hparams = hparams.problems[problem_idx]

    target_is_label = problem_hparams.target_modality[0].startswith(
      registry.Modalities.CLASS_LABEL)

    story_vocab = problem_hparams.vocabulary[babi_qa.FeatureNames.STORY]
    question_vocab = problem_hparams.vocabulary[babi_qa.FeatureNames.STORY]
    targets_vocab = problem_hparams.vocabulary["targets"]

    for num_predictions, prediction in enumerate(predictions):
      num_predictions += 1
      story = prediction[babi_qa.FeatureNames.STORY]
      question = prediction[babi_qa.FeatureNames.QUESTION]
      targets = prediction["targets"]
      outputs = prediction["outputs"]

      # Log predictions
      decoded_outputs = []
      decoded_scores = []
      if decode_hp.return_beams:
        output_beams = np.split(outputs, decode_hp.beam_size, axis=0)
        scores = None
        if "scores" in prediction:
          scores = np.split(prediction["scores"], decode_hp.beam_size, axis=0)
        for i, beam in enumerate(output_beams):
          tf.logging.info("BEAM %d:" % i)
          score = scores and scores[i]
          decoded = log_decode_results(
              story,
              question,
              beam,
              problem_name,
              num_predictions,
              story_vocab,
              question_vocab,
              targets_vocab,
              model_dir=estimator.model_dir,
              target_is_label= target_is_label,
              identity_output=decode_hp.identity_output,
              targets=targets)
          decoded_outputs.append(decoded)
          if decode_hp.write_beam_scores:
            decoded_scores.append(score)
      else:
        decoded = log_decode_results(
            story,
            question,
            outputs,
            problem_name,
            num_predictions,
            story_vocab,
            question_vocab,
            targets_vocab,
            model_dir=estimator.model_dir,
            target_is_label= target_is_label,
            identity_output=decode_hp.identity_output,
            targets=targets)
        decoded_outputs.append(decoded)

      # Write out predictions if decode_to_file passed
      if decode_to_file:
        for i, (d_story, d_question ,
                d_output, d_target) in enumerate(decoded_outputs):
          beam_score_str = ""
          if decode_hp.write_beam_scores:
            beam_score_str = "\t%.2f" % decoded_scores[i]

          decoded_file.write('STORY: %s \n\n' % _make_story_pretty(_remove_pad(str(d_story))))
          decoded_file.write('QUESTION: %s \n\n' % _remove_pad(str(d_question)))
          decoded_file.write('ANSWER: %s \n\n' % _remove_pad(str(d_target)))
          decoded_file.write('OUTPUT: %s \n' % (_remove_extra_eo_answer(str(d_output)) + beam_score_str + decode_hp.delimiter) )
          decoded_file.write('==================================================================\n')

      if (decode_hp.num_samples >= 0 and
          num_predictions >= decode_hp.num_samples):
        break

    if decode_to_file:
      decoded_file.close()

    tf.logging.info("Completed inference on %d samples." % num_predictions)  # pylint: disable=undefined-loop-variable


def _decode_filename(base_filename, problem_name, decode_hp):
  return ("{base}_{model}.{hp}.{problem}.beam{beam}.alpha{alpha}.decodes"
    .format(base=base_filename, model=FLAGS.model, hp=FLAGS.hparams_set,
    problem=problem_name, beam=str(decode_hp.beam_size),
    alpha=str(decode_hp.alpha)))


def log_decode_results(stories,
                       questions,
                       outputs,
                       problem_name,
                       prediction_idx,
                       stories_vocab,
                       questions_vocab,
                       targets_vocab,
                       targets=None,
                       model_dir=None,
                       target_is_label = False,
                       identity_output=False):
  """Log inference results."""
  decoded_stories = None
  decoded_questions = None
  if stories_vocab:
    if identity_output:
      decoded_stories = " ".join(map(str, stories.flatten()))
    else:
      decoded_stories = stories_vocab.decode(stories.flatten())
    # tf.logging.info("Inference results STORY: %s" % decoded_stories)

  if questions_vocab:
    if identity_output:
      decoded_questions = " ".join(map(str, questions.flatten()))
    else:
      decoded_questions = questions_vocab.decode(questions.flatten())
    # tf.logging.info("Inference results QUESTION: %s" % decoded_questions)

  decoded_targets = None
  decoded_outputs = None
  if identity_output:
    decoded_outputs = " ".join(map(str, outputs.flatten()))
    if targets is not None:
      decoded_targets = " ".join(map(str, targets.flatten()))
  else:
    if target_is_label:
      decoded_outputs = targets_vocab.decode(np.asscalar(outputs.flatten()))
      if targets is not None:
        decoded_targets = targets_vocab.decode(np.asscalar(targets.flatten()))
    else:
      decoded_outputs = targets_vocab.decode(outputs.flatten())
      if targets is not None:
        decoded_targets = targets_vocab.decode(targets.flatten())

  # tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
  # if targets is not None:
    # tf.logging.info("Inference results TARGET: %s" % decoded_targets)
  return decoded_stories, decoded_questions, decoded_outputs, decoded_targets


def _remove_pad(hyp):
  """Strips everything after the first <PAD> token, which is normally 1."""
  hyp = hyp.split()
  try:
    index = hyp.index(babi_qa.PAD_TOKEN)
    return ' '.join(hyp[0:index])
  except ValueError:
    # No PAD: return the string as-is.
    return ' '.join(hyp)

def _remove_extra_eo_answer(hyp):
  """Strips extra <EO_ANSWER> after the first <EO_ANSWER>."""
  hyp = hyp.split()
  try:
    index = hyp.index(babi_qa.EO_ANSWER_TOKEN)
    return ' '.join(hyp[0:index+1])
  except ValueError:
    # No EO_ANSWER: return the string as-is.
    return ' '.join(hyp)

def _make_story_pretty(story):
  pretty_story = story.replace(babi_qa.EOS_TOKEN, babi_qa.EOS_TOKEN + '\n')
  return str(pretty_story)
