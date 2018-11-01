"""Transformer-based models for bAbi tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import babi_qa
from tensor2tensor.models.research import universal_transformer_util

import tensorflow as tf

PAD = text_encoder.PAD_ID

# ============================================================================
# bAbI Transformer
# ============================================================================
@registry.register_model
class BabiTransformer(transformer.Transformer):

	@property
	def has_input(self):
		return True

	def estimator_spec_predict(self, features, use_tpu=False):
		"""Construct EstimatorSpec for PREDICT mode."""

		def _remove_summaries():
			g = tf.get_default_graph()
			key = tf.GraphKeys.SUMMARIES
			del g.get_collection_ref(key)[:]
			assert not g.get_collection(key)

		def _del_dict_non_tensors(d):
			for k in list(d.keys()):
				if not isinstance(d[k], tf.Tensor):
					del d[k]

		decode_hparams = self._decode_hparams
		infer_out = self.infer(
			features,
			beam_size=decode_hparams.beam_size,
			top_beams=(decode_hparams.beam_size
			           if decode_hparams.return_beams else 1),
			alpha=decode_hparams.alpha,
			decode_length=decode_hparams.extra_length,
			use_tpu=use_tpu)

		if isinstance(infer_out, dict):
			outputs = infer_out["outputs"]
			scores = infer_out["scores"]
		else:
			outputs = infer_out
			scores = None

		predictions = {
			"outputs": outputs,
			"scores": scores,
			babi_qa.FeatureNames.STORY: features.get(babi_qa.FeatureNames.STORY),
			babi_qa.FeatureNames.QUESTION: features.get(babi_qa.FeatureNames.QUESTION),
			"targets": features.get("infer_targets"),
		}


		_del_dict_non_tensors(predictions)
		export_out = {"outputs": predictions["outputs"]}
		if "scores" in predictions:
			export_out["scores"] = predictions["scores"]
		_remove_summaries()

		export_outputs = {
			tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				tf.estimator.export.PredictOutput(export_out)
		}

		if use_tpu:
			return tf.contrib.tpu.TPUEstimatorSpec(
				tf.estimator.ModeKeys.PREDICT,
				predictions=predictions,
				export_outputs=export_outputs)
		else:
			return tf.estimator.EstimatorSpec(
				tf.estimator.ModeKeys.PREDICT,
				predictions=predictions,
				export_outputs=export_outputs)

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
		target_modality = self._problem_hparams.modality.get("targets")

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
			batch_size = common_layers.shape_list(
				features[babi_qa.FeatureNames.STORY])[0]

			initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
		# Hack: foldl complains when the output shape is less specified than the
		# input shape, so we confuse it about the input shape.
		initial_output = tf.slice(initial_output, [0, 0, 0, 0],
		                          common_layers.shape_list(initial_output))
		target_modality = self._problem_hparams.modality.get("targets")
		if target_modality.is_class_modality:
			decode_length = 1
		else:
			decode_length = (common_layers.shape_list(
				features[babi_qa.FeatureNames.STORY])[1] +
			                 common_layers.shape_list(
				                 features[babi_qa.FeatureNames.QUESTION])[1] +
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
		                                     shape_invariants=[tf.TensorShape(
			                                     [None, None, None, None]),
		                                                       tf.TensorShape(
			                                                       [None, None, None,
			                                                        None, None]),
		                                                       tf.TensorShape(
			                                                       []), ],
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
				modality = self._problem_hparams.modality.get("targets")
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
				features[babi_qa.FeatureNames.STORY], [s[0] * s[1], s[2], s[3], s[4]])

			s = common_layers.shape_list(features[babi_qa.FeatureNames.QUESTION])
			features[babi_qa.FeatureNames.QUESTION] = tf.reshape(
				features[babi_qa.FeatureNames.QUESTION],
				[s[0] * s[1], s[2], s[3], s[4]])

		target_modality = self._problem_hparams.modality.get("targets")
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

	def _greedy_infer(self, features, decode_length, use_tpu=False):
		"""Greedy decoding."""
		return self._slow_greedy_infer(features, decode_length)

	def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
		"""Beam search decoding."""
		return self._beam_decode_slow(features, decode_length, beam_size,
		                              top_beams, alpha)

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
			                                  shape=[max_sentence_length,
			                                         embedding_size])
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
		                                                 encoder_self_attention_bias,
		                                                 hparams,
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
# bAbI Universal Transformer
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

