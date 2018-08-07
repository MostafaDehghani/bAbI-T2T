#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import decoding
from tensor2tensor.utils import usr_dir
from tensor2tensor.data_generators import babi_qa
from tensor2tensor.bin import t2t_decoder


import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
PAD = text_encoder.PAD
EOS = text_encoder.EOS

# ============================================================================
# decoder for bAbi Transformer
# ============================================================================

def decode_from_babi_dataset(estimator,
                        problem_name,
                        hparams,
                        decode_hp,
                        decode_to_file=None):

  """Perform decoding from dataset."""
  tf.logging.info("Performing local inference from dataset for %s.",
                  str(problem_name))


  # We assume that worker_id corresponds to shard number.
  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  # Setup decode output directory for any artifacts that may be written out
  output_dir = os.path.join(estimator.model_dir, "decode")
  tf.gfile.MakeDirs(output_dir)

  # If decode_hp.batch_size is specified, use a fixed batch size
  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      "shard": shard,
      "dataset_split": None,
      "max_records": decode_hp.num_samples
  }

  # Build the inference input function
  problem = hparams.problem
  infer_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

  predictions, output_dirs = [], []
  for decode_id in range(decode_hp.num_decodes):
    tf.logging.info("Decoding {}".format(decode_id))

    # Create decode directory if not in-memory decoding.
    if not decode_hp.decode_in_memory:
      output_dir = os.path.join(estimator.model_dir, "decode_%05d" % decode_id)
      tf.gfile.MakeDirs(output_dir)
      output_dirs.append(output_dir)

    # Get the predictions as an iterable
    predictions = estimator.predict(infer_input_fn)

    # Prepare output file writers if decode_to_file passed
    decode_to_file = decode_to_file or decode_hp.decode_to_file
    if decode_to_file:
      if decode_hp.shards > 1:
        decode_filename = decode_to_file + ("%.2d" % decode_hp.shard_id)
      else:
        decode_filename = decode_to_file
      output_filepath = decoding._decode_filename(decode_filename, problem_name,
                                                  decode_hp)
      parts = output_filepath.split(".")

      parts[-1] = "decoded"
      decoded_filepath = ".".join(parts)

      dir = os.path.dirname(decoded_filepath)
      if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)

      decoded_file = tf.gfile.Open(decoded_filepath, "w")

    problem_hparams = hparams.problem_hparams

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
              story_vocab,
              question_vocab,
              targets_vocab,
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
            story_vocab,
            question_vocab,
            targets_vocab,
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

          decoded_file.write('STORY: \n%s\n' % _make_story_pretty(str(d_story)))
          decoded_file.write('QUESTION: %s\n' %_remove_pad(str(d_question)))
          decoded_file.write('ANSWER: %s\n' % _remove_pad(str(d_target)))
          decoded_file.write('OUTPUT: %s\n' % (str(d_output) + beam_score_str + decode_hp.delimiter) )
          decoded_file.write('==================================================================\n')

      if (decode_hp.num_samples >= 0 and
          num_predictions >= decode_hp.num_samples):
        break

    if decode_to_file:
      tf.logging.info("Decoded results are written in: %s" % decoded_filepath)
      decoded_file.close()

    tf.logging.info("Completed inference on %d samples." % num_predictions)  # pylint: disable=undefined-loop-variable



def log_decode_results(stories,
                       questions,
                       outputs,
                       stories_vocab,
                       questions_vocab,
                       targets_vocab,
                       targets=None,
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
    decoded_outputs = targets_vocab.decode(outputs.flatten())
    if targets is not None:
      decoded_targets = targets_vocab.decode(targets.flatten())

  return decoded_stories, decoded_questions, decoded_outputs, decoded_targets


def _remove_pad(hyp):
  """Strips everything after the first <PAD> token, which is normally 1."""
  hyp = hyp.split()
  try:
    index = hyp.index(PAD)
    return ' '.join(hyp[0:index])
  except ValueError:
    # No PAD: return the string as-is.
    return ' '.join(hyp)


def _make_story_pretty(story):
  facts = story.replace(PAD,"").strip().split(".")
  pretty_story = ""
  for fact in facts:
    pretty_story += fact.strip() + ".\n"
  return pretty_story

def decode(estimator, hparams, decode_hp):
  """Decode from estimator. Interactive, from file, or from dataset."""
  decode_from_babi_dataset(
      estimator,
      FLAGS.problem,
      hparams,
      decode_hp,
      decode_to_file=FLAGS.decode_to_file)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  hp = t2t_decoder.create_hparams()
  decode_hp = t2t_decoder.create_decode_hparams()

  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)

  decode(estimator, hp, decode_hp)


if __name__ == "__main__":
  tf.app.run()