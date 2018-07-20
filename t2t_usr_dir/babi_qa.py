r"""Data generators for bAbi question answering dataset.


The dataset consists of 20 tasks for testing text understanding and reasoning
in the bAbI project (https://research.fb.com/downloads/babi/). The aim is that
each task tests a unique aspect of text and reasoning, and hence test different
capabilities of learning models. For more information check the following paper:
Jason Weston, Antoine Bordes, Sumit Chopra and Tomas Mikolov. Towards AI
Complete Question Answering: A Set of Prerequisite Toy Tasks, arXiv:1502.05698.
Available at: http://arxiv.org/abs/1502.05698

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import shutil
import tarfile


import six

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import babi_qa

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
PAD = text_encoder.PAD_ID


# A list of problem names that are registered by this module. This will get
# populated at module load time in the code at the bottom of this file.
REGISTERED_PROBLEMS = []


class BabiQaSentence(problem.Problem):
  """Base class for bAbi question answering problems."""
  def __init__(self, *args, **kwargs):

    super(BabiQaSentence, self).__init__(*args, **kwargs)

    self.max_sentence_length = None
    self.max_story_length = None
    self.max_question_length = None

    data_dir = os.path.expanduser(FLAGS.data_dir)
    metadata_path = os.path.join(data_dir, self.meta_data_filename())

    if tf.gfile.Exists(metadata_path):
      with tf.gfile.GFile(metadata_path, mode='r') as f:
        metadata = json.load(f)
      self.max_sentence_length = metadata['max_sentence_length']
      self.max_story_length = metadata['max_story_length']
      self.max_question_length = metadata['max_question_length']
    assert not self._was_reversed, 'This problem is not reversible!'
    assert not self._was_copy, 'This problem is not copyable!'

  @property
  def babi_subset(self):
    """The subset of dataset.
    This should be one of the following:
    {'en', 'en-10k', 'shuffled', 'shuffled-10k'}
    """
    raise NotImplementedError

  @property
  def babi_task_id(self):
    """The id of the babi task.
    This should be one of the following:
    {'qa0', 'qa1', 'qa1',...'q20'}, where qa0 means all tasks together.
    """
    raise NotImplementedError

  def dataset_filename(self):
    return 'babi_qa_' + self.babi_subset + '_' + babi_qa._TASKS[self.babi_task_id]

  @property
  def vocab_file(self):
    return self.babi_subset + '_' + babi_qa._TASKS[self.babi_task_id] + '.vocab'

  @property
  def vocab_filename(self):
      return "vocab.%s.%s" % (self.dataset_filename(),
                              text_problems.VocabType.TOKEN)
  @property
  def oov_token(self):
    """Out of vocabulary token. Only for VocabType.TOKEN."""
    return None
  
  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def get_labels_encoder(self, data_dir):
    """Builds encoder for the given class labels.
    Args:
      data_dir: data directory
    Returns:
      An encoder for class labels.
    """
    label_filepath = os.path.join(data_dir, self.vocab_filename)
    return text_encoder.TokenTextEncoder(label_filepath)


  def get_or_create_vocab(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                            replace_oov=self.oov_token)
    return encoder

  @property
  def dataset_splits(self):
    return [{'split': problem.DatasetSplit.TRAIN, 'shards': 1, },
      {'split': problem.DatasetSplit.EVAL, 'shards': 1, }]

  @property
  def is_generate_per_split(self):
    return True

  @property
  def joint_training(self):
    # training on data from all tasks.
    return True


  @property
  def truncated_story_length(self):
    if self.babi_task_id == 'qa3':
      return 130
    else:
      return 70

  def meta_data_filename(self):
    return (babi_qa._TASKS[self.babi_task_id] + '-meta_data.json')


  @property
  def num_train_shards(self):
    return self.dataset_splits[0]["shards"]

  @property
  def num_dev_shards(self):
    return self.dataset_splits[1]["shards"]


  def generate_text_for_vocab(self, data_dir, tmp_dir):
    # NOTE: for babi, we create the vocab from both train and test data.
    for dataset_split in [
        problem.DatasetSplit.TRAIN, problem.DatasetSplit.EVAL
    ]:

      for example in babi_qa._babi_parser(tmp_dir, self.babi_task_id, self.babi_subset,
                                  dataset_split, self.joint_training):

        context = ' '.join(example[babi_qa.FeatureNames.STORY])
        yield ' '.join(context.split())
        yield ' '.join(example[babi_qa.FeatureNames.QUESTION].split())
        yield example[babi_qa.FeatureNames.ANSWER]



  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """Generates training/dev data.


    Args:
      data_dir: The base directory where data and vocab files are stored.
      tmp_dir: temp directory to download and extract the dataset
      task_id: an optional integer
    """

    tmp_dir = babi_qa._prepare_babi_data(tmp_dir, data_dir)

    babi_qa._build_vocab(
        self.generate_text_for_vocab(data_dir, tmp_dir), data_dir,
        self.vocab_filename)

    train_paths = self.training_filepaths(
      data_dir, self.num_train_shards, shuffled=False)

    dev_paths = self.dev_filepaths(
      data_dir, self.num_dev_shards, shuffled=False)

    metadata_path = os.path.join(data_dir, self.meta_data_filename())


    train_parsed = babi_qa._babi_parser(tmp_dir,
                                             self.babi_task_id,
                                             self.babi_subset,
                                             problem.DatasetSplit.TRAIN,
                                             self.joint_training)

    dev_parsed = babi_qa._babi_parser(tmp_dir,
                                             self.babi_task_id,
                                             self.babi_subset,
                                             problem.DatasetSplit.EVAL,
                                             self.joint_training)

    encoder = self.get_or_create_vocab(data_dir)
    label_encoder = self.get_labels_encoder(data_dir)

    train_parsed_processed, dev_parsed_processed = (
      self._preprocess_babi_input_data(train_parsed, dev_parsed, encoder,
                                       label_encoder, metadata_path))

    generator_utils.generate_dataset_and_shuffle(
      self.generator(train_parsed_processed), train_paths,
      self.generator(dev_parsed_processed), dev_paths)

  def _preprocess_babi_input_data(self, train_parsed, dev_parsed, encoder_text,
                                  encoder_label, metadata_path):

    train_parsed_processed = []
    dev_parsed_processed = []

    def truncate_story(story):
      'Truncate a story to the specified maximum length.'
      return story[-self.truncated_story_length:]

    def example_processor(example, encoder_text, encoder_label):

      story = [encoder_text.encode(sentence) for sentence in
        truncate_story(example[babi_qa.FeatureNames.STORY])]
      question = encoder_text.encode(example[babi_qa.FeatureNames.QUESTION])
      answer = encoder_label.encode(example[babi_qa.FeatureNames.ANSWER])

      sentence_length.extend([len(sentence) for sentence in story])
      story_length.append(len(story))
      question_length.append(len(question))

      example[babi_qa.FeatureNames.STORY] = story
      example[babi_qa.FeatureNames.QUESTION] = question
      example[babi_qa.FeatureNames.ANSWER] = answer

      return example

    sentence_length = []
    story_length = []
    question_length = []

    for example in train_parsed:
      train_parsed_processed.append(
        example_processor(example, encoder_text, encoder_label))

    for example in dev_parsed:
      dev_parsed_processed.append(
        example_processor(example, encoder_text, encoder_label))

    self.max_sentence_length = max(sentence_length)
    self.max_story_length = max(story_length)
    self.max_question_length = max(question_length)

    with tf.gfile.Open(metadata_path, 'w') as f:
      f.write(json.dumps({'max_sentence_length': self.max_sentence_length,
        'max_story_length': self.max_story_length,
        'max_question_length': self.max_question_length}))

    return train_parsed_processed, dev_parsed_processed

  def generator(self, examples):
    """Reads examples and encodes them using the given encoders.

    Args:
      examples: all the examples in the data parsed by the dataset parser

    Yields:
      tf_examples that are encoded based ont the given encoders
    """

    def pad_input(story, question):
      'Pad sentences, stories, and queries to a consistence length.'
      for sentence in story:
        for _ in range(self.max_sentence_length - len(sentence)):
          sentence.append(PAD)
        assert len(sentence) == self.max_sentence_length

      for _ in range(self.max_story_length - len(story)):
        story.append([PAD for _ in range(self.max_sentence_length)])

      for _ in range(self.max_question_length - len(question)):
        question.append(PAD)

      assert len(story) == self.max_story_length
      assert len(question) == self.max_question_length

      return story, question

    for example in examples:
      story, question = pad_input(example[babi_qa.FeatureNames.STORY],
                                  example[babi_qa.FeatureNames.QUESTION])

      story_flat = [token_id for sentence in story for token_id in sentence]
      answer = example[babi_qa.FeatureNames.ANSWER]

      yield {babi_qa.FeatureNames.STORY: story_flat, babi_qa.FeatureNames.QUESTION: question,
             babi_qa.FeatureNames.ANSWER: answer, }

  def example_reading_spec(self):
    """Specify the names and types of the features on disk.

    Returns:
      The names and type of features.
    """
    data_fields = {
      babi_qa.FeatureNames.STORY: tf.FixedLenFeature(
        shape=[self.max_story_length, self.max_sentence_length], dtype=tf.int64),
      babi_qa.FeatureNames.QUESTION: tf.FixedLenFeature(
        shape=[1, self.max_question_length], dtype=tf.int64),
      babi_qa.FeatureNames.ANSWER: tf.FixedLenFeature(shape=[1], dtype=tf.int64)}

    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, mode, unused_hparams):
    """Preprocesses the example feature dict.

    Args:
      example: input example
      mode: training, eval, and inference
      unused_hparams: -

    Returns:
      The processed example
    """
    # add feature 'targets' to the example which is equal to Answer
    example['targets'] = example[babi_qa.FeatureNames.ANSWER]
    # In T2T, features are supposed to enter the pipeline as 3d tensors.
    # "inputs" and "targets" will be expended to 3d if they're not,
    # and we should expand other features if we define any
    example[babi_qa.FeatureNames.STORY] = tf.expand_dims(example[babi_qa.FeatureNames.STORY],
                                                 -1)
    example[babi_qa.FeatureNames.QUESTION] = tf.expand_dims(
        example[babi_qa.FeatureNames.QUESTION], -1)
    tf.logging.info(example[babi_qa.FeatureNames.QUESTION])
    return example

  def feature_encoders(self, data_dir):
    """Determines how features from each example should be encoded.

    Args:
      data_dir: The base directory where data and vocab files are stored.

    Returns:
      A dict of <feature name, Encoder> for encoding and decoding inference
       input/output.
    """
    # todo(dehghani): fix this to be able to use subword encoder as well.
    assert self.vocab_type == text_problems.VocabType.TOKEN

    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                            replace_oov=self.oov_token)
    label_encoder = self.get_labels_encoder(data_dir)


    return {babi_qa.FeatureNames.STORY: encoder,
            babi_qa.FeatureNames.QUESTION: encoder, 'targets': label_encoder}

  def hparams(self, defaults, unused_model_hparams):
    """Defines model hyperparameters.

    Args:
      defaults: default hparams
      unused_model_hparams: -

    """
    p = defaults
    p.stop_at_eos = int(True)

    story_vocab_size = self._encoders[babi_qa.FeatureNames.STORY].vocab_size
    question_vocab_size = self._encoders[babi_qa.FeatureNames.QUESTION].vocab_size

    p.input_modality = {
      babi_qa.FeatureNames.STORY: (registry.Modalities.SYMBOL, story_vocab_size),
      babi_qa.FeatureNames.QUESTION: (
      registry.Modalities.SYMBOL, question_vocab_size)}
    num_classes = self._encoders['targets'].vocab_size
    p.target_modality = (registry.Modalities.CLASS_LABEL, num_classes)

  def eval_metrics(self):
    """Specify the set of evaluation metrics for this problem.
    Returns:
      List of evaluation metrics of interest.
    """
    return [metrics.Metrics.ACC]

def _problems_to_register():
  """Problems for which we want to create datasets.

  To avoid a long file with class definition boilerplate for each problem, we
  are dynamically creating and registering problems. The set of problems to
  register is defined by this function. See below for the code that creates the
  classes and registers the problems.

  Returns:
    A dictionary mapping problem name to babi_task_id.
  """
  all_problems = {}

  # First define some problems using only concrete characters (i.e., no meta
  # characters).
  problems_on_different_tasks = {
      'AllTasks': 'qa0',
      'Task1': 'qa1',
      'Task2': 'qa2',
      'Task3': 'qa3',
      'Task4': 'qa4',
      'Task5': 'qa5',
      'Task6': 'qa6',
      'Task7': 'qa7',
      'Task8': 'qa8',
      'Task9': 'qa9',
      'Task10': 'qa10',
      'Task11': 'qa11',
      'Task12': 'qa12',
      'Task13': 'qa13',
      'Task14': 'qa14',
      'Task15': 'qa15',
      'Task16': 'qa16',
      'Task17': 'qa17',
      'Task18': 'qa18',
      'Task19': 'qa19',
      'Task20': 'qa20',
  }
  all_problems.update(problems_on_different_tasks)

  return all_problems

class BabiQaSentenceSingle(BabiQaSentence):
  @property
  def joint_training(self):
    # training on data from all tasks.
    return False



def _register_babi_problems():
  """It dynamically instantiates a class for each babi subsets-tasks.

   @registry.register_problem
   class BabiQaConcatAllTasks_10k(EditSequenceRegexProblem):
     @property
     def babi_task_id(self):
       return 'qa0'
     @property
     def babi_subset(self):
      return 'en-10k'

  It does not put the classes into the global namespace, so to access the class
  we rely on the registry or this module's REGISTERED_PROBLEMS list.
  It will be available as

     registry.problem('babi_qa_concat_all_tasks_10k')

  i.e., change camel case to snake case. Numbers are considered lower case
  characters for these purposes.
  """

  for (subset, subset_suffix) in [('en', '_1k'), ('en-10k', '_10k')]:
    for problem_name, babi_task_id in six.iteritems(_problems_to_register()):
      problem_class = type('BabiQaSentence' + problem_name + subset_suffix,
                           (BabiQaSentence,), {
                               'babi_task_id': babi_task_id,
                               'babi_subset': subset
                           })
      registry.register_problem(problem_class)
      REGISTERED_PROBLEMS.append(problem_class.name)


  for (subset, subset_suffix) in [('en', '_1k'), ('en-10k', '_10k')]:
    for problem_name, babi_task_id in six.iteritems(_problems_to_register()):
      problem_class = type('BabiQaSentenceSingle' + problem_name + subset_suffix,
                           (BabiQaSentenceSingle,), {
                               'babi_task_id': babi_task_id,
                               'babi_subset': subset
                           })
      registry.register_problem(problem_class)
      REGISTERED_PROBLEMS.append(problem_class.name)

_register_babi_problems()
