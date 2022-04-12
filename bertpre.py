#import BookscorpusTextFormatting
import WikiTextFormat
import BooksTextFormat
import TextSharding
import argparse
from GooglePretrainedWeightDownloader import GooglePretrainedWeightDownloader
import itertools
import multiprocessing
import os
import pprint
import subprocess

os.environ['BERT_PREP_WORKING_DIR']='/media/cjh/My Passport/Study/Nus_Project'
def main(args):
    working_dir= os.environ['BERT_PREP_WORKING_DIR']
    print("Bert Preparation")
    print('Working Directory:', working_dir)


    hdf5_tfrecord_folder_prefix = "/lower_case_" + str(args.do_lower_case) + "_seq_len_" + str(args.max_seq_length) \
                                  + "_max_pred_" + str(args.max_predictions_per_seq) + "_masked_lm_prob_" + str(
        args.masked_lm_prob) \
                                  + "_random_seed_" + str(args.random_seed) + "_dupe_factor_" + str(args.dupe_factor) \
                                  + "_shard_" + str(args.n_training_shards) + "_test_split_" + str(
        int(args.fraction_test_set * 100))

    directory_structure = {
        'download': working_dir + '/download',  # Downloaded and decompressed
        'extracted': working_dir + '/extracted',  # Extracted from whatever the initial format is (e.g., wikiextractor)
        'formatted': working_dir + '/formatted_one_article_per_line',
        # This is the level where all sources should look the same
        'sharded': working_dir + '/sharded',
        'tfrecord': working_dir + '/tfrecord' + hdf5_tfrecord_folder_prefix,
        'hdf5': working_dir + '/hdf5' + hdf5_tfrecord_folder_prefix,
    }
    if args.action == 'download':
        if not os.path.exists(directory_structure['download']):
            os.makedirs(directory_structure['download'])

        downloader = GooglePretrainedWeightDownloader(directory_structure['download'])
        downloader.download()

    elif args.action == 'text_formatting':
        if not os.path.exists(directory_structure['formatted']):
            os.makedirs(directory_structure['formatted'])

        if args.dataset == 'wikicorpus_en':
            wiki_path = directory_structure['extracted'] + '/wikicorpus_en'
            output_filename = directory_structure['formatted'] + '/wikicorpus_en_one_article_per_line.txt'
            wiki_formatter = WikiTextFormat.WikicorpusTextFormatting(wiki_path, output_filename, recursive=True)
            wiki_formatter.merge()

        if args.dataset =='bookscorpus':
            books_path = directory_structure['extracted'] + '/bookscorpus/epubtxt'

            output_filename = directory_structure['formatted'] + '/bookscorpus_one_book_per_line.txt'
            books_formatter = BooksTextFormat.BookscorpusTextFormatting(books_path, output_filename,
                                                                                  recursive=True)
            books_formatter.merge()
    elif args.action == 'sharding':

        args.input_files = [directory_structure['formatted'] + '/bookscorpus_one_book_per_line.txt', directory_structure['formatted'] + '/wikicorpus_en_one_article_per_line.txt']

        if not os.path.exists(directory_structure['sharded']):
            os.makedirs(directory_structure['sharded'])

        if not os.path.exists(directory_structure['sharded'] + '/' + args.dataset):
            os.makedirs(directory_structure['sharded'] + '/' + args.dataset)

        if not os.path.exists(directory_structure['sharded'] + '/' + args.dataset + '/training'):
            os.makedirs(directory_structure['sharded'] + '/' + args.dataset + '/training')

        if not os.path.exists(directory_structure['sharded'] + '/' + args.dataset + '/test'):
            os.makedirs(directory_structure['sharded'] + '/' + args.dataset + '/test')

        output_file_prefix = directory_structure['sharded'] + '/' + args.dataset + '/' + args.dataset

        segmenter = TextSharding.NLTKSegmenter()
        sharding = TextSharding.Sharding(args.input_files, output_file_prefix, args.n_training_shards,
                                         args.n_test_shards, args.fraction_test_set)

        sharding.load_articles()
        sharding.segment_articles_into_sentences(segmenter)
        sharding.distribute_articles_over_shards()
        sharding.write_shards_to_disk()


    elif args.action == 'create_tfrecord_files':

        if not os.path.exists(directory_structure['tfrecord'] + "/" + args.dataset):
            os.makedirs(directory_structure['tfrecord'] + "/" + args.dataset)

        if not os.path.exists(directory_structure['tfrecord'] + "/" + args.dataset + '/training'):
            os.makedirs(directory_structure['tfrecord'] + "/" + args.dataset + '/training')

        if not os.path.exists(directory_structure['tfrecord'] + "/" + args.dataset + '/test'):
            os.makedirs(directory_structure['tfrecord'] + "/" + args.dataset + '/test')

        last_process = None

        def create_record_worker(filename_prefix, shard_id, output_format='tfrecord', split='training'):

            bert_preprocessing_command = 'python create_pretraining_data.py'

            bert_preprocessing_command += ' --input_file=' +'\''+ directory_structure[
                'sharded'] +'\'' +'/' + args.dataset + '/' + split + '/' + filename_prefix + '_' + str(shard_id) + '.txt'

            bert_preprocessing_command += ' --output_file=' + '\''+directory_structure[
                'tfrecord']+'\'' + '/' + args.dataset + '/' + split + '/' + filename_prefix + '_' + str(
                shard_id) + '.' + output_format

            bert_preprocessing_command += ' --vocab_file=' + '\''+directory_structure[
                'download'] +'\''+ '/'+ args.vocab_file

            bert_preprocessing_command += ' --do_lower_case' if args.do_lower_case else ''

            bert_preprocessing_command += ' --max_seq_length=' + str(args.max_seq_length)

            bert_preprocessing_command += ' --max_predictions_per_seq=' + str(args.max_predictions_per_seq)

            bert_preprocessing_command += ' --masked_lm_prob=' + str(args.masked_lm_prob)

            bert_preprocessing_command += ' --random_seed=' + str(args.random_seed)

            bert_preprocessing_command += ' --dupe_factor=' + str(args.dupe_factor)

            bert_preprocessing_process = subprocess.Popen(bert_preprocessing_command, shell=True)

            last_process = bert_preprocessing_process

            # This could be better optimized (fine if all take equal time)

            if shard_id % args.n_processes == 0 and shard_id > 0:
                bert_preprocessing_process.wait()

            return last_process

        output_file_prefix = args.dataset

        for i in range(args.n_training_shards):
            last_process = create_record_worker(output_file_prefix + '_training', i, 'tfrecord', 'training')

        last_process.wait()

        for i in range(args.n_test_shards):
            last_process = create_record_worker(output_file_prefix + '_test', i, 'tfrecord', 'test')

        last_process.wait()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for Everything BERT-related'
    )
    parser.add_argument(
        '--action',
        type=str,
        help='Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords',
        choices={
            'download',  # Download and verify mdf5/sha sums
            'text_formatting',  # Convert into a file that contains one article/book per line
            'sharding',  # Convert previous formatted text into shards containing one sentence per line
            'create_tfrecord_files',  # Turn each shard into a TFrecord with masking and next sentence prediction info
            'create_hdf5_files'  # Turn each shard into a HDF5 file with masking and next sentence prediction info
        }
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify the dataset to perform --action on',
        choices={
            'bookscorpus',
            'wikicorpus_en',
            'wikicorpus_zh',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 'books_wiki_en_corpus',
            'pubmed_baseline',
            'pubmed_daily_update',
            'pubmed_fulltext',
            'pubmed_open_access',
            'google_pretrained_weights',
            'nvidia_pretrained_weights',
            'squad',
            'mrpc',
            'sst-2',
            'mnli',
            'cola',
            'all'
        }
    )

    parser.add_argument(
        '--input_files',
        type=str,
        help='Specify the input files in a comma-separated list (no spaces)'
    )

    parser.add_argument(
        '--n_training_shards',
        type=int,
        help='Specify the number of training shards to generate',
        default=1472
    )

    parser.add_argument(
        '--n_test_shards',
        type=int,
        help='Specify the number of test shards to generate',
        default=1472
    )

    parser.add_argument(
        '--fraction_test_set',
        type=float,
        help='Specify the fraction (0..1) of the data to withhold for the test data split (based on number of sequences)',
        default=0.1
    )

    parser.add_argument(
        '--segmentation_method',
        type=str,
        help='Specify your choice of sentence segmentation',
        choices={
            'nltk'
        },
        default='nltk'
    )

    parser.add_argument(
        '--n_processes',
        type=int,
        help='Specify the max number of processes to allow at one time',
        default=6
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Specify the base seed to use for any random number generation',
        default=12345
    )

    parser.add_argument(
        '--dupe_factor',
        type=int,
        help='Specify the duplication factor',
        default=5
    )

    parser.add_argument(
        '--masked_lm_prob',
        type=float,
        help='Specify the probability for masked lm',
        default=0.15
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        help='Specify the maximum sequence length',
        default=512
    )

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        help='Specify the maximum number of masked words per sequence',
        default=20
    )

    parser.add_argument(
        '--do_lower_case',
        type=int,
        help='Specify whether it is cased (0) or uncased (1) (any number greater than 0 will be treated as uncased)',
        default=1
    )

    parser.add_argument(
        '--vocab_file',
        type=str,
        help='Specify absolute path to vocab file to use)'
    )

    parser.add_argument(
        '--skip_wikiextractor',
        type=int,
        help='Specify whether to skip wikiextractor step 0=False, 1=True',
        default=0
    )

    parser.add_argument(
        '--interactive_json_config_generator',
        type=str,
        help='Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords'
    )

    args=parser.parse_args()
    main(args)