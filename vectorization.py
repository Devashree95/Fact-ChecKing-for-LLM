from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, WikiCorpus, MmCorpus
from gensim import similarities
from gensim import utils
import time
import sys
import logging
import os


def formatTime(seconds):
    """
    Takes a number of elapsed seconds and returns a string in the format h:mm.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d" % (h, m)
 
if __name__ == '__main__':
    
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Create a logger
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(filename='log.txt', format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%H:%M:%S')
    logging.root.setLevel(level=logging.INFO)

    ######################## Step 1 : Convert xml to dictionaly.txt ##############################################
  
    dump_file = 'enwiki-20230920-pages-articles-multistream1.xml-p1p41242.bz2'
    

    if True:    

        # Create an empty dictionary
        dictionary = Dictionary()
        
        # Create the WikiCorpus object. This doesn't do any processing yet since
        # we've supplied the dictionary.
        wiki = WikiCorpus(dump_file, dictionary=dictionary) 
        print(wiki.get_texts())
        
        print('Parsing Wikipedia to build Dictionary...')    
        sys.stdout.flush()
        
        t0 = time.time()

        dictionary.add_documents(wiki.get_texts(), prune_at=None)            
                        
        print('    Building dictionary took %s' % formatTime(time.time() - t0))
        print('    %d unique tokens before pruning.' % len(dictionary))
        sys.stdout.flush()
        
        keep_words = 100000    
    
        wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=keep_words)
        

        wiki.dictionary.save_as_text('dictionary.txt.bz2')
    else:
        # Nothing to do here.
        print('')
################################# STEP 2 ##############################################################
    if True:
    
        # Load the dictionary if you're just running this section.
        dictionary = Dictionary.load_from_text('dictionary.txt.bz2')
        wiki = WikiCorpus(dump_file, dictionary=dictionary)    
    
        # Turn on metadata so that wiki.get_texts() returns the article titles.
        wiki.metadata = True         
    
        print('\nConverting to bag of words...')
        sys.stdout.flush()
        
        t0 = time.time()
    
        # Generate bag-of-words vectors (term-document frequency matrix) and 
        # write these directly to disk.
        MmCorpus.serialize('bow.mm', wiki, metadata=True, progress_cnt=10000)
        
        print('    Conversion to bag-of-words took %s' % formatTime(time.time() - t0))
        sys.stdout.flush()

        # Load the article titles back
        id_to_titles = utils.unpickle('bow.mm.metadata.cpickle')
    
        # Create the reverse mapping, from article title to index.
        titles_to_id = {}

        # For each article
        for at in id_to_titles.items():
            # `at` is (index, (pageid, article_title))  e.g., (0, ('12', 'Anarchism'))
            # at[1][1] is the article title.
            # The pagied property is unused.
            titles_to_id[at[1][1]] = at[0]
        
        # Store the resulting map.
        utils.pickle(titles_to_id, 'titles_to_id.pickle')

        # We're done with the article titles so free up their memory.
        del id_to_titles
        del titles_to_id
    
    
        # To clean up some memory, we can delete our original dictionary and 
        # wiki objects, and load back the dictionary directly from the file.
        del dictionary
        del wiki  
        
        # Load the dictionary back from disk.
        dictionary = Dictionary.load_from_text('dictionary.txt.bz2')
        corpus_bow = MmCorpus('bow.mm')    
    
    else:
        print('\nLoading the bag-of-words corpus from disk.')
        corpus_bow = MmCorpus('bow.mm')    

    
    # ======== STEP 3: Learn tf-idf model ========
    if True:
        print('\nLearning tf-idf model from data...')
        t0 = time.time()
        
        # TODO - Why not normalize?
        model_tfidf = TfidfModel(corpus_bow, id2word=dictionary, normalize=False)

        print('    Building tf-idf model took %s' % formatTime(time.time() - t0))
        model_tfidf.save('tfidf.tfidf_model')
    
    # If we previously completed this step, just load the pieces we need.
    else:
        print('\nLoading the tf-idf model from disk.')
        model_tfidf = TfidfModel.load('tfidf.tfidf_model') 
        

    # ======== STEP 4: Convert articles to tf-idf ======== 
    if True:
        print('\nApplying tf-idf model to all vectors...')
        t0 = time.time()
        
        # Apply the tf-idf model to all of the vectors.    
        MmCorpus.serialize('corpus_tfidf.mm', model_tfidf[corpus_bow], progress_cnt=10000)
        
        print('    Applying tf-idf model took %s' % formatTime(time.time() - t0))
    else:
        # Nothing to do here.
        print('')

    # ======== STEP 5: Train LSI on the articles ========
    # Learn an LSI model from the tf-idf vectors.
    if True:
        
        # The number of topics to use.
        num_topics = 300
        
        # Load the tf-idf corpus back from disk.
        corpus_tfidf = MmCorpus('corpus_tfidf.mm')        
        
        # Train LSI
        print('\nLearning LSI model from the tf-idf vectors...')
        t0 = time.time()
        
        # Build the LSI model
        model_lsi = LsiModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)   
    
        print('    Building LSI model took %s' % formatTime(time.time() - t0))

        # Write out the LSI model to disk.
        model_lsi.save('lsi.lsi_model')
    else:
        # Load the tf-idf corpus and trained LSI model back from disk.
        corpus_tfidf = MmCorpus('corpus_tfidf.mm')
        model_lsi = LsiModel.load('lsi.lsi_model')
    
    # ========= STEP 6: Convert articles to LSI with index ========
    # Transform corpus to LSI space and index it
    if True:
        
        print('\nApplying LSI model to all vectors...')
        t0 = time.time()
           
        index = similarities.MatrixSimilarity(model_lsi[corpus_tfidf], num_features=num_topics)
        index.save('lsi_index.mm')
        
        print('    Applying LSI model took %s' % formatTime(time.time() - t0))
