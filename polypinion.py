'''
These functions are used in main
'''

from openai import OpenAI
import keys
import requests
import pandas as pd
import json
import uuid
import spacy
from inter_sql import all_it_from_ns, rank_from_desc, key_id_from_desc, commit_live_articles, commit_old_articles, commit_rankings, commit_cat, commit_img,articles_from_key, replace_live_articles, commit_urls
import datetime as dt

client = OpenAI(
            api_key=keys.gbt_api_key)
nlp = spacy.load('en_core_web_sm')


class newsapi:
    '''
    Class to query all top headlines from newsapi.
    '''

    def __init__(self, sources):
        self.sources = sources

    def top_stories(self):
        sources = self.sources
        desc = []
        headline = []
        url_ = []
        content = []
        img = []

        for i in sources:
            url = ('https://newsapi.org/v2/top-headlines?'
                   f'sources={i}&'
                   f'apiKey={keys.news_api_key2}')

            out = requests.get(url)
            out = out.json()['articles']
            len_ = len(out)
            for ii in range(0, len_):
                desc.append(out[ii]['description'])
                headline.append(out[ii]['title'])
                url_.append(out[ii]['url'])
                content.append(out[ii]['content'])
                img.append(out[ii]['urlToImage'])

        # remove double quotes
        desc_ = []
        for i in desc:
            if i is None:
                desc_.append(None)
            else:
                m = i.replace('"', "'")
                desc_.append(m)

        return desc_, headline, url_, content, img



class gbt_eval:
    '''
    class for evaluating clusters / news articles

    - cluster_events: clustering new news articles
    - eval: compare new and existing articles:
        -- new articles added, old articles dropped
        -- new clusters same headline: append
    '''

    def __init__(self, desc, headline, url_, content):
        self.desc = desc
        self.headline = headline
        self.url_ = url_
        self.contect = content

    def cluster_events(self):
        desc = self.desc

        client = OpenAI(
            api_key=keys.gbt_api_key)

        cluster = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": '''
                 You are given article descriptions and corresponding headlines. 
                 You can look at a group of articles, and use intuition to put related articles into the same cluster.
                 Take in the DESCRIPTIONS lists. Respond to the below prompt answering the TITLE, GIST, and DESC sections. 
                 Only use double quotes at the beginning and end data inside lists, if double quotes are used in articles, change them to single.
                 '''},
                {"role": "user",
                 "content": f'DESCRIPTIONS: {desc}. Group the list of article descriptions into groups of three articles minimum that '
                            f'are reporting on the same event. Each cluster needs a minimum of 3 sources. These need to be multiple stories on the same specific event.'
                            ''' An event is something that shares the same proper nouns, or specific events. Return text in the following json format:
                    {"out": [
                     {"TITLE": "Create a specific title with no more than 5 words",
                     "GIST": "clusters theme in 2 sentences",
                     "DESC": ["all descriptions that belong to this cluster",...]
                        },
                    {"TITLE": "Create a specific title with no more than 5 words",
                     "GIST": "clusters theme in 2 sentences",
                     "DESC": ["the descriptions that belong to this cluster",...]
                        }
                        ]
                        }
                '''}])
        return cluster


class gbt_author:
    '''
    class for taking new info and creating articles
    '''

    def __init__(self, cluster):
        self.cluster = cluster

    def to_json(self):
        event_clusters = self.cluster
        json_object = event_clusters.choices[0].message.content
        json_object_cln = json_object.replace("['",'["').replace("']",'"]').replace("\\xa0", "").replace("',",'",').replace(", '",', "').replace("', '",'", "').replace('\x01\x02\x10\x13','')
        json_object = json.loads(json_object_cln)
        self.json_object = json_object
        return json_object

    def parser(self, jso):
        df = pd.DataFrame.from_dict(jso['out'])

        new = df['GIST'].tolist()

        # Get all headlines and articles
        ref_hl, ref_desc = all_it_from_ns('title','live_articles'), all_it_from_ns('body','live_articles')
        #ref_hl,ref_desc = all_it_from_ns('headline'),all_it_from_ns('article')

        # similarity score
        out_ = self.attr_new_to_existing_cluser(ref_desc, new)
        #out_ = attr_new_to_existing_cluser(ref_desc, new)

        ref_index = []
        ref_db = []
        tups_hl = []
        tups_desc = []
        desc_ = []
        hl_ = []
        for i in range(0, len(new)):
            try:
                tups_hl.append((new[i], ref_hl[out_[i]])) # saves the new headlines and most similar headline
                tups_desc.append((new[i], ref_desc[out_[i]])) # saves the new cluster and most similar cluster
                ref_index.append(out_[i])
                desc = rank_from_desc(ref_desc[out_[i]])
                #desc = rank_from_desc(ref_desc[out_[i]]) # finds the rank for a description
                desc_.append(ref_desc[out_[i]])  # returns a list of articles, out_i gives an index
                hl_.append(ref_hl[out_[i]])
                ref_db.append(desc)
            except:
                tups_hl.append((new[i], 'no close relation'))
                tups_desc.append((new[i], 'no close relation'))
                ref_db.append('no close relation')
                desc_.append('no close relation')
                hl_.append('no close relation')

        out = self.evaluator_gbt(tups_hl, tups_desc)
        #out = evaluator_gbt(tups_hl, tups_desc)
        in_out = out.choices[0].message.content
        lst = eval(in_out)
        df['duplicate_info'],df['sme_hl'],df['sim_db_rnk'],df['sim_desc'],df['sim_hl'] = lst[0],lst[1],ref_db,desc_,hl_

        self.df = df
        return df

    def append_clusters(self, gists, desc):

        tup = list(zip(gists, desc))

        client = OpenAI(
            api_key=keys.gbt_api_key)

        app_cluster = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": '''
                       You are given a multiple tuples. The second value in each tuple contains multiple descriptions of a news event. Use these descriptions to complete the following tasks.
                       Task 1) Create TITLE_A, TITLE_B and TITLE_C. 
                            For TITLE_A: Create a newspaper headline based on descriptions that's extremely opinionated. 
                            For TITLE_B: Create a newspaper headline based on descriptions that's sarcastic. 
                            For TITLE_C: Create a newspaper headline based on descriptions with minimal words.
                       Task 2) Tag the description with all relevant categories based on the provided list, you can use multiple. List od categories: 'Politics', 'World News', 'Business' , 'Finance', 'Science' , 'Sports', 'Environment' , 'Arts & Entertainment', 'Misc'.
                       Only use double quotes at the beginning and end data inside lists. Only return the asked for information in the json example below.
                       '''},
                {"role": "user",
                 "content": f'TUPLES: {tup}. Using the second value in each tuple, create a TITLE_A and TITLE_B and assign appropriate clusters.'
                            ''' Return text in the following json format:
                    {"out": [
                    {"TITLE_A": "Extremely Opinionated Headline",
                     "TITLE_B": "Sarcastic Headline",
                     "TITLE_C": "Minimal Headline",
                     "CAT": ["category1", "category2",...]
                        },
                    {"TITLE_A": "Extremely Opinionated Headline",
                     "TITLE_B": "Sarcastic Headline",
                     "TITLE_C": "Minimal Headline",
                     "CAT": ["category1", "category2",...]
                        }
                        }
                        ]
                        }
                '''}])

        return app_cluster

    def get_art_data(self, img, desc,df):

        tup = list(zip(img, desc))
        flt_tup = [(i, f) for i, f in tup if i is not None]

        img_lst = []
        for m,r in df.iterrows():
            out = [i for i, f in flt_tup if f in r['DESC']]
            if len(out)==0:
                img_lst.append('NA')
            else:
                img_lst.append(out[0])

        return img_lst


    def find_existing_cluster(self,ref, new_item):
        art_ = []
        scr_ = []
        for i in range(0, len(ref)):
            ref_, new_ = nlp(ref[i]), nlp(new_item)
            ref_ = nlp(' '.join([str(t) for t in ref_ if t.pos_ in ['NOUN', 'PROPN']]))
            new_ = nlp(' '.join([str(t) for t in new_ if t.pos_ in ['NOUN', 'PROPN']]))
            sim = new_.similarity(ref_)
            art_.append(i)
            scr_.append(sim)

        # return those with the highest score
        tup = tuple(zip(art_,scr_))

        try:
            tuple_with_highest_value = max(tup, key=lambda x: x[1])

        except:
            tuple_with_highest_value = ('no sim', 0)

        return tuple_with_highest_value[0]

    def attr_new_to_existing_cluser(self, ref, new):
        sim_ind = []
        for i in range(0, len(new)):
            sim_ind_ = self.find_existing_cluster(ref, new[i])
            sim_ind.append(sim_ind_)
        return sim_ind

    def evaluator_gbt(self, tups_hl,tups_desc):
        client = OpenAI(
            api_key=keys.gbt_api_key)

        append_or_neglect = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": '''
                         You are given two lists of tuples.
                         Go through the list and evaluate each set of tuples, respond with a value corresponding to that pair of tuples in the list.
                         The first tuple value is a description a news cluster. The second is a a headline about a similar news cluster.
                          respond 0. You will create two lists as specified. Respond with each list, of 1 or 0 each value corresponding to the tuple in the list. The response should be in the following format: [[0,...][0,...]].
                          Give only the lists in the response.
                         '''},
                {"role": "user",
                 "content": f'''
                         First list: Use {tups_desc}. 
                         The first tuple value is a description of a news cluster. The second is an article on a similar news cluster.
                         If the first tuple value gives a majority of duplicate information to the second value, give 1. Else, give 0.
                         
                         Second list: use {tups_hl}
                         The first tuple value is a description a news cluster. The second is a headline.
                         If information from the first tuple value falls under the headline of second tuple value, give 1. Else, give 0.

                          '''}
            ]
        )

        return append_or_neglect

    def writer_new(self, url_, desc, img, df):

        #df = self.df

        tup = list(zip(url_, desc, img))

        #  unique information
        df_ = df[(df['duplicate_info'] == 0) & (df['sme_hl'] == 0)].reset_index()

        if df_.empty:
            pass
        else:
            stories = []
            urls_ = []
            img_ = []
            rge = range(0, len(df_['DESC']))


            for ii in rge:
                out = [(i, f) for i, f, k in tup if f in df_['DESC'][ii]]
                urls__ = [i for i, f, k in tup if f in df_['DESC'][ii]]
                img__ = [k for i, f, k in tup if f in df_['DESC'][ii]]

                writer = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system",
                         "content": '''
                                    You are given a list of tuples, the first entry in each tuple contains a url, the second a small description of an event.
                                    Using the small descriptions craft an unbiased news article at 150 words.
                                    '''},
                        {"role": "user",
                         "content": f'''
                                        TUPLES:{out}. 
                                        Create a news article using the descriptions in the tupple.'''}
                    ]
                )
                stories.append(writer.choices[0].message.content)
                urls_.append(urls__)
                img_.append(img__)

            df_['article'] = stories
            df_['urls'] = urls_
            df_['imgs'] = img_

        return df_

    def writer_existing(self, url_, desc, img, df):
        tup = list(zip(url_, desc,img))

        # unique information under an existing headline
        df__ = df[(df['duplicate_info'] == 0) & (df['sme_hl'] == 1)].reset_index()

        if df__.empty:
            df__['key_ids'] = 'NA'

        else:
            stories = []
            urls_ = []
            img_ = []
            key_ids = []

            rge = range(0, len(df__['DESC']))
            for ii in rge:
                out = [(i, f) for i, f, k in tup if f in df__['DESC'][ii]]
                urls__ = [i for i, f, k in tup if f in df__['DESC'][ii]]
                img__ = [k for i, f, k in tup if f in df__['DESC'][ii]]

                desc_sim = df__['sim_desc'][ii]
                k_id = key_id_from_desc(desc_sim)

                writer = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system",
                         "content": '''
                                            You are given a list of tuples, the first entry in each tuple contains a url,the second entry is a small description of an event. You are also given an existing article.
                                            Using the descriptions amend the exiting news article with new information from the decription. Put UPDATE: then add the ammended content at the bottom of the articles. 
                                            The amended news article should be unbiased and 150 words. Ensure the news article adds information not covered in the existing article. 
                                            '''},
                        {"role": "user",
                         "content": f'''
                                                TUPLES:{out}. EXISTING ARTICLE:{desc_sim}
                                                Create a news article using the descriptions in the tuple. Avoid duplicating information covered in the EXISTING ARTICLE'''}
                    ]
                )


                urls_.append(urls__)
                img_.append(img__)
                key_ids.append(k_id)
                stories.append(writer.choices[0].message.content)


            df__['article'] = stories
            df__['imgs'] = img_
            df__['urls'] = urls_
            df__['key_ids'] = key_ids

        return df__

    def annotator(self, articles):
        if len(articles.index) == 1:

            arts = articles['article'][0].split(". ")

            i = 0
            annotations = []
            while i < 5:
                writer = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": ''' 
                                     You are given a list of sentences.
                                     For each sentence return the most applicable letter: P if an argument is made with pathos, E if an argument is made with ethos, L if an argument is made with logos.
                                     Each tuple should have a corresponding letter. 
                                     Pathos is any instance where emotions are is appealed to in an argument.
                                     Ethos is any instance where the credibility of the author or an external character is appealed to in the argument.
                                     Logos is any instance where logical connections between ideas, such as facts, statistics, history, analogies are used in the argument.
                                    '''},
                        {"role": "user",
                         "content": f'''
                                     TUPLES:{arts}
                                     For the sentence in each tuple return a P, E, L, or N to the letter that best corresponds with that data.
                                     Here is an example of how to format the data [P,E, ...]  '''}
                    ]
                )

                lst_str = writer.choices[0].message.content
                # Remove square brackets and split by commas
                elements = lst_str[1:-1].split(',')

                # Strip whitespace and convert to string
                annotes = [element.strip() for element in elements]

                annotations.append(annotes)
                i += 1

                # parse through
                logic = []
                conf = []
                line_num = []
                df = pd.DataFrame(annotations)
                for iii in range(0, len(df.columns)):
                    # most frequent
                    let = df[iii].mode().values[0]
                    try:
                        num = df[iii].value_counts()[let]
                        logic.append(let), conf.append(num / 5), line_num.append(iii)
                    except:
                        pass


        elif len(articles.index) > 1:
            annotations = []
            for r,ii in articles.iterrows():
                arts = ii['article'].split(". ")

                i = 0
                annotations_ = []
                while i < 5:
                    writer = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system",
                             "content": ''' 
                                                     You are given a list of sentences.
                                                     If a statement dosen't make an argument give N.
                                                     For each sentence return the most applicable letter: P if an argument is made with pathos, E if an argument is made with ethos, L if an argument is made with logos.
                                                     Pathos is any instance where emotions are is appealed to in an argument.
                                                     Ethos is any instance where the credibility of the author or an external character is appealed to in the argument.
                                                     Logos is any instance where logical connections between ideas, such as facts, statistics, history, or analogies used in the argument.
                                                    '''},
                            {"role": "user",
                             "content": f'''
                                                     TUPLES:{arts}
                                                     For the sentence in each tuple return a P, E, or L, that best corresponds with that data.
                                                     If the sentence is only stating facts, give the letter N.
                                                     Here is an example of how to format the data [P,E, ...]  '''}
                        ]
                    )

                    lst_str = writer.choices[0].message.content
                    # Remove square brackets and split by commas
                    elements = lst_str[1:-1].split(',')

                    # Strip whitespace and convert to string
                    annotes = [element.strip() for element in elements]

                    annotations_.append(annotes)
                    i += 1
                annotations.append(annotations_)

            #parse through
            logic = []
            conf = []
            line_num = []
            for i in range(0,len(annotations)):
                df = pd.DataFrame(annotations[i])

                logic_ = []
                conf_ = []
                line_num_ =[]

                for iii in range(0,len(df.columns)):
                     # most frequent
                    let = df[iii].mode().values[0]
                    try:
                        num = df[iii].value_counts()[let]
                        logic_.append(let),conf_.append(num/5),line_num_.append(iii)
                    except:
                        pass
                logic.append(logic_),conf.append(conf_),line_num.append(line_num_)
        return logic, conf, line_num

    def add_annotations(self,articles,logic,conf,line):
        if len(articles.index) == 1:
            annotated_art = []
            art = articles['article'][0]
            tup = list(zip(logic, conf, line))
            m = [(x, y, z) for x, y, z in tup if y >= 0.8]
            for ii in range(0, len(m)):
                arts = art.split(". ")
                arts[m[ii][2]] = f"/{m[ii][0]}/" + arts[m[ii][2]] + f"/{m[ii][0]}/"
                art = '. '.join(arts)
            annotated_art.append(art)
            articles['article'] = annotated_art

        else:
            annotated_art = []
            for i in range(0, len(articles.index)):
                art = articles['article'][i]
                tup = list(zip(logic[i], conf[i], line[i]))
                m = [(x, y, z) for x, y, z in tup if y >= 0.8]
                for ii in range(0, len(m)):
                    arts = art.split(".")
                    arts[m[ii][2]] = f"/{m[ii][0]}/" + arts[m[ii][2]] + f"/{m[ii][0]}/"
                    art = '. '.join(arts)
                annotated_art.append(art)
            articles['article'] = annotated_art

        return articles


def check_lists_similarity(lists):
    if not lists:
        return False  # If no lists provided, return False

    for i in range(len(lists[0])):
        # Create a dictionary to count occurrences of each letter at position i
        letter_count = {}

        # Iterate through each list and count occurrences of the letter at position i
        for lst in lists:
            if i < len(lst):  # Ensure index is within range
                letter = lst[i]
                if letter in letter_count:
                    letter_count[letter] += 1
                else:
                    letter_count[letter] = 1

        # Check if any letter occurred at least 3 times at position i
        for count in letter_count.values():
            if count >= 3:
                return True  # At least 3 lists have the same letter at position i

    return False  # No position found where at least 3 lists have the same letter

class commit_db:
    '''
    Class for committing articles to the database
    '''
    def __init__(self, art_1,art_2):
        self.art_1 = art_1
        self.art_2 = art_2
        self.date_ = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        rank_ = all_it_from_ns('main_rank', 'live_rankings')
        try:
            self.rank_ = max(rank_)
        except:
            self.rank_ = 0

    def commit_art2(self):
        articles = self.art_2
        rank = self.rank_
        date_ = self.date_

        if articles.empty:
            pass

        elif len(articles.index) == 1:
            rnk_lst = rank + 1
            key_id = str(uuid.uuid4())
            df = articles_from_key(articles['key_ids'][0])
            old_date_ = pd.to_datetime(str(df['create_date'].values[0]))
            old_date = old_date_.strftime('%Y-%m-%d %H:%M:%S')

            # give the old articles a uuid and commit to old article table

            commit_old_articles(
                key_id=articles['key_ids'][0],
                title=df['title'].values[0],
                title_A=df['title_A'].values[0],
                title_B=df['title_B'].values[0],
                title_C=df['title_C'].values[0],
                body=repr(df['body'].values[0]),
                create_date=date_
            )



            replace_live_articles(
                key_id=articles['key_ids'][0],
                title=articles['TITLE'].values[0],
                title_A=articles['TITLE_A'].values[0],
                title_B=articles['TITLE_B'].values[0],
                title_C=articles['TITLE_C'].values[0],
                body=repr(articles['article'].values[0]),
                create_date= old_date,
                updated_date=date_)

            # Commit Rankings
            commit_rankings(key_id=articles['key_ids'][0],
                            main_rank=int(rnk_lst),
                            date=date_)

            # Commit Images
            for i in range(0, len(articles['img'])):
                commit_img(key_id=articles['key_ids'][0],
                           img=articles['img'][i],
                           date=date_)

            # Commit Categories
            for i in range(0, len(articles['CAT'])):
                commit_cat(key_id=articles['key_ids'][0],
                           cat=articles['CAT'][i][0],
                           date=date_)

            for i in range(0, len(articles['urls'])):
                commit_urls(key_id=articles['key_ids'][0],
                           urls=articles['urls'][i],
                           date=date_)

            print(f"1: Articles, on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        else:

            rnk_lst = list(range(rank, rank + len(articles.index)))
            articles['rnk'] = rnk_lst

            for i, r in articles.iterrows():
                df = articles_from_key(r['key_ids'][0])
                key_id = str(uuid.uuid4())

                if len(r['img']) == 1:
                    commit_img(key_id=df['key_id'][0],
                               img=r['img'],
                               date=date_)
                else:
                    for ii in range(0, len(r['img'])):
                        commit_img(key_id=df['key_id'][0],
                                   img=r['img'][ii],
                                   date=date_)
                if len(r['CAT']) == 1:
                    commit_cat(key_id=key_id,
                               cat=r['CAT'][0],
                               date=date_)
                else:
                    for ii in range(0, len(r['CAT'])):
                        commit_cat(key_id=df['key_id'][0],
                                   cat=r['CAT'][ii][0],
                                   date=date_)

                for ii in range(0, len(r['urls'])):
                    commit_urls(key_id=key_id,
                               url=r['urls'][ii][0],
                               date=date_)

                # Commit articles
                commit_live_articles(
                    key_id=df['key_id'][0],
                    title=r['TITLE'],
                    title_A=r['TITLE_A'],
                    title_B=r['TITLE_B'],
                    title_C=r['TITLE_C'],
                    body=repr(r['article']),
                    create_date=df['create_date'][0],
                    updated_date=date_)

                # Commit Rankings
                commit_rankings(key_id=df['key_id'][0],
                                main_rank=int(r['rnk']),
                                date=date_)

    def commit_art1(self):
        articles_1 = self.art_1
        rank_ = self.rank_
        date_ = self.date_

        if articles_1.empty:
            pass

        elif len(articles_1.index) == 1:
            rnk_lst = max(rank_) + 1
            key_id = str(uuid.uuid4())

            # Commit articles
            commit_live_articles(
                key_id=key_id,
                title= articles_1['GIST'].values[0],
                title_A=articles_1['TITLE_A'].values[0],
                title_B=articles_1['TITLE_B'].values[0],
                title_C=articles_1['TITLE_C'].values[0],
                body= repr(articles_1['article'].values[0]),
                create_date= date_,
                updated_date= date_)

            # Commit Rankings
            commit_rankings(key_id=key_id,
                               main_rank=int(rnk_lst),
                               date=date_)

            # Commit Images
            for i in range(0, len(articles_1['img'])):
                commit_img(key_id=key_id,
                              img = articles_1['img'][i],
                              date = date_)

            # Commit Categories
            for i in range(0, len(articles_1['CAT'])):
                commit_cat(key_id=key_id,
                              cat=articles_1['CAT'][i][0],
                              date=date_)

            for i in range(0, len(articles_1['urls'])):
                commit_urls(key_id=key_id,
                            url=articles_1['urls'][i][0],
                            date=date_)


            print(f"1: Articles, on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        else:

            rnk_lst = list(range(rank_, rank_ + len(articles_1.index)))
            articles_1['rnk'] = rnk_lst

            for i, r in articles_1.iterrows():
                key_id = str(uuid.uuid4())


                if len(r['imgs']) == 1:
                    commit_img(key_id=key_id,
                                  img=r['imgs'][0],
                                  date=date_)
                else:
                    for ii in range(0, len(r['imgs'])):
                        commit_img(key_id=key_id,
                                      img=r['imgs'][ii],
                                      date=date_)

                if len(r['CAT']) == 1:
                    commit_cat(key_id=key_id,
                                  cat=r['CAT'][0],
                                  date=date_)
                else:
                    for ii in range(0, len(r['CAT'])):
                        commit_cat(key_id=key_id,
                                      cat=r['CAT'][ii],
                                      date=date_)

                for ii in range(0, len(r['urls'])):
                    commit_urls(key_id=key_id,
                               url=r['urls'][ii],
                               date=date_)

                # Commit articles
                commit_live_articles(
                    key_id=key_id,
                    title=r['GIST'],
                    title_A=r['TITLE_A'],
                    title_B=r['TITLE_B'],
                    title_C=r['TITLE_C'],
                    body=repr(r['article']),
                    create_date=date_,
                    updated_date=date_)

                # Commit Rankings
                commit_rankings(key_id=key_id,
                                   main_rank=int(r['rnk']),
                                   date=date_)

            print(f"{str(len(articles_1.index))}: Articles, on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")










