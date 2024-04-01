'''
Polypinion Newstream

Description: Old method that works with the current flask API.
Data source is news api. It then feeds to gpt to create articles.
It calls the database and uses gpt to evaluate if articles are similar
enough to append in the database or not.

It's pretty buggy. Mainly GPT responses will give json that fails.

Moving to Bing API is a better method. I wasn't able to create the database
request/appends in mainV2.

We were working quickly to create multiple features (deduping, different titles etc.)
that tech debt wasn't taken care of. I apologize.
'''


from polypinion import newsapi, gbt_eval, gbt_author, commit_db
import datetime as dt
from inter_sql import all_it_from_ns, commit_error
import time
import sys
import json


while(True):
    # Date and time for each iteration.
    try:
        # Instantiate with sources
        sources = ['bloomberg', 'associated-press', 'fox-news',
                   'cnn', 'nbc-news', 'reuters', 'the-washington-post']

        # Call/receive stories
        news = newsapi(sources)
        ts = news.top_stories()

        #parse data
        desc, headline, url, content, img = ts[0], ts[1], ts[2], ts[3], ts[4]

        # Append news article database
        print(f"news api successful: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()

        evl = gbt_eval(desc=desc, headline=headline, url_=url, content=content)
        event_clusters = evl.cluster_events()

        # Append cluster database, assign each cluster a UUID
        print(f"Cluster successful: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()

        # Ask the writer to prepare the dataframe
        writer = gbt_author(event_clusters)
        jso = writer.to_json()
        df_clstr = writer.parser(jso)

        clustr_appends = writer.append_clusters(df_clstr['TITLE'].tolist(),df_clstr['DESC'].tolist())

        json_object = clustr_appends.choices[0].message.content
        json_object_cln = json_object.replace("['", '["').replace("']", '"]').replace("\\xa0", "").replace("',",'",').replace( ", '", ', "').replace("', '", '", "').replace(', "You',", 'You")
        json_object = json.loads(json_object_cln)

        title_a,title_b,title_c,cat= [],[],[],[]
        for i in range(0,len(json_object['out'])):
            title_a.append(json_object['out'][i]['TITLE_A'])
            title_b.append(json_object['out'][i]['TITLE_B'])
            title_c.append(json_object['out'][i]['TITLE_C'])
            cat.append(json_object['out'][i]['CAT'])

        # Create the articles

        df_clstr['TITLE_A'],df_clstr['TITLE_B'],df_clstr['TITLE_C'],df_clstr['CAT'] = title_a,title_b,title_c,cat
        df_clstr['img'] = writer.get_art_data(img, desc, df_clstr)


        # Ask the
        articles_1 = writer.writer_new(url, desc, img, df_clstr)
        print(f"Writer1 successful: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        articles_2 = writer.writer_existing(url, desc, img, df_clstr)
        print(f"Writer2 successful: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()


        logic, conf, line = writer.annotator(articles_1)
        art1_amd = writer.add_annotations(articles_1, logic, conf, line)


        logic,conf,line = writer.annotator(articles_2)
        art2_amd = writer.add_annotations(articles_2,logic,conf,line)

        articles_1['article'] = art1_amd['article']
        m = articles_1[articles_1['TITLE']=='Chinese Panda and Hacking']

        # Append SQLITE3 Database
        c_db = commit_db(art_1 = art1_amd,art_2 = articles_2)

        c_db.commit_art1()
        c_db.commit_art2()

    except:
        commit_error("ERROR", dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("ERROR")
        sys.stdout.flush()

    # Wait 15 min before reiterating
    time.sleep(900)
