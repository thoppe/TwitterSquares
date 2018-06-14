from scrape_utils import auth, jprint
from tweepy.streaming import StreamListener
from tweepy import Stream
import tweepy
import json, os, codecs
from tqdm import tqdm
import requests
import shutil
import sys

max_total = 180
word = sys.argv[1]

save_dest = 'data/users/{}'.format(word)
os.system('mkdir -p "{}"'.format(save_dest))

save_dest2 = 'data/profile_image/{}'.format(word)
os.system('mkdir -p "{}"'.format(save_dest2))

progress = tqdm(total=max_total)
api = tweepy.API(auth)
search_results = api.search(q=word, lang='en', count=10)
count = 0

for tweet in tweepy.Cursor(api.search,
                           q=word,
                           rpp=100,
                           result_type="recent",
                           include_entities=True,
                           lang="en").items():
    js = tweet._json

    if "user" not in js:
        print ("MISSING USER")
        continue

    user = js["user"]
    user_id = user["id_str"]

    f_save = os.path.join(save_dest, user_id)
        
    if os.path.exists(f_save):
        continue

    key = "profile_image_url"
    if key not in user:
        continue
    url = user[key]

    try:
        assert('_normal.' in url)
    except:
        print ("VERY WEIRD URL", url)
        continue
        
    url = url.replace('_normal.', '.')
        
    try:
        r = requests.get(url, stream=True)
    except requests.exceptions.SSLError:
        print ("SSL Error, sleeping for 10")
        time.sleep(5)
        continue
        
    f_save2 = os.path.join(save_dest2, user_id)
    with open(f_save2, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)

    with codecs.open(f_save, 'w', 'utf-8') as FOUT:
        FOUT.write( json.dumps(user, indent=2) )

    jprint(user)
    progress.update()
    
    count += 1
    if count >= max_total: break
        
        
progress.close()

