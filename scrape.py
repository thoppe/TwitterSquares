"""Scape data for Twitter Squares

Usage:
  scrape.py <term> <n_images>

Options:
  -h --help       Show this screen.
"""

from utils import auth, jprint
from tweepy.streaming import StreamListener
from tweepy import Stream
import tweepy
import json, os, codecs
from tqdm import tqdm
import requests
import shutil
import sys

from docopt import docopt
dargs = docopt(__doc__)

total_images = int(dargs["<n_images>"])
name = dargs["<term>"]

save_dest2 = 'data/profile_image/{}'.format(name)
os.system('mkdir -p "{}"'.format(save_dest2))

progress = tqdm(total=total_images)
api = tweepy.API(auth)
search_results = api.search(q=name, lang='en', count=10)
count = 0

for tweet in tweepy.Cursor(api.search,
                           q=name,
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

    f_save2 = os.path.join(save_dest2, user_id)   
    if os.path.exists(f_save2):
        continue


        
    try:
        r = requests.get(url, stream=True)
    except requests.exceptions.SSLError:
        print ("SSL Error, sleeping for 10")
        time.sleep(5)
        continue
        
    f_save2 = os.path.join(save_dest2, user_id)
    with open(f_save2, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)

    #with codecs.open(f_save, 'w', 'utf-8') as FOUT:
    #    FOUT.write( json.dumps(user, indent=2) )

    jprint(user)
    progress.update()
    
    count += 1
    if count >= total_images: break
        
        
progress.close()

