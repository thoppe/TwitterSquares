from scrape_utils import auth, jprint
from tweepy.streaming import StreamListener
from tweepy import Stream
import json, os, codecs
from tqdm import tqdm
import requests
import shutil
import sys

word = sys.argv[1]

save_dest = 'data/users/{}'.format(word)
os.system('mkdir -p {}'.format(save_dest))

save_dest2 = 'data/profile_image/{}'.format(word)
os.system('mkdir -p {}'.format(save_dest2))

progress = tqdm()

class Listener(StreamListener):

    def on_data(self, data):
        js = json.loads(data)

        if "user" not in js:
            return True

        user = js["user"]
        user_id = user["id_str"]

        f_save = os.path.join(save_dest, user_id)
        
        if os.path.exists(f_save):
            return True

        key = "profile_image_url"
        if key not in user:
            return True
        url = user[key]

        try:
            assert('_normal.' in url)
        except:
            print ("VERY WEIRD URL", url)
            return True
        
        url = url.replace('_normal.', '.')
        
        try:
            r = requests.get(url, stream=True)
        except requests.exceptions.SSLError:
            print ("SSL Error, sleeping for 10")
            time.sleep(5)
            return True
        
        f_save2 = os.path.join(save_dest2, user_id)
        with open(f_save2, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)

        with codecs.open(f_save, 'w', 'utf-8') as FOUT:
            FOUT.write( json.dumps(user, indent=2) )

        jprint(user)
        progress.update()
        
        
        return(True)

    def on_error(self, status):
        print ("Streaming exited with error", status)

        
twitterStream = Stream(auth, Listener())
twitterStream.filter(languages=['en'],track=[word,])
