import os
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import configparser
from tweepy.streaming import StreamListener

import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

# Read Credentials
Config = configparser.ConfigParser()
assert(os.path.exists("credentials.ini"))
Config.read("credentials.ini")
consumer_key = Config.get("TwitterCredentials","consumer_key")
consumer_secret = Config.get("TwitterCredentials","consumer_secret")
access_token = Config.get("TwitterCredentials","access_token")
access_token_secret = Config.get("TwitterCredentials","access_token_secret")


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(
    auth,
    wait_on_rate_limit=False,
    wait_on_rate_limit_notify=True,
    compression=True,
)

def jprint(js):
    json_str = json.dumps(js, indent=4, sort_keys=True)
    print(highlight(json_str, JsonLexer(), TerminalFormatter()).strip())

def lookup_userIDs(user_ids):
    data = api.lookup_users(user_ids=user_ids)
    users = [x._json for x in data]
    return users

def get_followers(user_id):
    cursor = tweepy.Cursor(api.followers_ids, id = user_id)
    follower_ids = []
    for k,page in enumerate(cursor.pages()):
        follower_ids.extend(page)
        if k>5: break
        
    return follower_ids

def get_rate_limits():
    return api.rate_limit_status()['resources']

class Listener(StreamListener):

    def on_data(self, data):
        js = json.loads(data)
        jprint(js)
        return(True)

    def on_error(self, status):
        print ("Streaming exited with error", status)

        
#help(api.lookup_users)
#data = api.rate_limit_status()
#print data['resources']['users']['/users/lookup']

if __name__ == "__main__":

    twitterStream = Stream(auth, Listener())
    #help(twitterStream.sample)
    twitterStream.sample(languages=['en'])
    exit()
    
    
    metasemantic_ID = 3249034201
    jb_ID = 70802518
    data = lookup_userIDs([metasemantic_ID,jb_ID])
    jprint(data)

    data = get_followers(metasemantic_ID)
    jprint(data)
