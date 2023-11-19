import pandas as pd
import base64
from requests import post, get
import json
import os
import pandas as pd
import pickle

client_id = 'aea59cdce77243ccb7f7e119268fc048'
client_secret = '46952a489d3c4860bcc4fe07cc579273'

def get_token():
  auth_string = client_id + ":" + client_secret
  auth_bytes = auth_string.encode("utf-8")
  auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

  url = "https://accounts.spotify.com/api/token"
  headers = {
      "Authorization": "Basic " + auth_base64,
      "Content-Type": "application/x-www-form-urlencoded"
  }
  data = {"grant_type": "client_credentials"}
  result = post(url, headers=headers, data=data)
  json_result = json.loads(result.content)
  token = json_result["access_token"]
  return token

def get_auth_header(token):
  return {"Authorization": "Bearer " + token}

token = get_token()

# USER
def get_user(token):
  url = "https://api.spotify.com/v1/users/7hfnwjm84q9p2zofkvgj1fk13/playlists"
  headers = get_auth_header(token)

  result = get(url, headers=headers)
  json_result = json.loads(result.content)["items"][12]["tracks"]['href']
  return json_result

# USER TRACKS
def get_user_tracks(token):
  url = get_user(token)
  headers = get_auth_header(token)

  result = get(url, headers=headers)
  json_result = json.loads(result.content)["items"]
  return json_result

# SONG DATA
def get_song_data(token, idx):
  items = get_user_tracks(token)

  song_uri = items[idx]["track"]["uri"]
  song_id = song_uri[14:]

  url = "https://api.spotify.com/v1/audio-features/" + song_id
  headers = get_auth_header(token)

  result = get(url, headers=headers)
  json_result = json.loads(result.content)
  return json_result

def get_album(token, idx):
  items = get_user_tracks(token)

  song_uri = items[idx]["track"]["uri"]
  song_id = song_uri[14:]

  url = "https://api.spotify.com/v1/tracks/" + song_id
  headers = get_auth_header(token)

  result = get(url, headers=headers)
  json_result = json.loads(result.content)["album"]["images"][0]["url"]
  return json_result

artist = []
name = []
acoustic = []
dance = []
energy = []
instrumental = []
liveness = []
loudness = []
speech = []
tempo = []
valence = []
popularity = []
scores = []
album = []


items = get_user_tracks(token)
for idx, item in enumerate(items):
  data = get_song_data(token, idx)

  artist.append(items[idx]["track"]["artists"][0]["name"])
  name.append(items[idx]["track"]["name"])
  acoustic.append(data["acousticness"])
  dance.append(data["danceability"])
  energy.append(data["energy"])
  instrumental.append(data["instrumentalness"])
  liveness.append(data["liveness"])
  loudness.append(data["loudness"])
  speech.append(data["speechiness"])
  tempo.append(data["tempo"])
  valence.append(data["valence"])
  popularity.append(items[idx]["track"]["popularity"])
  album.append(get_album(token, idx))

  tempoScore = (data["tempo"] - 60) / 140 * 0.4
  energyScore = data["energy"] * .3
  loudScore = (data["loudness"] + 15) / 15 * 0.3

  scores.append(tempoScore + energyScore + loudScore)

out = pd.DataFrame({'SCORE':scores,
                    'artist':artist,
                    'name':name,
                    'dance':dance,
                    'acoustic':acoustic,
                    'energy':energy,
                    'instrumental':instrumental,
                    'liveness':liveness,
                    'loudness':loudness,
                    'speech':speech,
                    'tempo':tempo,
                    'valence':valence,
                    'popularity':popularity,
                    'album':album

})

def get_song_data(token, idx):
  items = get_user_tracks(token)

  song_uri = items[idx]["track"]["uri"]
  song_id = song_uri[14:]

  url = "https://api.spotify.com/v1/audio-features/" + song_id
  headers = get_auth_header(token)

  result = get(url, headers=headers)
  json_result = json.loads(result.content)
  return json_result

paths = ["Mr. Brightside", "Nicki Minaj - Starships (Explicit)", "Outkast - Hey Ya! (Official HD Video)", "Shakira - Hips Don't Lie (Official 4K Video) ft. Wyclef Jean", "DNCE - Cake By The Ocean", "Mark Ronson - Uptown Funk (Official Video) ft. Bruno Mars", "MACKLEMORE & RYAN LEWIS - THRIFT SHOP FEAT. WANZ (OFFICIAL VIDEO)", "Taio Cruz - Dynamite (Official UK Version)", "Spice Girls - Wannabe (Official Music Video)", "Pitbull - Timber (Official Video) ft. Ke$ha", "WALK THE MOON - Shut Up and Dance (Official Video)", "Soulja Boy Tell'em - Crank That (Soulja Boy) (Official Music Video)", "Ke$ha - TiK ToK (Official HD Video)", "Icona Pop - I Love It (feat. Charli XCX) [OFFICIAL VIDEO]", "Cee Lo Green - Forget You", "Beyoncé - Crazy In Love ft. JAY Z", "Pitbull - Fireball ft. John Ryan", "Flo Rida - Right Round (feat. Ke$ha) [US Version] (Official Video)", "Flo Rida - My House [Official Video]", "Taylor Swift - 22 (Taylor's Version) (Lyric Video)", "Rihanna - Pon de Replay (Internet Version)", "Rihanna - Umbrella (Orange Version) (Official Music Video) ft. JAY-Z", "Lizzo - Truth Hurts (Official Video)", "Flo Rida - Good Feeling [Official Video]", "Ke$ha - Die Young (Official Video)"]
danceScores = list(zip(out.name[:25] + " -> " + out.artist[:25], out.dance[:25], paths, out.album[:25]))
print(danceScores)

file = open('songs_data.pkl', 'wb')
pickle.dump(danceScores, file)
file.close()