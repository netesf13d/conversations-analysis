# -*- coding: utf-8 -*-
"""
Script to make a dummy conversation out of a genuine one.
"""

import os
import json
from pathlib import Path

import numpy as np


PathObj = Path | str

# module conversations in parent directory
pdir = Path(__file__).resolve().parents[1]
rng = np.random.default_rng()


# =============================================================================
# Constants
# =============================================================================

MEDIA_LIST_ENTRIES = (
    'photos', 'videos', 'audio_files', 'files', 'share', 'gifs',
    )

MEDIA_DICT_ENTRIES = ('sticker',)

PARTICIPANTS = [
    "Grace Hopper",
    "Charles Babbage",
    "Edsger Dijkstra",
    "Alonzo Church",
    "Haskell Curry",
    "Ada lovelace",
    "Alan Turing",
    "Claude Shannon",
    "John von Neumann",
    "Frances Elizabeth Allen",
    "Lynn Conway",
    "Stephen Cole Kleene",
    "Ida Rhodes",
    "Dennis Richie",
    "Al-Jazari",
    "Joseph-Marie Jacquard",
    "George Boole",
    ]


WORD_FREQ_FILE = pdir / "scripts/lemmas_60k.txt"
WORDS = np.loadtxt(WORD_FREQ_FILE, dtype=str, delimiter='\t',
                   skiprows=9, usecols=(1,))
FREQS = np.loadtxt(WORD_FREQ_FILE, dtype=float, delimiter='\t',
                   skiprows=9, usecols=(3,))
FREQS = FREQS / np.sum(FREQS)


# =============================================================================
# Functions
# =============================================================================

def make_content(word_list: list[str],
                 prob_distrib: np.ndarray,
                 nb_words: int)-> str:
    """
    Make a random chain of `nb_words` words drawn from `word_list` according
    to `prob_distrib`.
    """
    words = rng.choice(word_list, nb_words, p=prob_distrib)
    return ' '.join(words)


def edit_media(messages: list[dict],
               media_name: str,
               media_fpath: str,
               media_fnames: list[str])-> None:
    """
    In-place modification of `media_name` entries in the list of messages.
    """
    for msg in messages:
        if media_name in msg:
            ts = msg['timestamp_ms'] // 1000
            for media in msg[media_name]:
                media['creation_timestamp'] = ts
                media['uri'] = media_fpath + '/' + rng.choice(media_fnames)


def make_dummy_conversation(conversation: dict,
                            participants_map: dict[str, str],
                            conv_entries: dict,
                            media_entries: dict,
                            timestamp_blur: float = 3600*1000)-> None:
    """
    In-place modify a messenger conversation to anonymize participants, media,
    post date, etc.
    
    The modifications are:
        - Participants, accoding to a map str -> str
        - Main conversation entries: title, thread_path, image
        - Media entries in messages, by selecting a random filename in a list
        - Post date, by adding a gaussian noise
        

    Parameters
    ----------
    conversation : dict
        Conversation dict loaded from a json file.
    participants_map : dict[str, str]
        Mapping: original participant -> dummy participant.
    conv_entries : dict
        First level conversation entries to change.
        conv[entry] <- conv_entries[entry]
        Entries can be:
            - 'title': str, the conversation's title
            - 'thread_path', str
            - 'image', dict {'uri': str, 'creation_timestamp': int}
    media_entries : dict
        Media entries {'photos', 'videos', 'audio', 'files'} to modify.
        Entries not specified are suppressed from the conversation, that is,
        if 'videos' is not in media_entries, the modified conversation
        messages will not have 'videos' entries.
        The format is entry -> dict {'path': str, 'names': list[str]}
        For each message containing a given entry, a random filename is
        selected and the media's URI is set to path + filename.
    timestamp_blur : float, optional
        Variance of a gaussian noise added to the original messages timestamps,
        in ms.
        The default is 3600*1000, 1 hour.

    """
    ## new participants
    conversation['participants'] = [{'name': participants_map[p]}
                                    for part in conv['participants']
                                    for _, p in part.items()]
    ## Other conversation entries
    for k, v in conv_entries.items():
        if isinstance(v, str):
            conversation[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                conversation[k][kk] = vv
    ## Modify messages
    for msg in conversation['messages']:
        # Blur post date
        msg['timestamp_ms'] += int(rng.normal(0, timestamp_blur))
        # Map participant
        msg['sender_name'] = participants_map[msg['sender_name']]
        # Map reactions actors
        if 'reactions' in msg:
            for r in msg['reactions']:
                r['actor'] = participants_map[r['actor']]
        # Map users
        if 'users' in msg:
            for u in msg['users']:
                u['name'] = participants_map[u['name']]
        # Change content
        if 'content' in msg:
            msg['content'] = make_content(
                WORDS, FREQS, len(msg['content'].split(' ')))
    ## modify media
    for media in MEDIA_LIST_ENTRIES:
        if media in media_entries:
            edit_media(conversation['messages'], media,
                       media_entries[media]['path'],
                       media_entries[media]['names'])
        else:
            for msg in conversation['messages']:
                msg.pop(media, None)
    
    for media in MEDIA_DICT_ENTRIES:
        if media in media_entries: # not implemented
            pass
        else:
            for msg in conversation['messages']:
                msg.pop(media, None)
    ## Sort themessages according to timestamp
    conv['messages'].sort(key=lambda x: x['timestamp_ms'], reverse=True)
            


# =============================================================================
# Conversation modification parameters
# =============================================================================

conv1_entries = {
    'thread_path': "inbox/conversation1_bhmqhe56cg",
    'title': "Messenger conversation 1",
    'image': {'uri': "messages/photos/35290677_10844307674177198_122556601_n_1083573017619774.jpg"},
    }
conv2_entries = {
    'thread_path': "inbox/conversation2_kaopxbamlx",
    'title': "Messenger conversation 2",
    }

media11_entries = {
    'photos': {
        'path': "messages/inbox/conversation1_bhmqhe56cg/photos",
        'names': [
            "19904520_10208127587885633_1713364929_n_18308127587885633.jpg",
            "11977269_10153620340532840_1968679088_n_25153625540532800.jpg",
            ]
        },
    }
media12_entries = {
    'photos': {
        'path': "messages/inbox/conversation1_bhmqhe56cg/photos",
        'names': [
            "59009878_10153320063304375_1557865147_n_40153600763304375.jpg",
            "224788901_2362450767254173_8587861743830743026_n_2862457170580507.jpg",
            "323983958_476302114268703_2686348971644145406_n_476302171635310.jpg",
            "436242145_708685903681168_5751970373501785447_n_771685906347885.jpg",
            "880260085_1271944218008755_4218874397365404007_n_1271984410344089.jpg",
            ]
        },
    }
media21_entries = {
    'photos': {
        'path': "messages/inbox/conversation1_bhmqhe56cg/photos",
        'names': [
            "164041344_670649500964262_1048848627152310104_n_249269161329607.jpg",
            ]
        },
    }


dir_path = pdir / "examples/example_archive/"
conv_paths = [
    "facebook-yourname31415927_1/messages/inbox/conversation1_bhmqhe56cg",
    "facebook-yourname31415927_2/messages/inbox/conversation1_bhmqhe56cg",
    "facebook-yourname31415927_1/messages/inbox/conversation2_kaopxbamlx",
    ]
conv_entries = [conv1_entries, conv1_entries, conv2_entries]
media_entries = [media11_entries, media12_entries, media21_entries]


# =============================================================================
# Script
# =============================================================================

for path, c_entries, m_entries in zip(conv_paths, conv_entries, media_entries):
    root, dirs, files = next(os.walk(dir_path / path))
    msg_files = [dir_path / path / f for f in files
                 if f.split('_')[0] == "message"]
    
    for file in msg_files:
        with open(file, 'rb') as f:
            conv = json.load(f)

        it = iter(PARTICIPANTS)
        participants_map = {p: next(it)
                            for part in conv['participants']
                            for _, p in part.items()}
        participants_map['Facebook User'] = 'Facebook User'
        make_dummy_conversation(conv, participants_map, c_entries, m_entries)
        
        mod_file = file.parent / ('mod_' + file.name)
        with open(mod_file, 'wt', encoding='utf-8') as f:
            json.dump(conv, f, indent=2, ensure_ascii=True)
        

