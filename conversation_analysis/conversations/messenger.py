# -*- coding: utf-8 -*-
"""
This module implements Facebook Messenger-specific functionality for browsing
and loading conversations from archived files.

It exposes the following functionality:
    - <list_messenger_conversations>
      A function to list the conversations contained in archives files.
    - <MessengerConversation>
      A subclass of <conversation.Conversation> implementing Messenger-specific
      methods to load a conversation from JSON and HTML files, and extend an
      existing one with additional files.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Self
from unicodedata import normalize

import numpy as np
from bs4 import BeautifulSoup, Tag

from ..utils import PathObj
from .conversation import Message, Conversation



HTML_MSG_ELTS = {
    'message': ['div', {'class': '_a6-g'}],
    #
    'participants': ['div', {'class': '_2ph_ _a6-h'}],
    #
    'sender': ['div', {'class': '_2ph_ _a6-h _a6-i'}],
    'timestamp': ['div', {'class': '_a72d'}],
    #
    'content': ['div', {'class': '_2ph_ _a6-p'}],
    'reactions': ['ul', {'class': '_a6-q'}]
    }

HTML_DATE_FORMAT = {
    'en_US': "%b %d, %Y %I:%M:%S%p",
    'fr_FR': "%d/%m/%Y, %H:%M",
    }


# =============================================================================
# Functions
# =============================================================================

INBOX_DIRS = {'inbox', 'archived_threads', 'ee2e_cutovers',
              'filtered_threads', 'messages_requests'}


def list_messenger_conversations(
        dir_path: PathObj,
        return_relative_path: bool = False)-> dict[str, list[str]]:
    """
    Return the distinct conversations contained in `dir_path` and the paths
    to directories containing the data.

    This is most useful if the conversations span over multiple archives, for
    instance when creating archive files each year.

    This function browses both JSON and HTML archives.

    Parameters
    ----------
    dir_path : PathObj
        Path to the directory containing the archives.
    return_relative_path : bool, optional
        If True, return the list of paths relative to dir_path.
        The default is False.

    Returns
    -------
    conversations : dict[str, list[str]]
        conversation name -> paths to conversation data

    """
    conversations = {}
    name_map = {}

    for root, dirs, files in os.walk(dir_path):
        path = Path(root).resolve()
        if path.stem in INBOX_DIRS: # converation dirs are in INBOX_DIRS
            for d in dirs:
                try: # conv dirs usually have the form <name>_<id>
                    d1, d2 = d.split('_', 1) # d = <name>_<id>
                except ValueError: # but not always
                    d1 = d2 = d # d = id
                if d1 not in name_map and d2 not in name_map: # new conversation
                    name_map[d1] = name_map[d2] = d
                    conversations[d] = [(path / d).as_posix()]
                elif d1 not in name_map: # name as changed, id is known
                    name_map[d1] = name_map[d2]
                    conversations[name_map[d2]].append((path / d).as_posix())
                elif d2 not in name_map: # name is known, id has changed
                    name_map[d2] = name_map[d1]
                    conversations[name_map[d1]].append((path / d).as_posix())
                else: # name is known, id is known
                    conversations[name_map[d1]].append((path / d).as_posix())

    if return_relative_path:
        for name, paths in conversations.items():
            paths = [Path(p).relative_to(dir_path).as_posix() for p in paths]
            conversations[name] = paths

    return conversations


# =============================================================================
#
# =============================================================================

def _resolved_uri(message: dict, resolved_path: Path)-> dict:
    """
    Modify the relative URLs of `uri` entries of message items to inclute full
    path resolution of `resolved_path`.
    """
    stem = resolved_path.stem
    for k, v in message.items():
        if isinstance(v, list):
            for item in v:
                try:
                    uri = item['uri']
                except KeyError:
                    break
                else:
                    # Do not change uri if it does not extend path
                    if uri.find(stem) == -1:
                        break
                    # Extend path with uri
                    uri = uri.split(stem)[-1][1:]
                    item['uri'] = (resolved_path / uri).as_posix()
    return message


##### JSON-related #####

def _load_json(file: PathObj)-> dict:
    """
    Load a Facebook Messenger JSON file archive.
    """
    with open(file, 'rb') as f:
        f = f.read()
        # Expand escape chars that do not correspond to unicode escape
        f = re.sub(rb'\\\\', rb'\\\\\\\\', f)
        f = re.sub(rb'\\"', rb'\\\\"', f)
        # Convert unicode escaped to unicode chars
        f = f.decode('unicode_escape')
        f = f.encode('latin1').decode('utf-8')
        # Normalize characters
        f = normalize('NFC', f)
    return json.loads(f, strict=False)


def _adapt_keys(message: dict)-> dict:
    """
    Adapt the messages' keys from loaded JSON file to match parameter names of
    <conversation.Message>.
    """
    # Adapth main keys
    message['sender'] = message.pop('sender_name')
    message['timestamp'] = message.pop('timestamp_ms') * 1e-3 # convert to seconds
    if 'audio_files' in message:
        message['audio'] = message.pop('audio_files')
    if 'share' in message:
        message['shared'] = message.pop('share')
    # Adapt item keys
    for k, v in message.items():
        if isinstance(v, list):
            for item in v:
                try:
                    item['timestamp'] = item.pop('creation_timestamp')
                except KeyError:
                    break
        elif isinstance(v, dict): # shared
            try:
                v['text'] = v.pop('share_text')
            except KeyError:
                pass

    return message


##### HTML-related #####

def _load_html(file: PathObj)-> list:
    """
    Load a Facebook Messenger HTML file archive.
    """
    with open(file, 'rt', encoding='utf-8') as f:
        soup = BeautifulSoup(f, "html.parser").find('div', role='main')
    return soup.contents


def _parse_raw_messages(raw_messages: list[Tag],
                        locale: str)-> tuple[list[dict], set[str]]:
    """
    Parse a list of raw messages into a list of dict suitable for Conversation
    instanciation.

    Currently, some message entries are ignored:
        - all the media and shared links

    Parameters
    ----------
    raw_messages : list[Tag]
        A list of HTML tags parsed with Beautifulsoup, each referring to a
        distinct message.
    locale : str
        Locale code of the whatsapp application from which the archive was
        created. The code is in the form language-COUNTRY, eg en-US, de-DE.

    Returns
    -------
    messages : list[dict]
        The messages as a list of dicts suitable for instanciation of
        <conversation.Message> objects.
    participants : set[str]
        The participants of the conversation.

    """
    date_fmt = HTML_DATE_FORMAT[locale]

    ## set participants
    p = raw_messages[0].find(*HTML_MSG_ELTS['participants']).text
    p = p.split(': ')[1].split(', ')
    participants = set(p[:-1] + p[-1].split(' and '))
    ## parse messages
    messages = []
    for m in raw_messages:
        if (ts:=m.find(*HTML_MSG_ELTS['timestamp'])) is None:
            continue
        sender = m.find(*HTML_MSG_ELTS['sender']).text
        timestamp = datetime.strptime(ts.text, date_fmt).timestamp()
        message = {'sender': sender,
                   'timestamp': timestamp}

        raw_contents = m.find(*HTML_MSG_ELTS['content']).contents[0].contents
        content = raw_contents[1].text
        message['content'] = content

        reactions = m.find(*HTML_MSG_ELTS['reactions'])
        if reactions is not None:
            reactions = [r.text for r in reactions]
            message['reactions'] = [{r[1:]: r[0]} for r in reactions]

        messages.append(message)

    return messages, participants




# =============================================================================
# MessengerConversation class
# =============================================================================

class MessengerConversation(Conversation):
    """
    Subclass of <Conversation>. Adds functionality specific to Facebook
    Messenger.

    - set_default_participant(name, condition) -> None
        set the default participant name
    """

    @classmethod
    def from_facebook_json(cls,
                           dir_paths: list[PathObj],
                           resolve_uri: bool = True)-> Self:
        """
        Return a MessengerConversation instance from a list of paths to
        facebook archive conversations in JSON format.

        The conversation might be split in multiple files, for instance
        when archiving on a yearly basis.
        Duplicate messages from overlapping archiving periods are removed, and
        the messages are sorted.

        The message entries have the following keys:
            - "sender_name"
            - "timestamp_ms"
            - "content"
            - "reactions"
            - "photos"
            - "videos"
            - "audio_files"
            - "files"
            - "share"
            - "gifs"
        Some keys present in message entries are ignored:
            - "users"
            - "is_geoblocked_for_viwer"
            - "is_unsent"
            - "call_duration"
            - "sticker"
            - "type"

        Parameters
        ----------
        dir_paths : list[PathObj]
            List of directories containing the conversation files.
            Multiple directory paths are allowed because the conversation
            might be split in multiple files, for instance when archiving on
            a yearly basis.
        resolve_uri : bool, optional
            If True, resolve the uri paths of the media. The default is True.

        Returns
        -------
        MessengerConversation
            The assembled conversation.

        """
        dir_paths = [Path(path).resolve() for path in dir_paths]
        messages = []
        participants = set()
        name = ''
        title = ''
        # keys = set()

        for path in dir_paths:
            root, dirs, files = next(os.walk(path))
            msg_files = [path / f for f in files
                         if f.split('_')[0] == "message"]
            for file in msg_files:
                # Load messages JSON file
                msgs = _load_json(file)
                # Update participants from each file
                participants |= {next(iter(p.values()))
                                 for p in msgs['participants']}
                # Add parsed messages from the file, resolve media uri
                # keys |= set.union(*[set(m.keys()) for m in msgs['messages']])
                if resolve_uri:
                    messages += [Message(**_resolved_uri(_adapt_keys(m), path))
                                 for m in msgs['messages']]
                else:
                    messages += [Message(**_adapt_keys(m))
                                 for m in msgs['messages']]
            # Set name and title from the first file of the list
            name = name if name else path.stem
            title = title if title else msgs['title']
        # Remove duplicates and sort messages by post date
        messages = np.unique(messages).tolist()
        # Participants that deleted their account appear as 'sender': ""
        if messages:
            participants.update(p for m in messages for p in m.participants)

        # print(keys)

        return cls(messages, participants, name, title)


    @classmethod
    def from_facebook_html(cls,
                           dir_paths: list[PathObj],
                           locale: str,
                           resolve_uri: bool = True)-> Self:
        """
        Load a MessengerConversation from a list of paths to facebook archive
        conversations in HTML format.

        The resulting conversation messages are sorted by post date.
        Duplicate messages from overlapping archiving periods are removed, and
        the messages are sorted.

        Currently, some message entries are ignored:
            - all the media and shared links

        Parameters
        ----------
        dir_paths : list[PathObj]
            List of directories containing the conversation files.
            Multiple directory paths are allowed because the conversation
            might be split in multiple files, for instance when archiving on
            a yearly basis.
        locale : str
            Locale code of the whatsapp application from which the archive was
            created. The code is in the form language-COUNTRY, eg en-US, de-DE.
        resolve_uri : bool, optional
            If True, resolve the uri paths of the media. The default is True.

        """
        dir_paths = [Path(path).resolve() for path in dir_paths]
        messages = []
        participants = set()
        name = ''
        title = ''

        for path in dir_paths:
            root, dirs, files = next(os.walk(path))
            msg_files = [path / f for f in files
                         if f.split('_')[0] == "message"]
            for file in msg_files:
                raw_msgs = _load_html(file)

                msgs, part = _parse_raw_messages(raw_msgs, locale)

                # Add parsed messages from the file, resolve media uri
                if resolve_uri:
                    messages += [Message(**_resolved_uri(m, path))
                                 for m in msgs]
                else:
                    messages += [Message(**m) for m in msgs]

                participants |= part

        messages = np.unique(messages).tolist()
        return cls(messages, participants, name, title)


    def extend(self, dir_paths: list[PathObj])-> None:
        """
        TODO
        Extend an existing conversation by appending archives.

        Parameters
        ----------
        dir_paths : list[PathObj]
            List of directories containing the conversation files.

        """
        raise NotImplementedError

