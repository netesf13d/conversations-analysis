# -*- coding: utf-8 -*-
"""
This module implements Whatsapp-specific functionality for loading
conversations from archived files.

It exposes the following functionality:
    - <WhatsappConversation>
      A subclass of <conversation.Conversation> implementing Whatsapp-specific
      methods to load a conversation from exported text files, and extend an
      existing one with additional files.

Whatsapp files archives are highly locale dependent. The datetime format along
with some token (indicating omitted or joined file) depend on the language of
the system in which the app is installed. Ths appropriate format/value may thus
be missing for your locale.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np

from ..utils import PathObj
from .conversation import Message, Conversation


# =============================================================================
#
# =============================================================================

MEDIA_PREFIXES = {'photos': ('IMG',),
                  'videos': ('VID',),
                  'audio': ('AUD', 'PTT'),
                  'gifs': ()}
MEDIA_TYPES = {ext: k for k, exts in MEDIA_PREFIXES.items() for ext in exts}


## Regex for date detection: the start of a message
DATE_REGEX = {
    'en-US': r'[0-3][0-9]/[01][0-9]/[0-9]{4}, [0-1][0-9]:[0-5][0-9] [aApP][mM]',
    'fr-FR': r'[0-3][0-9]/[01][0-9]/[0-9]{4}, [0-2][0-9]:[0-5][0-9]',
    }
DATE_REGEX = {k: re.compile(v) for k, v in DATE_REGEX.items()}


## strptime string
DATE_FORMAT = {
    'en-US': "%d/%m/%Y, %I:%M %p",
    'fr-FR': "%d/%m/%Y, %H:%M",
    }


## 'joined files' token
JOINED_FILE = {
    'en-US': '(joined file)',
    'fr-FR': '(fichier joint)'}


## 'media omitted' token
MEDIA_OMITTED = {
    'en-US': '<Media omitted>',
    'fr-FR': '<Médias omis>'}


## Regex to detect shared link
LINK_REGEX = re.compile(r'https?:\/\/[^\s]+')


# =============================================================================
# Functions
# =============================================================================


def _parse_raw_messages(raw_messages: list[tuple[float, str]],
                        locale: str)-> tuple[list[dict], set[str]]:
    """
    Parse a list of raw messages into a list of dict suitable for Conversation
    instanciation.

    Parameters
    ----------
    raw_messages : list[tuple[float, str]]
        A raw message has the form (timestamp, raw_text).
        The raw_text contains information about
           - the sender
           - a possible joined or omitted file
           - the message's content.
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
    joined_file = JOINED_FILE[locale]

    messages = []
    participants = set()
    for msg in raw_messages:
        try:
            sender, raw_content = msg[1].split(': ', 1)
        except ValueError:
            continue

        raw_content = raw_content.rstrip('\n')

        message = {'timestamp': msg[0],
                   'sender': sender[3:]}
        participants.add(sender[3:])

        ## Parse for the presence of a media
        rc = raw_content.split('\n', 1)
        if (file:=rc[0]).endswith(joined_file):
            content = rc[1] if len(rc) > 1 else ''
            # parse content as file
            filename = file.strip('\u200e').strip(' ').split(' ')[0]
            prefix = filename.split('-', 1)[0]
            try:
                media_type = MEDIA_TYPES[prefix]
            except KeyError:
                media_type = 'files'
            message[media_type] = [{'uri': filename, 'timestamp': msg[0]}]
        else:
            content = raw_content

        ## Parse for the presence of a link
        if (match_:=LINK_REGEX.match(content)):
            message['shared'] = {'link': match_.group(), 'text': ''}
            content = content[match_.end():]

        message['content'] = content
        messages.append(message)

    return messages, participants


def _resolved_uri(message: dict, resolved_path: Path)-> dict:
    """
    Modify the relative URLs of `uri` entries of message items to inclute full
    path resolution of `resolved_path`.
    """
    for k, v in message.items():
        if isinstance(v, list):
            for item in v:
                try:
                    uri = item['uri']
                except KeyError:
                    break
                else:
                    item['uri'] = (resolved_path / uri).as_posix()
    return message


# =============================================================================
# WhatsappConversation class
# =============================================================================

class WhatsappConversation(Conversation):

    @classmethod
    def from_whatsapp_txt(cls, dir_path: PathObj, locale: str)-> Self:
        """
        Load a WhatsappConversation from an archive saved in text format.

        The resulting conversation messages are sorted by post date.

        Notes
        -----
        There are several limitations to loading a Whatsapp conversation from
        text:
        * The timestamps are limited to minute resolution
        * It is not possible to recover reactions to messages
        * Multiple media published at once appear as distinct messages
        * The text file format is dependent on the language, hence data loading
          might not be adapted to your language

        Parameters
        ----------
        dir_path : PathObj
            Path to the directory containing the archive. This directory
            contains saved media along with the conversation as a text file.
        locale : str
            Locale code of the whatsapp application from which the archive was
            created. The code is in the form language-COUNTRY, eg en-US, de-DE.

        """
        dir_path = Path(dir_path).resolve()
        name = dir_path.stem
        title = dir_path.stem

        date_regex = DATE_REGEX[locale]
        date_fmt = DATE_FORMAT[locale]

        raw_messages = []
        root, dirs, files = next(os.walk(dir_path))
        msg_files = [dir_path / f for f in files if f.split('.')[-1] == 'txt']
        for file in msg_files:
            with open(file, 'rt', encoding='utf-8') as f:
                file = f.read()

            # raw_msgs = []
            dt = 0
            start, stop = 0, 0
            for date in date_regex.finditer(file):
                # Current message
                stop = date.start()
                raw_messages.append((dt, file[start:stop]))
                # Next message
                start = date.end()
                dt = datetime.strptime(date.group(), date_fmt).timestamp()
            raw_messages.append((dt, file[start:]))


        msgs, participants = _parse_raw_messages(raw_messages, locale)
        messages = [Message(**_resolved_uri(m, dir_path)) for m in msgs]
        # Remove duplicates and sort messages by post date
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

