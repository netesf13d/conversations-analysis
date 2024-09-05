# -*- coding: utf-8 -*-
"""
This module implements the core Message and Conversation management
functionality.

It defines:
- Items, NamedTuples holding information about the different media or other
  shared content. These are:
  <Reaction>, <Photo>, <Video>, <Audio>, <File>, <Shared>, <Gif>.
- <MediaInfo>, NamedTuple holding all the relevant information relative to
  a shared media. Can be used to scrape the conversation media for instance.
- <Message>, the container class for messages.
- <Conversation>, the base class representing a conversation, essentially as
  a list of <Message>s. It implements the methods for data extraction and basic
  analysis (eg messages filtration).

Classes from this module are not meant to be instanciated directly.
<Conversation>s should be instanciated from the appropriate subclass, either
<MessengerConversation> or <WhatsappConversation>. <Message>s should be
instanciated by a <Conversation>.
"""

import json
import re
from collections import Counter
from datetime import datetime
from numbers import Real
from typing import NamedTuple, Self

import numpy as np

from ..utils import strip_diacritics, CRLF_REGEX, PUNCTUATION_REGEX, PathObj


# =============================================================================
# Message components classes
# =============================================================================

class Reaction(NamedTuple):
    """Reaction to message's `content`."""
    reaction: str
    actor: str

class Photo(NamedTuple):
    uri: str
    timestamp: float # seconds

class Video(NamedTuple):
    uri: str
    timestamp: float

class Audio(NamedTuple):
    uri: str
    timestamp: float

class File(NamedTuple):
    uri: str
    timestamp: float

class Shared(NamedTuple):
    """Shared link."""
    link: str = ""
    text: str = ""

class Gif(NamedTuple):
    uri: str


class MediaInfo(NamedTuple):
    """Container class for relevant media information"""
    media_type: str
    conv_name: str
    sender: str
    uri: str
    timestamp: float


Item = Reaction | Photo | Audio | Video | File | Shared | Gif


def _load_item(item: list[dict] | dict | None,
               item_cls: Item)-> list[Item] | Item | None:
    """
    Unpack a dict/list[dict] into the corresponding Item/list[Item].

    Parameters
    ----------
    item : list[dict] | dict | None
        DESCRIPTION.
        In some cases, the item elements contain additional keys that do
        not appear as attributes of the corresponding Item class. These are
        considered irrelevant and are not passed to the initializer.
        An example is the 'video' item of old conversations, which contains
        a 'thumbnail' -> dict key.
    item_cls : Item
        The corresponding item class (eg Photo, Video, etc).

    """
    if item is None:
        if item_cls is Shared:
            return None
        else:
            return []
    if isinstance(item, list):
        try:
            return [item_cls(**i) for i in item]
        except TypeError: # irrelevant keys are present: remove them
            item = [{k: i[k] for k in item_cls._fields} for i in item]
            return [item_cls(**i) for i in item]
    if isinstance(item, dict):
        return item_cls(**item)
    raise TypeError(f"incorrect type for item: {type(item)}")



# =============================================================================
# Message class
# =============================================================================

class Message():
    """
    Container class for conversation messages.

    Notes
    -----
    * General audio and voicenote messages are not distinguished.
    * Equality is implemented by comparison of all attributes.
    * Comparison operators are implemented through comparison of the tuple
      (timestamp, sender).

    Attributes
    ----------
    # Mandatory
    - sender : str, The message sender
    - timestamp : float, the message creation timestamp in seconds
    # Optional
    - content : str, the message's text content
    - reactions : list[Reaction], reactions to the message
    - photos : list[Photo], shared photos
    - videos : list[Video], shared videos
    - audio : list[Audio], shared audio
    - files : list[Files], shared files
    - gifs : list[Gif], gifs
    - shared : list[Shared], shared link
    # Properties
    - participants : set[str], participant to the message, that is the sender
                     and reaction actors

    Methods
    -------
    - asdict() : dict, return the message as a dict
    - replace_participants(participant_map) :
    - date() : datetime.datetime, return the creation date
    - words() : list[str], the words in message's content
    - nb_words() : int, number of words in message's content
    - nb_chars() : int, number of characters in message's content

    """

    __slots__ = (
        'sender',
        'timestamp',
        'content',
        'reactions',
        'photos',
        'videos',
        'audio',
        'files',
        'gifs',
        'shared',
        # 'sticker', # ignore
        )


    def __init__(self,
                 sender: str,
                 timestamp: float,
                 *,
                 content: str = '',
                 reactions: list[dict] | None = None,
                 photos: list[dict] | None = None,
                 videos: list[dict] | None = None,
                 audio: list[dict] | None = None,
                 files: list[dict] | None = None,
                 gifs: list[dict] | None = None,
                 shared: dict | None = None,
                 **ignore_kw)-> None:
        """


        Notes
        -----
        * A message with no content has content == ''
        """
        # Mandatory attributes
        self.sender: str = sender
        self.timestamp: float = timestamp # seconds
        # Non-mandatory attributes
        self.content: str = content
        self.reactions: list[Reaction] = _load_item(reactions, Reaction)
        self.photos: list[Photo] = _load_item(photos, Photo)
        self.videos: list[Video] = _load_item(videos, Video)
        self.audio: list[Audio] = _load_item(audio, Audio)
        self.files: list[File] = _load_item(files, File)
        self.gifs: list[Gif] = _load_item(gifs, Gif)
        self.shared: Shared | None = _load_item(shared, Shared)


    def __repr__(self)-> str:
        cls_name = self.__class__.__name__
        space = ' ' * len(cls_name)
        repr_ = (f"{cls_name}(sender='{self.sender}',\n"
                 f"{space} timestamp={self.timestamp}")
        for attr in self.__slots__[2:]:
            if (val:=getattr(self, attr)):
                if isinstance(val, list):
                    val = [v._asdict() for v in val]
                elif isinstance(val, str):
                    val = repr(val)
                elif isinstance(val, Item):
                    val = val._asdict()
                repr_ += f",\n        {attr}={val}"
        repr_ += ")"
        return repr_


    def __eq__(self, other: Self)-> bool:
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in self.__slots__)

    def __ne__(self, other: Self)-> bool:
        return any(getattr(self, attr) != getattr(other, attr)
                   for attr in self.__slots__)

    def __lt__(self, other: Self)-> bool:
        return ((self.timestamp, self.sender)
                < (other.timestamp, other.sender))

    def __le__(self, other: Self)-> bool:
        return ((self.timestamp, self.sender)
                <= (other.timestamp, other.sender))

    def __gt__(self, other: Self)-> bool:
        return ((self.timestamp, self.sender)
                > (other.timestamp, other.sender))

    def __ge__(self, other: Self)-> bool:
        return ((self.timestamp, self.sender)
                >= (other.timestamp, other.sender))


    def asdict(self)-> dict:
        """
        Convert Message to dict.
        """
        dict_ = {}
        for attr in self.__slots__:
            if (val:=getattr(self, attr)):
                if isinstance(val, list):
                    dict_[attr] = [v._asdict() for v in val]
                elif isinstance(val, (str, Real)):
                    dict_[attr] = val
                elif isinstance(val, Item):
                    dict_[attr] = val._asdict()
        return dict_


    def replace_participants(self, participant_map: dict[str, str])-> None:
        """
        Replace the participant names in the message.
        """
        self.sender = participant_map[self.sender]
        if self.reactions is not None:
            self.reactions = [Reaction(r.reaction, participant_map[r.actor])
                              for r in self.reactions]


    @property
    def participants(self)-> set[str]:
        """
        Participants to the message: sender and reaction actors.
        """
        if self.reactions is not None:
            return {self.sender} | {r.actor for r in self.reactions}
        return {self.sender}


    def date(self)-> datetime:
        """
        Message timestamp converted to a datetime.datetime object.
        """
        return datetime.fromtimestamp(self.timestamp)


    def words(self)-> list[str]:
        """
        Return the words of the message's `content` as a list.
        """
        words = CRLF_REGEX.sub(' ', self.content)
        words = PUNCTUATION_REGEX.sub(' ', words)
        words = [w for w in words.split(' ') if w]
        return words


    def nb_words(self)-> int:
        """
        Number of words in the message's `content`.
        """
        return len(self.words())


    def nb_chars(self)-> int:
        """
        Number of characters in the message's `content`.
        """
        return len(self.content)


# =============================================================================
# Conversation class
# =============================================================================

class Conversation():
    """
    Class representing a conversation.

    This class is basically a list of <Message> with additional methods to
    filter messages, extract data and extract relevant information.

    Notes
    -----
    * General audio and voicenote messages are not distinguished.
    * Comparison operators are implemented through comparison of the tuple
      (timestamp, sender)

    Attributes
    ----------
    - name : str, The conversation name
    - Title : str, the conversation title
    - messages : list[Message], the messages in the conversation
    - participants : dict[str, int], participants to the conversation
        the mapping is participant -> alphabetic ordinal

    Class methods
    -------------
    - from_file(file) : Load a conversation previously saved in JSON.

    Methods
    -------
    - asdict() -> dict
        return the conversation as a dict
    - replace_participants(participant_map) -> None
        replace the conversation participants names
    - filter_by(timeframe, participants, items, content_pattern) -> Conversation,
        return a Conversation with filtered messages
    - media_info(media_types) -> list[MediaInfo]
        return informations on the conversation media files
    - message_data() -> dict[str, np.ndarray],
        return messages data statistics
    - reactions_data() -> dict[str, np.ndarray]
        return reactions data statistics
    - word_count(groups, casefold, remove_diacritics) -> dict[str, Counter]
        return the word counts in the conversation for each group
    - match_pattern(pattern, casefold, remove_diacritics) -> list[Message]
       return messages whose content matches the given pattern
    """

    media_types = {'photos', 'videos', 'audio', 'files', 'gifs'}

    def __init__(self,
                  messages: list[Message],
                  participants: set[str],
                  name: str = '',
                  title: str = '')-> None:
        """
        Create a Conversation from:
            - A list of <Message>
            - The set of participants
            - The name of the conversation
            - The title of the conversation
        """
        self.name: str = name
        self.title: str = title
        self.messages: list[Message] = messages
        self._participants: set[str] = participants


    @classmethod
    def from_file(cls, file: PathObj)-> Self:
        """
        Instanciate a Conversation from a previous instance exported as a JSON
        file.

        Parameters
        ----------
        file : PathObj
            The JSON file to load.

        Returns
        -------
        Conversation
            The conversation.

        Examples
        --------
        >>> conv_dict = conversation.asdict()
        >>> conv_file = "./conversation.json"
        >>> with open(conv_file, 'wt', encoding='utf-8') as f:
                json.dump(conv_dict, f, indent=2, ensure_ascii=True)
        >>> conversation = Conversation.from_file(conv_file)

        """
        with open(file, 'rt', encoding='utf-8') as f:
            conv_dict = json.load(f)
        messages = [Message(**m) for m in conv_dict['messages']]
        participants = set(conv_dict['participants'])
        name = conv_dict['name']
        title = conv_dict['title']
        return cls(messages, participants, name, title)


    def __repr__(self)-> str:
        if len(self.messages) < 5:
            msgs = repr(self.messages)
        else:
            msgs = (repr(self.messages[:2])[:-1]
                    + ",\n ...\n "
                    + repr(self.messages[-2:])[1:])
        repr_ = (f"{self.__class__.__name__}(\n"
                 f"    messages={msgs},\n"
                 f"    participants={self._participants},\n"
                 f"    name={repr(self.name)},\n"
                 f"    title={repr(self.title)})")
        return repr_


    def __len__(self)-> int:
        return len(self.messages)


    def asdict(self)-> dict:
        """
        Convert to dict. Suitable for JSON export.
        """
        dict_ = {
            'name': self.name,
            'title': self.title,
            'participants': [p for p in self.participants],
            'messages': [m.asdict() for m in self.messages],
            }
        return dict_


    @property
    def participants(self)-> dict[str, int]:
        return {p: i for i, p in enumerate(sorted(self._participants))}


    def replace_participants(self, participant_map: dict[str, str])-> None:
        """
        Replace the participants in the conversation according to
        `participant_map`.
        """
        # Raise KeyError on misspecified participant to replace
        non_part = [p for p in participant_map if p not in self._participants]
        if non_part:
            raise KeyError(f"replacing inexistent participant(s): {non_part}")
        # replace participants in messages
        pmap = {p: participant_map[p] if p in participant_map else p
                for p in self._participants}
        for m in self.messages:
            m.replace_participants(pmap)
        # update participants
        self._participants = set(pmap.values())


    def filter_by(self, /,
                  timeframe: tuple[datetime, datetime] = None,
                  participants: set[str] | None = None,
                  items: set[str] | None = None,
                  content_pattern: str = '')-> Self:
        """
        Filter the conversation messages.

        Parameters
        ----------
        timeframe : tuple[datetime, datetime], optional
            Pair of datetime.datetime t0, t1. Select messages such that post
            date `timestamp` verifies t0 <= timestamp < t1.
            The default is None, which selects all messages.
        participants : set[str] | None, optional
            Select messages which `sender` attribute is in `participants`.
            The default is None, which selects all participants.
        items : set[str] | None, optional
            Select messages if they have at least one non-empty attribute among
            `items` (eg photos, videos, etc).
            The default is None, which selects all messages.
        content_pattern : str, optional
            Select messages if their `content` matches a regex pattern.
            The default is '', which selects all messages.

        Raises
        ------
        KeyError/AttributeError
            If a filter contains a misspecified element.
            - non-existent participant
            - non-existent Message item

        Returns
        -------
        Conversation
            A Conversation object encapsulating the filtered messages.

        """
        messages = self.messages
        if timeframe is not None:
            t0, t1 = timeframe[0].timestamp(), timeframe[1].timestamp()
            messages = [m for m in messages if
                        (t0 <= m.timestamp and m.timestamp < t1)]
        if participants is not None:
            non_part = [p for p in participants if p not in self._participants]
            if non_part:
                raise KeyError(f"inexistent participant(s): {non_part}")
            messages = [m for m in messages if (m.sender in participants)]
        if items is not None:
            non_items = [p for p in items if p not in Message.__slots__]
            if non_items:
                raise KeyError(f"inexistent item(s): {non_items}")
            messages = [m for m in messages
                        if any(getattr(m, i) for i in items)]
        if content_pattern:
            messages = [m for m in messages
                        if re.search(content_pattern, m.content)]

        participants = set.union(set(), *[m.participants for m in messages])

        return self.__class__(messages, participants,
                              'filt_' + self.name, self.title)


    def media_info(self, media_types: list[str])-> list[MediaInfo]:
        """
        Return media information on all the conversation media in media_types.

        The media information structured as MediaInfo instances, with the
        following attributes:
          - media_type: str, the media type (photo, video, audio, file, gif)
          - conv_name: str, the name of the conversation
          - sender: str, participant who published the file
          - uri: str, resolved path to the file
          - timestamp: float, file publication timestamp
        """
        if any(mt not in self.media_types for mt in media_types):
            raise ValueError(f"`media_types` must be in {self.media_types}")
        media_info = []
        for media_type in media_types:
            for msg in self.messages:
                if (media:=getattr(msg, media_type)) is not None:
                    for m in media:
                        mi = MediaInfo(media_type.rstrip('s'),
                                       self.name,
                                       msg.sender,
                                       m.uri,
                                       m.timestamp)
                        media_info.append(mi)
        return media_info


    def messages_data(self)-> dict[str, np.ndarray]:
        """
        Returns data pertaining to the messages' `content`, that is, the
        written text.

        The returned data are:
            - `timestamp`: int, timestamp of the message post date
            - `sender`: str, name of the message sender

            - `has_content`: int, 1 if message has a content else 0
            - `nb_words`: int, number of words in the message's content
            - `nb_chars`: int, number of characters in the message's content

            - `nb_reactions`: int, number of reactions to message

            - `nb_photos`: int, number of photos in message
            - `nb_videos`: int, number of videos in message
            - `nb_audio`: int, number of audio files message
            - `has_share`: int, 1 if message has a content else 0
            - `has_gif`: int, 1 if message has a content else 0

        """
        data = {
            'timestamp': np.array([m.timestamp for m in self.messages],
                                  dtype=np.float64),
            'sender': np.array([m.sender for m in self.messages]),
            # content-related
            'has_content': np.array([int(bool(m.content)) for m in self.messages]),
            'nb_words': np.array([m.nb_words() for m in self.messages]),
            'nb_chars': np.array([m.nb_chars() for m in self.messages]),
            # reactions-related
            'nb_reactions': np.array([len(m.reactions) for m in self.messages]),
            # media-related
            'nb_photos': np.array([len(m.photos) for m in self.messages]),
            'nb_videos': np.array([len(m.videos) for m in self.messages]),
            'nb_audio': np.array([len(m.audio) for m in self.messages]),
            'nb_gifs': np.array([len(m.gifs) for m in self.messages]),
            'has_shared': np.array([int(bool(m.shared)) for m in self.messages]),
            }
        return data


    def reactions_data(self)-> dict[str, np.ndarray]:
        """
        Returns data pertaining to the messages' `reactions`.

        The returned data are:
            - `timestamp`, timestamp of the message post date
            - `sender`, name of the message sender
            - `actor`, name of the reaction actor
            - `reaction`, the reaction
        """
        messages = [m for m in self.messages if m.reactions]
        data = {
            'timestamp': np.array([m.timestamp for m in messages
                                                     for r in m.reactions],
                                  dtype=np.float64),
            'sender': np.array([m.sender for m in messages
                                              for r in m.reactions]),
            'actor': np.array([r.actor for m in messages
                                       for r in m.reactions]),
            'has_reaction': np.array([1 for m in messages
                                        for r in m.reactions]),
            'reaction': np.array([r.reaction for m in messages
                                             for r in m.reactions])
            }
        return data


    def word_counts(self,
                    groups: dict[str, str] | None = None,
                    casefold: bool = True,
                    remove_diacritics: bool = True)-> dict[str, Counter]:
        """
        For each group of participants, returns the distinct words in the
        conversation messages sent by the group and their count.

        Parameters
        ----------
        groups : dict[str, str] | None, optional
            Map participant -> group.
            The default is None, which selects all participants individually.
        casefold : bool, optional
            Casefold the words before counting. The default is True.
        remove_diacritics : bool, optional
            Remove diacritics from the words before counting.
            The default is True.

        Returns
        -------
        dict[str, Counter]
            For each group, a collections.Counter object representing the
            distinct words and their count.

        """
        if groups is None:
            groups = {}
        groups |= {k: k for k in self.participants if k not in groups.keys()}
        #
        wd_count = {g: Counter() for g in groups.values()}
        for p in self.participants:
            # Turn into one long message for more efficient processing
            words = ' '.join(m.content for m in self.messages if m.sender == p)
            words = CRLF_REGEX.sub(' ', words) # remove CR/LF
            words = PUNCTUATION_REGEX.sub(' ', words) # remove punctuation
            if casefold:
                words = words.casefold()
            if remove_diacritics:
                words = strip_diacritics(words)
            wd_count[groups[p]].update(w for w in words.split(' ') if w)
        return wd_count


    def match_pattern(self,
                      pattern: str,
                      casefold: bool = True,
                      remove_diacritics: bool = True)-> Self:
        """
        Return the messages which content contains a regex pattern.

        Parameters
        ----------
        pattern : str
            The pattern to serach for. Must be a valid regular expression.
        casefold : bool, optional
            Casefold the words before searching the pattern.
            The default is True.
        remove_diacritics : bool, optional
            Remove diacritics from the words before searching the pattern.
            The default is True.

        """
        messages = []
        for m in self.messages:
            if (msg:=m.content):
                if casefold:
                    msg = msg.casefold()
                if remove_diacritics:
                    msg = strip_diacritics(msg)
                if re.search(pattern, msg):
                    messages.append(m)
        return self.__class__(messages, self._participants,
                              'filt_' + self.name, self.title)

