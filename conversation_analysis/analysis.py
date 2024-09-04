# -*- coding: utf-8 -*-
"""
TODO doc
Functionality for conversation analysis

It exposes the following functionality:
    - <ConversationStats>
      .
    - <word_count_dataframe>
      .

"""

from collections import Counter
from copy import copy

import numpy as np
import pandas as pd


# =============================================================================
# Utils
# =============================================================================

def datetime_multiindex(index: pd.DatetimeIndex)-> pd.MultiIndex:
    """
    Decompose a DatetimeIndex into a components MultiIndex:
        year, month, day, hour, minute, second, microsecond,
        month_name, day_name, timestamp

    The original index is kept as timestamp.
    """
    idx_arr = [index.year, index.month, index.day,
               index.hour, index.minute, index.second,
               index.microsecond,
               index.strftime('%m - %b'), index.strftime('%w - %a'), index]
    names = ['year', 'month', 'day',
             'hour', 'minute', 'second',
             'microsecond',
             'month_name', 'day_name', 'timestamp']
    idx = pd.MultiIndex.from_arrays(idx_arr, names=names)
    # idx.name = 'date'
    return idx



class ConversationStats:
    """
    TODO doc

    General container for conversation messages.

    Notes
    -----
    * General audio and voicenote messages are not distinguished.
    * Comparison operators are implemented through comparison of the tuple
      (timestamp, sender)

    Attributes
    ----------
    - dataframe : pd.DataFrame
        ,,
    - participants : set[str]
        participant to the message, that is the sender
                     and reaction actors

    Methods
    -------
    - get_dataframe(timespan, groups) :
    - sum() :
    - binned_sum(participant_map) :
    - rolling_sum() :
    """

    def __init__(self,
                 timestamp: np.ndarray,
                 group: np.ndarray,
                 data: dict[str, np.ndarray],
                 )-> None:
        """
        Instanciate a ConversationStats from a list of events.

        Events comprise three informations:
        - The timestamp of the event (eg a message post date)
        - The group to which the event is related (eg the author of a message)
        - Data relative to the event (eg the number of words in the message)
        Events are structured as a DataFrame organized as:
        datetime multiindex | _group  col1  col2 ...

        Parameters
        ----------
        timestamp : np.ndarray[float]
            Timestamps of the events expressed in seconds.
        group : np.ndarray[str]
            The group to which the event is related. The data analysis methods
            allow for mapping and grouping by the elements of this group.
        data : dict[str, np.ndarray]
            A mapping name -> data. Each entry is represented as a column.
            Data should contain only numeric values, which can be summed
            meaningly.

        """
        self.groups = {p: p for p in np.unique(group)}
        self.dataframe = self._build_dataframe(timestamp, group, data)


    @staticmethod
    def _build_dataframe(timestamp: np.ndarray,
                         group: np.ndarray,
                         data: dict[str, np.ndarray])-> pd.DataFrame:
        """
        DataFrame construction function.
        - Create a datetime multiindex
        - Set the group as the first column ('_group')
        - Set the remaining data as the subsequent columns
        """
        index = datetime_multiindex(pd.to_datetime(timestamp, unit='s'))
        data = copy(data)
        data.pop('_group', None)
        data = {'_group': group} | data
        return pd.DataFrame(data=data, index=index).sort_index()


    def get_dataframe(self,
                      timespan: tuple | None,
                      groups: dict[str, str] | None = None,
                      )-> pd.DataFrame:
        """
        Extract the data in a given timespan and map the '_group' column values
        according to `groups`.

        Parameters
        ----------
        timespan : tuple | None, pair (t0, t1) of tuple[int, ...]
            (year, month, day, hour, minute, second, microsecond) that define
            timestamp boundaries for data selection.
            Data is selected such that t0 <= t < t1. If the tuples are shorter,
            the missing values are assumed minimal. For instance (2020, 6)
            corresponds to (2020, 6, 1, 0, 0, 0, 0).
            The default is None, which selects all the data.
        groups : dict[str, str] | None, optional
            Mapping old_group -> new_group applied to the '_group' column.
            If specified, it must not miss any group.
            The default is None, which keeps original groups.

        """
        if timespan is None:
            df = self.dataframe.copy()
        else:
            df = self.dataframe.loc[timespan[0]:timespan[1], :].copy()

        if groups is not None:
            df.replace({'_group': groups}, inplace=True)

        return df


    def sum(self, *,
            groups: dict[str, str] | None = None,
            timespan: tuple | None = None,
            )-> pd.DataFrame:
        """
        Compute the columns sum in the given `timespan`, grouped by `groups`.
        (see <get_dataframe> documentation)

        Returns a DataFrame (group x entry) of the sums.
        """
        df = self.get_dataframe(timespan, groups)
        return df.groupby("_group").sum()


    def binned_sum(self,
                   binning_entries: tuple[str, ...],
                   *,
                   groups: dict[str, str] | None = None,
                   timespan: tuple | None = None,
                   )-> pd.DataFrame:
        """
        Return binned sum in the given `timespan`, aggregating the senders
        as `groups`.
        TODO doc

        Parameters
        ----------
        binning_entries : tuple[str, ...]
            DESCRIPTION.
        groups : dict[str, str] | None, optional
            Mapping old_group -> new_group applied to the '_group' column.
            The default is None (no mapping).
        timespan : tuple | None, pair (t0, t1) of tuple[int, ...]
            Time boundaries for data selection (see <get_dataframe> doc).
            The default is None, which selects all data.

        """
        df = self.get_dataframe(timespan, groups)

        # Get index after grouping by `binning_entries` to reindex dataframe
        levels = [df.index.names.index(b) for b in binning_entries]
        # If '_group' column is not dropped there is an error
        # TODO pandas select multilevel values on multiindex
        idx = df.drop("_group", axis=1, inplace=False).groupby(level=levels)
        idx = idx.any().index.unique()
        #
        data = {}
        for name, group in df.groupby("_group"):
            group.drop("_group", axis=1, inplace=True)
            g = group.groupby(level=levels, sort=True).sum()
            g = g.reindex(idx, fill_value=0)
            for col in g.columns:
                data[(col, name)] = g[col].to_numpy(dtype=int)
        # construct bin count DataFrame
        gdf = {'index': g.index,
               'columns': (cols:=sorted(data.keys())),
               'data': np.array([data[c] for c in cols]).T,
               'index_names': idx.names,
               'column_names': ['quantity', 'group']}
        gdf = pd.DataFrame.from_dict(gdf, orient='tight')

        return gdf


    def rolling_sum(self,
                    sampling_freq: str,
                    window_size: int,
                    win_type: str | None = None,
                    win_args: dict | None = None,
                    *,
                    period: str | None = None,
                    groups: dict[str, str] | None = None,
                    timespan: tuple | None = None,
                    )-> pd.DataFrame:
        """
        TODO doc

        Parameters
        ----------
        sampling_freq : str
            DESCRIPTION.
        window_size : int
            DESCRIPTION.
        win_type : str | None, optional
            DESCRIPTION. The default is None.
        win_args : dict | None, optional
            DESCRIPTION. The default is None.
        period : str | None, optional
            DESCRIPTION. The default is None.
        groups : dict[str, str] | None, optional
            Mapping old_group -> new_group applied to the '_group' column.
            The default is None (no mapping).
        timespan : tuple | None, pair (t0, t1) of tuple[int, ...]
            Time boundaries for data selection (see <get_dataframe> doc).
            The default is None, which selects all data.

        Returns
        -------
        gdf : DataFrame
            DESCRIPTION.

        """
        if win_args is None:
            win_args = {}

        df = self.get_dataframe(timespan, groups)

        ## Preprocess dataframe indexation and apply window
        if period is None:
            idx_names = []
        else:
            if period in {'microsecond', 'month_name', 'day_name', 'timestamp'}:
                raise ValueError(f"aggregation impossible over `{period}`")
            idx_names = df.index.names[:df.index.names.index(period)+1]
        t = df.index.get_level_values('timestamp') # .floor(sampling_freq)
        tref = pd.DataFrame({'year': 1970, 'month': 1, 'day': 1},
                            index=range(len(t)))
        tref = tref.assign(**{n: getattr(t, n) for n in idx_names})
        tref = pd.to_datetime(tref).astype('int64')
        dt = pd.TimedeltaIndex(t.astype('int64') - tref)
        dt += pd.to_timedelta(sampling_freq) // 2 # center bins
        df.index = dt.floor(sampling_freq)

        ## Set rolling indexation and final indexation
        idx = pd.timedelta_range(min(df.index), max(df.index),
                                 freq=sampling_freq)
        i = idx.astype('int64')
        if period is None: # datetime
            gdf_idx = pd.DatetimeIndex(i)
            gdf_idx = datetime_multiindex(gdf_idx)
            gdf_idx = gdf_idx.droplevel(['day_name', 'month_name'])
            gdf_idx = gdf_idx.set_levels(i, level=-1)
        else: # timedelta
            gdf_idx = idx.components.iloc[:, :5]
            gdf_idx['timedelta'] = i
            gdf_idx = pd.MultiIndex.from_frame(gdf_idx)

        ## compute rolling count
        # if groups is not None:
        #     df.replace({'_group': self._groups_map(groups)}, inplace=True)
        data = {}
        for name, group in df.groupby("_group"):
            group.drop("_group", axis=1, inplace=True)
            g = group.resample(rule=sampling_freq).sum()
            g = g.reindex(idx, fill_value=0)
            g = g.rolling(window=window_size, min_periods=1, center=True,
                          win_type=win_type)
            g = g.mean(**win_args)
            for col in g.columns:
                data[(col, name)] = g[col].to_numpy()
        ## construct rolling count dataframe
        gdf = {'index': g.index,
               'columns': (cols:=sorted(data.keys())),
               'data': np.array([data[c] for c in cols]).T,
               'index_names': ['datetime'],
               'column_names': ['quantity', 'group']}
        gdf = pd.DataFrame.from_dict(gdf, orient='tight')
        gdf.index = gdf_idx

        return gdf


# =============================================================================
#
# =============================================================================

def word_count_dataframe(word_count: dict[str, Counter],
                         words: list[str])-> pd.DataFrame:
    """
    Structure raw word counts into a DataFrame (group x word).

    Parameters
    ----------
    word_count : dict[str, Counter]
        A mapping group -> Conter (word -> count) representing the distinct
        words and their count.
    words : list[str]
        The list of counted words. Useful if a word has zero count, since it
        won't appear in the word_count counters.

    """
    entries = {}
    for user, wd_count in word_count.items():
        entries[user] = [wd_count[w] for w in words]
    df = pd.DataFrame(data=entries, index=words).T
    df.index.name = 'group'
    df.columns.name = 'counts'
    return df

