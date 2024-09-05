# -*- coding: utf-8 -*-
"""
This module implements some functionality for conversation data analysis.

It exposes the following functionality:
    - <ConversationStats>
      Wrapper around a pandas.DataFrame which implements convenience methods
      for data analysis:
          - A sum
          - A binned sum
          - A rolling sum
    - <word_count_dataframe>
      Funtion to structure word counts as a pandas.DataFrame.

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
    Thin wrapped around a <pandas.DataFrame> that implements convenience
    methods for its processing and manipulation.

    Each row of the dataframe represents an event indexed by its date. It has
    the following structure:
    - It is indexed by a MultiIndex representing the date with the following
      levels: (year, month, day, hour, minute, second, microsecond,
               month_name, day_name, timestamp)
    - The columns represent the data. The first column corresponds to the
      group related to the event (eg the message sender). The remaining
      columns must contain numerical data (eg the message length, the presence
      of photos, etc).

    Attributes
    ----------
    - dataframe : pandas.DataFrame
    - groups : dict[str, str], mapping g -> g for all groups present in the
      dataframe

    Methods
    -------
    - get_dataframe(timespan, new_groups) : pandas.DataFrame
        Extract the rows within the `timespan` and map the groups according
        to `new_groups`.
    - sum(groups, timespan) : pandas.DataFrame
        For each group, sum along the dataframe columns.
    - binned_sum(binning_entries, groups, timespan) : pandas.DataFrame
        For each unique indices values in `binning_entries` and each group,
        sum along the dataframe columns.
    - rolling_sum(sampling_freq, window_size, win_type, win_args,
                  period, groups, timespan) : pandas.DataFrame
        Rolling sum along the columns, possibly aggregating the data modulo a
        given period.
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
                      new_groups: dict[str, str] | None = None,
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

        if new_groups is not None:
            df.replace({'_group': new_groups}, inplace=True)

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
        Compute the sum of the columns in the given `timespan`, grouping both
        by group and by the unique values of the indices corresponding to
        `binning_entries`.

        Returns a DataFrame (binning_entries x (group, entry)) of the sums.

        Parameters
        ----------
        binning_entries : tuple[str, ...]
            Tuple of dataframe multiindex names. The values must be among
            {'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond',
             'month_name', 'day_name'}.
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
                    sampling_freq: pd.Timedelta | str,
                    window_size: int,
                    win_type: str | None = None,
                    win_args: dict | None = None,
                    *,
                    period: str | None = None,
                    groups: dict[str, str] | None = None,
                    timespan: tuple | None = None,
                    )-> pd.DataFrame:
        """
        Compute the windowed rolling sum of the columns for each group in the
        given timespan, with atomic timestep set by `sampling_freq`.

        If a `period` is specified, the rolling sum is computed over the data
        aggregated modulo that period.

        Returns a DataFrame (time_multiindex x (group, entry)) of the sums.
        The MultiIndex time_multiindex depends on whether `period` is set:
        - If set, levels are the components of a Timedelta index:
          (days, hours, minutes, seconds, milliseconds, timedelta)
        - If not set, levels are the components of a Timestamp index:
          (year, month, day, hour, minute, second, microsecond, timestamp)

        Parameters
        ----------
        sampling_freq : pandas.Timedelta | str
            The sampling frequency expressed as a Timedelta. It represents the
            granularity of the rolling sum, the atomic timestep.
            The value is converted internally into a pandas.Timedelta, hence
            it accepts all types of parameters that can be passed to the
            constructor.
        window_size : int
            The window size, in units of sampling_freq. See DataFrame.rolling
            documentation.
        win_type : str | None, optional
            The window type. See DataFrame.rolling documentation.
            The default is None.
        win_args : dict | None, optional
            Additional window parameters. See DataFrame.rolling documentation.
            The default is None.
        period : str | None, optional
            Must be in {'year', 'month', 'day', 'hour', 'minute', 'second'}.
            Aggregation period for the data, before computing the rolling sum.
            For instance, setting period='day' will yield a rolling sum along
            the day time.
            The default is None.
        groups : dict[str, str] | None, optional
            Mapping old_group -> new_group applied to the '_group' column.
            The default is None (no mapping).
        timespan : tuple | None, pair (t0, t1) of tuple[int, ...]
            Time boundaries for data selection (see <get_dataframe> doc).
            The default is None, which selects all data.

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

