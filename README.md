# conversations-analysis

This package implements functionality to analyze conversations. From file loading to data vizualization.


## Overview

The package features:
* Conversation loading and instanciation from files (subpackage `conversations`)
  - From Facebook Messenger archives. Support for both JSON and HTML archives, although it is only partial for HTML.
  - From Whatsapp text archives
* Conversation manipulation and data extraction (subpackage `conversations`)
  - Search in conversation messages, filter the messages
  - Extract media information
  - Extract raw conversation statistics
* Analysis of conversation data (module `analysis`)
  Centered around pandas dataframes manipulation. Some standard procedures are implemented.
  - The sum of the various entries
  - The time-binned sum of the various entries
  - The rolling sum of the various entries
* Data vizualization (module `plot`)
  Plotting functions adapted to the three different analysis functions.
  - Pie charts for the simple sums
  - Bar plots for binned sums
  - Stack plots for rolling sums


## Exeample usage

For a thorough study of an example dummy conversation, two Jupyter notebooks are available [here](https://github.com/netesf13d/conversations-analysis/tree/main/examples), which can be easily adapted to your archives files. Nevertheless, using the package to get quick analytics report on the conversation is easy!

### Importing a conversation and exporting data for analysis

```
from conversation_analysis import (MessengerConversation, ConversationStats,
                                   pie_plot, bar_plot, stack_plot)

# conv_paths is the list of paths where your conversation archives are located
conv = MessengerConversation.from_facebook_json(conv_paths)
messages_data = conv.messages_data()

timestamp = messages_data.pop('timestamp')
group = messages_data.pop('sender')
messages_stats = ConversationStats(timestamp, group, data=messages_data)
```

### Getting global conversation stats

```
df = messages_stats.sum() # pandas DataFrame
fig, axs = pie_plot(df)
```

<p align="center">
    <img src="https://github.com/netesf13d/conversations-analysis/blob/main/examples/figures_and_data/msg_pc_participant.png" width="600" />
</p>


### Getting hourly messages stats

```
df = messages_stats.binned_sum(binning_entries=('hour',), groups=None, timespan=None)
fig, ax = bar_plot(df['has_content'])
```
<p align="center">
    <img src="https://github.com/netesf13d/conversations-analysis/blob/main/examples/figures_and_data/msg_bp_participants_hour.png" width="600" />
</p>


### Time evolution of the number of messages sent

```
df = messages_stats.rolling_sum(sampling_freq='5D', window_size=10,
                                win_type='gaussian', win_args={'std': 2})
fig, ax = stack_plot(df['has_content'], baseline='wiggle',
                     timescale='day', xlabel_strftime='%Y-%m')
```
<p align="center">
    <img src="https://github.com/netesf13d/conversations-analysis/blob/main/examples/figures_and_data/msg_sp_participants_whole.png" width="600" />
</p>


### Computing word counts statistics

```
from conversation_analysis import word_count_dataframe

wd_counts = conv.word_counts(groups=None, casefold=True, remove_diacritics=True)
words = ['road', 'parameter', 'astronaut', 'media', 'strongly', 'call']
word_count_df = word_count_dataframe(wd_counts, words)
fig, axs = pie_plot(word_count_df)
```
<p align="center">
    <img src="https://github.com/netesf13d/conversations-analysis/blob/main/examples/figures_and_data/word_pc.png" width="600" />
</p>


## Dependencies

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)


## Notes

The typing annotations in the code are by no means rigorous. They are made to facilitate the understanding of the nature of various parameters.


