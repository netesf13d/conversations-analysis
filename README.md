# conversations-analysis

A package to analyze conversations, from file loading to data vizualization. You can get insights about the various participants activity, their usage of medias, words, reactions, and many more. The conversation manipulation and data analysis is essentially agnostic to which messaging application was used (apart from data import).

Messaging applications currently supported are:
* Facebook messenger archives both in JSON and HTML format. support is partial for HTML (media info is not loaded).
* Whatsapp archives in text format. These, however, suffer from several limitations:
  - Reactions are absent from the exported archive (a whatsapp feature)
  - Some components of text archives are locale dependent: date formatting, joined and missing files texts tokens. I set those I could, but yours is likely to be missing. Feel free to ask for an update.


## Detailed overview

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


## Example usage

For a thorough study of an example dummy conversation, two Jupyter notebooks are available [here](https://github.com/netesf13d/conversations-analysis/tree/main/examples), which can be easily adapted to your archives files. Nevertheless, using the package to get quick analytics report on the conversation is easy!

### Importing a conversation and exporting data for analysis

You can download your conversations archive from [https://accountscenter.facebook.com/info_and_permissions/dyi/](https://accountscenter.facebook.com/info_and_permissions/dyi/). Prefer the JSON format for a better support. After unzipping, your conversations are located in `facebook-<yourname><randomnumber>/your_activity_across_facebook/messages/inbox/`. Just provide the path to your conversation to instanciate a `MessenerConversation`.
```
from conversation_analysis import (MessengerConversation, ConversationStats,
                                   pie_plot, bar_plot, stack_plot)

# conv_paths is the list of paths where your conversation archive is located
# it may be distributed over multiple archive files, for example
conv_paths = ['facebook-user1234_archive2017/your_activity_across_facebook/messages/inbox/myconversation',
              'facebook-user1234_archive2018/your_activity_across_facebook/messages/inbox/myconversation']
conv = MessengerConversation.from_facebook_json(conv_paths)
messages_data = conv.messages_data()
```

Export messages statistics and use them to instanciate a `ConversationStats` object, which provides easy to use methods to compute various quantities as shown hereafter.
```
timestamp = messages_data.pop('timestamp')
group = messages_data.pop('sender')
messages_stats = ConversationStats(timestamp, group, data=messages_data)
```


### Getting global conversation stats

The `ConversationStats.sum` method, as its name suggests, sums the various quantities to get global statistics, suitable to plot in a pie chart.
```
df = messages_stats.sum() # pandas DataFrame
fig, axs = pie_plot(df)
```
<p align="center">
    <img src="https://github.com/netesf13d/conversations-analysis/blob/main/examples/figures_and_data/msg_pc_participant.png" width="600" />
</p>


### Getting hourly messages stats

The `ConversationStats.binned_sum` method does the same as the above, excepts that it first bins the data according to the selected post date binning entry. The example here shows binned data by `'hour'`, but it ccould also be by `'day_name'`, `'month_name'`, `'year'` and so on. Such data is suitably represented as a bar plot.
```
df = messages_stats.binned_sum(binning_entries=('hour',), groups=None, timespan=None)
fig, ax = bar_plot(df['has_content'])
```
<p align="center">
    <img src="https://github.com/netesf13d/conversations-analysis/blob/main/examples/figures_and_data/msg_bp_participants_hour.png" width="600" />
</p>


### Time evolution of the number of messages sent

The `ConversationStats.rolling_sum` method provides a more advanced processing of the data by windowing a rolling sum. This can be used to get fine grained info on the time evolution of various statistics, well represented with a stack plot.
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

Finally, you can compute other stats easily: reaction usage, media usage, who receives reactions, etc. The functions above can be easily adapted to any quatity that you can think of. For example, here is how to compute and plot word count statistics.
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


