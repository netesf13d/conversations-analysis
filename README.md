# conversations-analysis
Load and analyze conversations

This package implements functionality to analyze conversations. From file
loading to data vizualization.

It features:
* <Conversation> loading and instanciation from files (subpackage `conversations`)
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


## Usage

<p align="center">
    <img src="https://github.com/netesf13d/conversation-analysis/examples/figures and data/word_pc.png" width="300" />
</p>



## Dependencies

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)


## Notes

The typing annotations in the code are by no means rigorous. They are made to facilitate the understanding of the nature of various parameters.


